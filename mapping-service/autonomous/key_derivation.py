"""Bidirectional key derivation, declared as templates rather than regexes.

A mapping joins a source record to an inventory folder by deriving the same key
from both sides. Today each builder writes two independent regexes and relies on
a person to keep them consistent; nothing checks that they agree, and a
disagreement shows up as cases that silently fail to match.

Here both sides are declared as templates over named parts, and the key is a
selection of those parts put through the same normalisers. Agreement becomes a
property of the declaration instead of something to hope for, and because a
template both parses and formats, a derivation can be dry-run over real evidence
before any mapping is executed.

Templates are deliberately weaker than regex: literals and named parts only. A
capture rule says "folders are named EXE_<year>_<yy>-<number>-<code>", and that
sentence is what a template writes down.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


class DerivationError(RuntimeError):
    pass


PART = re.compile(r"\{(?P<name>[a-z][a-z0-9_]*)(?::(?P<kind>d|a|\*))?\}")

# Part kinds keep a template readable while still bounding what a part may
# swallow. The default stops at any literal character the template itself uses,
# so a template adapts to its own separators.
KIND_PATTERNS = {"d": r"\d+", "a": r"[A-Za-z0-9]+", "*": r".+"}

PAD = re.compile(r"^pad:(\d{1,2})$")

# The first five are the vocabulary the rest of the spec already uses for
# whole-field normalisation; the rest only make sense on an extracted part.
# They share one list so a compiler cannot pick a name that parses here but
# means nothing, or vice versa.
KNOWN_NORMALIZERS = frozenset(
    {
        "trim",
        "casefold",
        "collapse_space",
        "slash_to_hyphen",
        "alnum",
        "upper",
        "lower",
        "strip_zeros",
        "year2to4",
    }
)


def is_known_normalizer(name: str) -> bool:
    return name in KNOWN_NORMALIZERS or bool(PAD.match(name))


def normalise(value: str, normalizers: Sequence[str], *, year_pivot: int = 30) -> str:
    """Apply the declared normalisers, in order, to one extracted part."""
    result = value
    for name in normalizers:
        if name == "strip_zeros":
            digits = result.lstrip("0")
            result = digits or "0"
        elif name == "upper":
            result = result.upper()
        elif name in {"lower", "casefold"}:
            result = result.casefold()
        elif name == "collapse_space":
            result = re.sub(r"\s+", " ", result).strip()
        elif name == "slash_to_hyphen":
            result = result.replace("/", "-")
        elif name == "alnum":
            result = re.sub(r"[^0-9a-zA-Z]+", "", result)
        elif name == "trim":
            result = result.strip()
        elif name == "year2to4":
            # Idempotent by design. The same part name carries one set of
            # normalisers on both sides of a join, and a reference writes a
            # two-digit year where its folder writes four; refusing the
            # four-digit form would make one side unkeyable and the join silently
            # empty.
            if result.isdigit() and len(result) == 4:
                pass
            elif result.isdigit() and len(result) == 2:
                century = 19 if int(result) >= year_pivot else 20
                result = f"{century}{result}"
            else:
                raise DerivationError(
                    f"year2to4 needs a two- or four-digit year, got {result!r}"
                )
        elif match := PAD.match(name):
            result = result.rjust(int(match.group(1)), "0")
        else:
            raise DerivationError(f"Unknown normaliser {name!r}")
    return result


@dataclass(frozen=True)
class Template:
    """A literal-and-parts pattern that both parses and formats."""

    pattern: str
    match_mode: str = "exact"

    def __post_init__(self) -> None:
        if self.match_mode not in {"exact", "prefix"}:
            raise DerivationError(f"Unknown match_mode {self.match_mode!r}")
        if not PART.search(self.pattern):
            raise DerivationError(f"Template names no parts: {self.pattern!r}")
        object.__setattr__(self, "_regex", self._compile())

    @property
    def part_names(self) -> tuple[str, ...]:
        return tuple(match.group("name") for match in PART.finditer(self.pattern))

    def _compile(self) -> re.Pattern[str]:
        names = list(self.part_names)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            # Otherwise this surfaces as re.error about a redefined group, which
            # says nothing about what to write instead.
            raise DerivationError(
                f"Template {self.pattern!r} names part(s) {duplicates} more than once. "
                "Each part must have its own name; a four-digit year and the two-digit year "
                "inside it are two parts, not one."
            )
        out: list[str] = ["^"]
        position = 0
        for match in PART.finditer(self.pattern):
            out.append(re.escape(self.pattern[position : match.start()]))
            kind = match.group("kind")
            if kind:
                body = KIND_PATTERNS[kind]
            else:
                # An unqualified part runs up to the literal that follows it, so
                # "{yy}-{number}" splits on the hyphen instead of swallowing it.
                # A part with nothing after it takes the rest, which is why a
                # trailing part in prefix mode has to declare its kind.
                following = self.pattern[match.end() : match.end() + 1]
                body = f"[^{re.escape(following)}]+" if following else r".+"
            out.append(f"(?P<{match.group('name')}>{body})")
            position = match.end()
        out.append(re.escape(self.pattern[position:]))
        # In prefix mode anything may follow: Exeter appends a free-text address
        # to some folder names, so the derived key is a prefix of the folder.
        out.append(r"(?P<_tail>.*)$" if self.match_mode == "prefix" else "$")
        try:
            return re.compile("".join(out))
        except re.error as exc:
            raise DerivationError(f"Template {self.pattern!r} is not usable: {exc}") from exc

    def parse(self, text: str) -> dict[str, str] | None:
        match = getattr(self, "_regex").match((text or "").strip())
        if not match:
            return None
        parts = match.groupdict()
        parts.pop("_tail", None)
        return parts

    def format(self, parts: dict[str, str]) -> str:
        def replace(match: re.Match[str]) -> str:
            name = match.group("name")
            if name not in parts:
                raise DerivationError(f"Cannot format {self.pattern!r}: missing part {name!r}")
            return str(parts[name])

        return PART.sub(replace, self.pattern)


@dataclass(frozen=True)
class KeyDerivation:
    """One declaration that keys both sides of a join.

    Each side may declare alternatives, tried in order. Real references arrive in
    shape variants -- an optional classification segment, a trailing code that is
    sometimes absent -- and without alternatives those rows simply fail to key,
    which reads as "no scan exists" rather than "the template did not fit".
    """

    source_templates: tuple[Template, ...]
    inventory_templates: tuple[Template, ...]
    key_parts: tuple[str, ...]
    normalizers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    defaults: dict[str, str] = field(default_factory=dict)
    year_pivot: int = 30

    def __post_init__(self) -> None:
        if not self.key_parts:
            raise DerivationError("A derivation needs at least one key part")
        if not self.source_templates or not self.inventory_templates:
            raise DerivationError("Both sides need at least one template")
        # An unknown normaliser would otherwise make every key unbuildable, and
        # a key that never builds reads as "no case matched" rather than as the
        # configuration error it is.
        for part, names in self.normalizers.items():
            unknown = [name for name in names if not is_known_normalizer(name)]
            if unknown:
                raise DerivationError(
                    f"part {part!r} declares unknown normaliser(s) {unknown}; "
                    f"choose from {sorted(KNOWN_NORMALIZERS)} or pad:N"
                )
        for side, templates in (
            ("source", self.source_templates),
            ("inventory", self.inventory_templates),
        ):
            for template in templates:
                missing = [
                    name
                    for name in self.key_parts
                    if name not in template.part_names and name not in self.defaults
                ]
                if missing:
                    raise DerivationError(
                        f"{side} template {template.pattern!r} does not produce key part(s) "
                        f"{missing} and no default is declared; every alternative must be able "
                        "to build the whole key or the two sides cannot agree"
                    )

    @property
    def source_template(self) -> Template:
        return self.source_templates[0]

    @property
    def inventory_template(self) -> Template:
        return self.inventory_templates[0]

    @staticmethod
    def _first_parse(templates: Sequence[Template], value: str) -> dict[str, str] | None:
        for template in templates:
            parts = template.parse(value)
            if parts is not None:
                return parts
        return None

    def _key(self, parts: dict[str, str] | None) -> tuple[str, ...] | None:
        if parts is None:
            return None
        try:
            return tuple(
                normalise(
                    parts.get(name, self.defaults.get(name, "")),
                    self.normalizers.get(name, ()),
                    year_pivot=self.year_pivot,
                )
                for name in self.key_parts
            )
        except DerivationError:
            return None

    def source_key(self, value: str) -> tuple[str, ...] | None:
        return self._key(self._first_parse(self.source_templates, value))

    def inventory_key(self, value: str) -> tuple[str, ...] | None:
        return self._key(self._first_parse(self.inventory_templates, value))

    def explain(self, value: str, *, side: str) -> str:
        """Say why a value produced no key, for a spec that joins nothing."""
        templates = self.source_templates if side == "source" else self.inventory_templates
        parts = self._first_parse(templates, value)
        if parts is None:
            return f"no {side} template matched {value!r}"
        for name in self.key_parts:
            raw = parts.get(name, self.defaults.get(name, ""))
            try:
                normalise(raw, self.normalizers.get(name, ()), year_pivot=self.year_pivot)
            except DerivationError as exc:
                return (
                    f"{side} part {name}={raw!r} rejected its normalisers: {exc}. "
                    "A part name carries one set of normalisers on both sides, so the same name "
                    "must hold the same format on each; name the two differently if it does not."
                )
        return f"{side} value {value!r} keys without error"

    def inventory_value_for(self, source_value: str) -> str | None:
        """Render the inventory-side text a source value should match.

        Because a template both parses and formats, a spec can manufacture the
        folder name a reference ought to have. The verifier uses this to build
        synthetic candidates and prove a rule rejects ambiguity, without needing
        a second, hand-written pattern that could disagree with the first.
        """
        parts = self._first_parse(self.source_templates, source_value)
        if parts is None:
            return None
        template = self.inventory_templates[0]
        filled = dict(self.defaults)
        filled.update(parts)
        for name in template.part_names:
            filled.setdefault(name, "0")
        try:
            return template.format(filled)
        except DerivationError:
            return None

    def describe(self) -> dict[str, Any]:
        return {
            "source_templates": [item.pattern for item in self.source_templates],
            "source_match_mode": self.source_template.match_mode,
            "inventory_templates": [item.pattern for item in self.inventory_templates],
            "inventory_match_mode": self.inventory_template.match_mode,
            "key_parts": list(self.key_parts),
            "normalizers": {name: list(values) for name, values in self.normalizers.items()},
            "defaults": dict(self.defaults),
        }


def from_declaration(payload: dict[str, Any]) -> KeyDerivation:
    """Build a derivation from the JSON shape a compiler would emit."""
    try:
        source_mode = payload.get("source_match_mode", "exact")
        inventory_mode = payload.get("inventory_match_mode", "exact")
        return KeyDerivation(
            source_templates=tuple(
                Template(item, source_mode) for item in payload["source_templates"]
            ),
            inventory_templates=tuple(
                Template(item, inventory_mode) for item in payload["inventory_templates"]
            ),
            key_parts=tuple(payload["key_parts"]),
            defaults=dict(payload.get("defaults") or {}),
            normalizers={
                name: tuple(values) for name, values in (payload.get("normalizers") or {}).items()
            },
            year_pivot=int(payload.get("year_pivot", 30)),
        )
    except KeyError as exc:
        raise DerivationError(f"Derivation declaration is missing {exc}") from exc


def index_inventory(
    rows: Iterable[dict[str, str]],
    derivation: KeyDerivation,
    *,
    folder_field: str = "folder",
) -> tuple[dict[tuple[str, ...], list[dict[str, str]]], list[str]]:
    """Key every inventory row, returning the index and the rows that would not key."""
    index: dict[tuple[str, ...], list[dict[str, str]]] = {}
    unparsed: list[str] = []
    for row in rows:
        folder = str(row.get(folder_field) or "")
        key = derivation.inventory_key(folder)
        if key is None:
            unparsed.append(folder)
            continue
        index.setdefault(key, []).append(row)
    return index, unparsed
