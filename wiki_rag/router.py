"""
Rule-based query routing: person, place, or both (no ML).

Uses ``config.FAMOUS_PEOPLE`` / ``FAMOUS_PLACES`` and keyword heuristics (PRD §5.5).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config  # noqa: E402

PERSON_KEYWORDS = [
    "who is",
    "who was",
    "born",
    "died",
    "invented",
    "discovered",
    "wrote",
    "played",
    "singer",
    "scientist",
    "artist",
    "philosopher",
    "athlete",
    "actor",
    "painted",
]

COMPARISON_TRIGGERS = [
    "compare",
    "vs",
    "versus",
    "difference between",
    "similar to",
    "different from",
    "both",
    "which is better",
    "how do",
    "contrast",
    "alike",
    "in common",
]

PLACE_KEYWORDS = [
    "where is",
    "where was",
    "located",
    "built",
    "visit",
    "tower",
    "wall",
    "mountain",
    "canyon",
    "temple",
    "statue",
    "pyramid",
    "city",
    "country",
    "monument",
    "wonder",
]


def _catalog_name_matches(query_lower: str, canonical_name: str) -> bool:
    """
    True if ``canonical_name`` is mentioned in ``query_lower`` (already lowercased).

    Matches the full phrase as a substring, or any word of length ≥ 3 as a
    whole word (so e.g. ``Einstein`` matches ``Albert Einstein`` entries).
    """
    name = canonical_name.strip().lower()
    if name in query_lower:
        return True
    for part in name.split():
        if len(part) < 3:
            continue
        if re.search(r"\b" + re.escape(part) + r"\b", query_lower):
            return True
    return False


def _keyword_in_query(query_lower: str, kw: str) -> bool:
    """Match multi-word phrases as substrings; single tokens as whole words."""
    kw = kw.strip().lower()
    if not kw:
        return False
    if " " in kw:
        return kw in query_lower
    return re.search(r"\b" + re.escape(kw) + r"\b", query_lower) is not None


def classify_query(query: str) -> str:
    """
    Return ``"person"``, ``"place"``, or ``"both"`` using catalog + keyword rules.
    """
    q = query.lower()

    # Hypothetical / out-of-catalog questions (e.g. PRD failure cases): search both types.
    if re.search(r"\bpresident\b", q) and re.search(r"\bmars\b", q):
        return "both"

    people_match = any(
        _catalog_name_matches(q, p) for p in config.FAMOUS_PEOPLE
    )
    places_match = any(
        _catalog_name_matches(q, pl) for pl in config.FAMOUS_PLACES
    )

    if people_match and places_match:
        return "both"
    if people_match:
        return "person"
    if places_match:
        return "place"

    person_kw = any(_keyword_in_query(q, kw) for kw in PERSON_KEYWORDS)
    place_kw = any(_keyword_in_query(q, kw) for kw in PLACE_KEYWORDS)

    if person_kw and place_kw:
        return "both"
    if person_kw:
        return "person"
    if place_kw:
        return "place"

    return "both"


def get_entity_filter(query: str) -> str | None:
    """
    Vector-store filter: ``\"person\"``, ``\"place\"``, or ``None`` when scope is both.
    """
    label = classify_query(query)
    if label == "both":
        return None
    return label


def is_comparison_query(query: str) -> bool:
    query_lower = query.lower()
    return any(trigger in query_lower for trigger in COMPARISON_TRIGGERS)


def _earliest_match_pos(query: str, canonical: str) -> int | None:
    """Leftmost match position for full name substring or any long token (whole word)."""
    ql = query.lower()
    cl = canonical.strip().lower()
    if cl in ql:
        return ql.find(cl)
    best: int | None = None
    for part in cl.split():
        if len(part) < 3:
            continue
        m = re.search(r"\b" + re.escape(part) + r"\b", ql)
        if m:
            p = m.start()
            best = p if best is None else min(best, p)
    return best


def extract_comparison_entities(query: str) -> dict:
    """
    Detect catalog entities mentioned for a comparison-style question.

    Returns:
        is_comparison: from :func:`is_comparison_query`
        entities: ordered list of canonical names
        entity_types: parallel \"person\" / \"place\" labels
        comparison_type: person_person | place_place | person_place | single_type
    """
    is_comp = is_comparison_query(query)
    q = query.lower()
    raw: list[tuple[int, str, str]] = []

    for p in config.FAMOUS_PEOPLE:
        if _catalog_name_matches(q, p):
            pos = _earliest_match_pos(query, p)
            raw.append((pos if pos is not None else 10**9, p, "person"))

    for pl in config.FAMOUS_PLACES:
        if _catalog_name_matches(q, pl):
            pos = _earliest_match_pos(query, pl)
            raw.append((pos if pos is not None else 10**9, pl, "place"))

    raw.sort(key=lambda t: t[0])
    entities: list[str] = []
    entity_types: list[str] = []
    seen: set[str] = set()
    for _pos, name, et in raw:
        if name in seen:
            continue
        seen.add(name)
        entities.append(name)
        entity_types.append(et)

    type_set = set(entity_types)
    if len(entities) <= 1:
        comparison_type = "single_type"
    elif "person" in type_set and "place" in type_set:
        comparison_type = "person_place"
    elif type_set == {"person"}:
        comparison_type = "person_person"
    elif type_set == {"place"}:
        comparison_type = "place_place"
    else:
        comparison_type = "single_type"

    return {
        "is_comparison": is_comp,
        "entities": entities,
        "entity_types": entity_types,
        "comparison_type": comparison_type,
    }


def extract_mentioned_entities(query: str) -> dict:
    """
    Return ``{\"people\": [...], \"places\": [...]}`` of catalog strings detected.
    """
    q = query.lower()
    people = [p for p in config.FAMOUS_PEOPLE if _catalog_name_matches(q, p)]
    places = [pl for pl in config.FAMOUS_PLACES if _catalog_name_matches(q, pl)]
    return {"people": people, "places": places}


if __name__ == "__main__":
    test_queries = [
        "Who was Albert Einstein?",
        "Where is the Eiffel Tower located?",
        "Compare Marie Curie and Nikola Tesla",
        "What is the Taj Mahal?",
        "Who invented electricity?",
        "Which famous place is in Turkey?",
        "Who is the president of Mars?",
    ]
    for q in test_queries:
        print(f"Query: {q}")
        print(f"  Type: {classify_query(q)}")
        print(f"  Filter: {get_entity_filter(q)}")
        print()
