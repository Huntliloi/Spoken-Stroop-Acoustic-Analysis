"""Conservative, target-position mapping for serial Stroop color responses.

The first spoken color assigned to an item determines accuracy.  A following
correction is not treated as a new item.  At most two additional color words
can be searched, and only when the recording contains a surplus of recognized
words relative to the remaining target positions.  This prevents an ordinary
substitution in a 12-word recording from being mistaken for a correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


TARGETS = {
    "Hanzi": tuple("黄红绿蓝绿黄蓝红蓝绿红黄"),
    "Dots": tuple("红黄蓝绿绿蓝黄红黄红绿蓝"),
    "STT": tuple("黄红绿蓝绿蓝黄红红蓝绿黄"),
}
LABELS = {"Hanzi": "W", "Dots": "C", "STT": "CW"}
COLORS = set("红黄蓝绿")


@dataclass(frozen=True)
class ItemMapping:
    item_index: int
    target_word: str
    first_token_index: int | None  # zero based in the full recognized sequence
    final_token_index: int | None
    first_word: str | None
    final_word: str | None
    correct_first: bool
    status: str


def normalized_colors(candidates: list[str]) -> list[str]:
    traditional = str.maketrans({"紅": "红", "黃": "黄", "藍": "蓝", "綠": "绿"})
    return [word.translate(traditional) for word in candidates
            if word.translate(traditional) in COLORS]


def _short_sequence_mapping(words: list[str], target: tuple[str, ...]) -> list[ItemMapping]:
    """Align an incomplete observed sequence, allowing missing target items.

    Maximize matches, then prefer the least displacement from the recording
    order.  This applies to fewer than 12 recognized words only.  It does not
    manufacture values for missing words.
    """
    best: tuple[int, int] | None = None
    best_positions: tuple[int, ...] | None = None
    for positions in combinations(range(12), len(words)):
        score = sum(word == target[pos] for word, pos in zip(words, positions))
        displacement = sum(pos - j for j, pos in enumerate(positions))
        candidate = (score, -displacement)
        if best is None or candidate > best:
            best, best_positions = candidate, positions
    by_position = {} if best_positions is None else dict(zip(best_positions, range(len(words))))
    result = []
    for i, expected in enumerate(target):
        j = by_position.get(i)
        result.append(ItemMapping(i + 1, expected, j, j,
                                  words[j] if j is not None else None,
                                  words[j] if j is not None else None,
                                  j is not None and words[j] == expected,
                                  "single_response" if j is not None else "unrecognized_item"))
    return result


def map_items(words: list[str], task: str) -> tuple[list[ItemMapping], list[int]]:
    """Return 12 target positions and any trailing, unassigned token indices."""
    target = TARGETS[task]
    if len(words) < 12:
        return _short_sequence_mapping(words, target), []
    result: list[ItemMapping] = []
    j = 0
    for i, expected in enumerate(target):
        remaining_targets = 12 - i
        surplus = max(0, len(words) - j - remaining_targets)
        lookahead = min(2, surplus)
        end = min(j + lookahead + 1, len(words))
        match = next((k for k in range(j, end) if words[k] == expected), None)
        if match is None:
            result.append(ItemMapping(i + 1, expected, j, j, words[j], words[j],
                                      False, "uncorrected_error"))
            j += 1
        else:
            result.append(ItemMapping(i + 1, expected, j, match, words[j],
                                      words[match], match == j,
                                      "correct_first" if match == j else "self_corrected"))
            j = match + 1
    return result, list(range(j, len(words)))


def accuracy(mapping: list[ItemMapping]) -> tuple[int, float]:
    correct = sum(int(item.correct_first) for item in mapping)
    return correct, correct / 12.0
