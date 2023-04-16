import string
import numpy as np
import Levenshtein

from fortepyan import MidiPiece

pitch_map = {20 + it: string.printable[it] for it in range(90)}


def process_piece(piece: MidiPiece, n: int = 16):
    df = piece.df.copy()
    df["pitch_char"] = df.pitch.map(pitch_map)

    howmany = piece.size - n + 1

    pitch_chars = df.pitch_char.values

    ngrams = ["".join(pitch_chars[i:i + n]) for i in range(howmany)]

    df = df[:howmany].reset_index(drop=True)
    df["ngram"] = ngrams

    pitch_sequence = ''.join(pitch_chars)

    return df, pitch_sequence


def expand_sequences(pitch_sequence, it, jt, n, distance: int = 40):
    left = pitch_sequence[it: it + n]
    right = pitch_sequence[jt: jt + n]

    left_scores = []
    for shift in range(distance):
        left = pitch_sequence[it - shift: it + n]
        right = pitch_sequence[jt - shift: jt + n]
        d = Levenshtein.distance(left, right)
        left_scores.append(d)

    righ_scores = []
    for shift in range(distance):
        left = pitch_sequence[it: it + n + shift]
        right = pitch_sequence[jt: jt + n + shift]
        d = Levenshtein.distance(left, right)
        righ_scores.append(d)

    return left_scores, righ_scores


def filter_overlaping_sequences(idxs: list[int], n: int) -> list[int]:
    # Initialize a list to store the indexes to keep
    keep = []

    # Iterate over the indexes and check for overlaps
    for it, idx in enumerate(idxs):
        # Check if this index overlaps with any of the previous ones
        is_overlapping = any(idx - prev_idx < n for prev_idx in idxs[:it])

        # Only keep the index if it does not overlap with any previous sequence
        if not is_overlapping:
            keep.append(idx)

    return keep


def get_shift_limit(shifts: list[int], threshold: int = 5) -> int:
    kernel = np.ones(threshold)
    convolved = np.convolve(np.diff(shifts), kernel, mode="valid")

    hits = convolved == threshold
    if hits.any():
        acceptable_shift = np.where(hits)[0][0]
    else:
        acceptable_shift = len(shifts)

    return acceptable_shift


def calculate_group_shifts(
    pitch_sequence: list[str],
    idxs: list[int],
    threshold: int,
    n: int,
) -> tuple[list[int], list[int]]:
    ls, rs = [], []
    for it in idxs:
        lefts, rights = [], []
        for jt in idxs:
            left, right = expand_sequences(pitch_sequence, it, jt, n, 100)

            left_shift = get_shift_limit(left, threshold)
            lefts.append(left_shift)

            right_shift = get_shift_limit(right, threshold)
            rights.append(right_shift)
            # print('left:', left_shift, 'right:', right_shift)
        print(lefts)
        print(rights)
        print('---')
        # ls.append(min(lefts))
        # rs.append(min(rights))
        ls.append(lefts)
        rs.append(rights)

    return ls, rs
