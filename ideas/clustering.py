import string
import numpy as np
import Levenshtein
import pandas as pd
from matplotlib import pyplot as plt

import fortepyan as ff
from fortepyan import MidiPiece
from fortepyan.viz.structures import PianoRoll

pitch_map = {20 + it: string.printable.strip()[it] for it in range(90)}


def process_piece(piece: MidiPiece, n: int = 16):
    df = piece.df.copy()
    df["pitch_char"] = df.pitch.map(pitch_map)

    howmany = piece.size - n + 1

    pitch_chars = df.pitch_char.values

    ngrams = ["".join(pitch_chars[it:it + n]) for it in range(howmany)]
    df["gram_duration"] = df.start.shift(-n) - df.start

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

    keep = np.array(keep)
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
    left_shifts, right_shifts = [], []
    for it in idxs:
        lefts, rights = [], []
        for jt in idxs:
            left, right = expand_sequences(pitch_sequence, it, jt, n, 100)

            left_shift = get_shift_limit(left, threshold)
            lefts.append(left_shift)

            right_shift = get_shift_limit(right, threshold)
            rights.append(right_shift)

        left_shifts.append(lefts)
        right_shifts.append(rights)

    left_shifts = np.array(left_shifts)
    right_shifts = np.array(right_shifts)
    return left_shifts, right_shifts


def select_variants(
    idxs: list[int],
    left_shifts: np.array,
    right_shifts: np.array,
):
    scores = []
    for it in range(5):
        for jt in range(5):
            left_shift = jt * 5
            right_shift = it * 5
            ids = right_shifts >= right_shift
            jds = left_shifts >= left_shift

            kds = ids & jds
            top_row = kds.sum(1).argmax()
            score = {
                "left_shift": left_shift,
                "right_shift": right_shift,
                "row": top_row,
                "n_variants": kds[top_row].sum(),
                "expansion": left_shift + right_shift,
                "idxs": idxs[kds[top_row]],
            }
            scores.append(score)

    score = pd.DataFrame(scores)

    # Set the target length to be half the length of idxs, rounded up to the nearest integer,
    # but ensure that it is at least 3 and no more than the length of idxs
    target = min(max(len(idxs) * 0.5, 3), len(idxs))

    vds = score.n_variants >= target
    idx = score[vds].expansion.argmax()

    selected = score[vds].iloc[idx]
    return selected


def deduplicate_ngrams(ngrams: list[str], threshold: int) -> list[str]:
    selected_grams = []
    for gram in ngrams:
        if not any([Levenshtein.distance(gram, s) <= threshold for s in selected_grams]):
            selected_grams.append(gram)

    return selected_grams


def remove_duplicate_ngrams(ngrams: list[str], threshold: int) -> list[str]:
    unique_ngrams = []

    # Iterate through each n-gram in the input list
    for ngram in ngrams:
        # Check if the current n-gram is similar to any selected n-gram
        # by comparing their Levenshtein distances
        is_similar = any([
            Levenshtein.distance(ngram, existing_ngram) <= threshold
            for existing_ngram in unique_ngrams
        ])

        # If the current n-gram is not similar to any existing unique n-grams,
        # add it to the unique_ngrams list
        if not is_similar:
            unique_ngrams.append(ngram)

    return unique_ngrams


def get_gram_variants(
    gram: str,
    pitch_sequence: list[str],
    df: pd.DataFrame,
    n: int,
) -> dict:
    idxs = np.where(df.ngram == gram)[0]
    idxs = filter_overlaping_sequences(idxs, n)

    # Fuzzy-wuzzy extension of thee seeds, *threshold* is a measure
    # of deviation between two sequences that are being compared
    # "if the sequence is extended further, next *threshold* notes
    # are going to be different between both sequences
    threshold = 4
    left_shifts, right_shifts = calculate_group_shifts(
        pitch_sequence=pitch_sequence,
        idxs=idxs,
        threshold=threshold,
        n=n,
    )

    # Use those thresholds to find groups of similar fragments
    # based on the same ngram seed
    variant = select_variants(
        idxs=idxs,
        left_shifts=left_shifts,
        right_shifts=right_shifts,
    )

    return variant.to_dict()


def draw_vairant(
    variant: dict,
    piece: MidiPiece,
    n: int,
):
    howmany = variant["n_variants"]
    left_shift = variant["left_shift"]
    right_shift = variant["right_shift"]
    idxs = variant["idxs"]
    print('left:', left_shift, 'right:', right_shift)

    fig, axes = plt.subplots(nrows=howmany, ncols=1, figsize=[10, 2 * howmany])
    for ax, it in zip(axes, idxs):
        p = piece[it - left_shift: it + n + right_shift]
        pr = PianoRoll(p)
        ff.roll.draw_piano_roll(ax=ax, piano_roll=pr)
        ax.set_title(f"Index: {it}")


def midi_piece_clustering(piece: MidiPiece, n: int):
    df, pitch_sequence = process_piece(piece, n)

    gram_counts = df.ngram.value_counts()
    ids = gram_counts > 2
    top_grams = gram_counts[ids].index

    duplication_threshold = 2 / 3 * n
    selected_grams = remove_duplicate_ngrams(top_grams, threshold=duplication_threshold)

    variants = []
    for gram in selected_grams:
        variant = get_gram_variants(
            gram=gram,
            pitch_sequence=pitch_sequence,
            df=df,
            n=n,
        )
        if variant["n_variants"] > 1:
            variants.append(variant)

    return variants
