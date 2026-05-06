"""
Utilities for constructing (M, d, k) RLL codewords via the finite-state transition diagram (FSTD)
and the M-mod precoder described in:

  "Construction of Bio-Constrained Code for DNA Data Storage"
  (Wang et al., IEEE Comm. Lett., 2019)

This file focuses on generating the RLL codewords only (homopolymer/run-length constraint).
GC-balance and concatenation are handled in later steps.
"""

from __future__ import annotations

import math
from typing import Iterable, Iterator, List, Sequence, Tuple, Set

from src.utils.constraints import DEFAULT_ALPHABET
from src.utils.eval import info_density
from src.utils.constraints import is_valid_codeword, is_run_length_controlled, is_gc_balanced


def _generate_transition_sequences(
    n: int,
    m: int,
    d: int,
    k: int,
) -> Iterator[Tuple[int, ...]]:
    """
    Generate all length-n (M,d,k) transition sequences over Z_M.

    (M,d,k) constraint: between consecutive non-zero symbols, there are
    at least d and at most k zeros. Zeros represent "no transition",
    non-zeros represent a transition to a different symbol.

    The FSTD has states s_i for i = 0..k, where i is the current run of zeros.
    Transitions:
      - output 0: allowed if i < k, next state s_{i + 1}
      - output non-zero: allowed if i >= d, next state s_0

    Note: At position 0, we allow a non-zero without requiring d leading zeros.
    This matches the standard RLL convention (no constraint before the first transition).
    """
    assert n >= 0
    assert m >= 2
    assert 0 <= d <= k

    def dfs(pos: int, zeros_run: int, out: List[int]) -> Iterator[Tuple[int, ...]]:
        if pos == n:
            yield tuple(out)
            return

        # Option 1: output 0 (no transition)
        if zeros_run < k:
            out.append(0)
            yield from dfs(pos + 1, zeros_run + 1, out)
            out.pop()

        # Option 2: output non-zero (transition)
        if zeros_run >= d:
            for sym in range(1, m):
                out.append(sym)
                yield from dfs(pos + 1, 0, out)
                out.pop()

    # Allow a non-zero at the first position regardless of d by seeding zeros_run = d
    return dfs(0, d, [])


def _precoder(
    transition_sequence: Iterable[int],
    m: int,
    *,
    seed: int = 0,
) -> List[int]:
    """
    Apply the M-mod precoder: y_i = y_{i - 1} + x_i (mod M).
    """
    assert m >= 2
    assert 0 <= seed < m

    y: List[int] = []
    prev = seed
    for x in transition_sequence:
        prev = (prev + x) % m
        y.append(prev)
    return y


def build_rll_codewords(
    n: int,
    *,
    m: int = 4,
    d: int = 0,
    k: int = 2,
    alphabet: Sequence[str] = DEFAULT_ALPHABET,
    seeds: Iterable[int] | None = None,
) -> List[str]:
    """
    Construct length-n RLL codewords using the FSTD + M-mod precoder.

    Args:
        n: codeword length
        m: alphabet size (DNA uses 4)
        d, k: (M, d, k) constrained parameters on the transition sequence
        alphabet: symbol mapping for output (len(alphabet) must equal m)
        seeds: optional initial values y_0; if None, use all 0..m-1

    Returns:
        List of unique length-n codewords over `alphabet`.
    """
    assert n >= 0
    assert m >= 2
    assert 0 <= d <= k
    assert len(alphabet) == m
    alphabet = tuple(alphabet)

    if seeds is None:
        seeds = range(m)

    codewords_set: set[str] = set()
    ell = k + 1
    for x_seq in _generate_transition_sequences(n, m, d, k):
        for seed in seeds:
            y_seq = _precoder(x_seq, m, seed=seed)
            codeword = "".join(alphabet[val] for val in y_seq)
            if is_run_length_controlled(codeword, ell):
                codewords_set.add(codeword)
    return sorted(codewords_set)


def build_crll_codewords(
    n: int,
    *,
    m: int = 4,
    d: int = 0,
    k: int = 2,
    alphabet: Sequence[str] = DEFAULT_ALPHABET,
    seeds: Iterable[int] | None = None,
    gc_lower: float = 0.4,
    gc_upper: float = 0.6,
) -> Set[str]:
    """
    Construct length-n C-RLL codewords that satisfy both:
      - run-length limit (via FSTD + precoder)
      - GC balance within [gc_lower, gc_upper]
    """
    assert 0 <= gc_lower <= gc_upper <= 1
    assert n >= 0
    assert m >= 2
    assert 0 <= d <= k
    assert len(alphabet) == m
    alphabet = tuple(alphabet)

    if seeds is None:
        seeds = range(m)

    min_gc = math.ceil(gc_lower * n)
    max_gc = math.floor(gc_upper * n)
    gc_symbols = {"G", "C"}

    codewords_set: set[str] = set()

    def dfs(
        pos: int,
        zeros_run: int,
        prev: int,
        run_len: int,
        gc_count: int,
        out: List[str],
    ) -> None:
        if pos == n:
            if min_gc <= gc_count <= max_gc:
                codewords_set.add("".join(out))
            return

        remaining = n - pos
        if gc_count > max_gc:
            return
        if gc_count + remaining < min_gc:
            return

        # Option 1: output 0 (no transition)
        if zeros_run < k:
            if run_len < k + 1:
                y = prev
                out.append(alphabet[y])
                next_gc = gc_count + (1 if alphabet[y] in gc_symbols else 0)
                dfs(pos + 1, zeros_run + 1, prev, run_len + 1, next_gc, out)
                out.pop()

        # Option 2: output non-zero (transition)
        if zeros_run >= d:
            for x in range(1, m):
                y = (prev + x) % m
                out.append(alphabet[y])
                next_gc = gc_count + (1 if alphabet[y] in gc_symbols else 0)
                dfs(pos + 1, 0, y, 1, next_gc, out)
                out.pop()

    for seed in seeds:
        # Seed sets y_0; we start with no output yet.
        dfs(0, d, seed, 0, 0, [])

    return sorted(codewords_set)


def main() -> None:
    """
    Main function to test the implementation.
    """
    n = 10
    m = 4
    d = 0
    k = 2
    ell = k + 1
    alphabet = DEFAULT_ALPHABET
    seeds = None
    gc_lower = 0.40
    gc_upper = 0.60
    codewords = build_crll_codewords(
        n,
        m=m,
        d=d,
        k=k,
        alphabet=alphabet,
        seeds=seeds,
        gc_lower=gc_lower,
        gc_upper=gc_upper,
    )
    print(f"cardinality: {len(codewords)}")
    print(f"are all codewords valid: {all(is_valid_codeword(c, ell, gc_lower=gc_lower, gc_upper=gc_upper) for c in codewords)}")
    print(f"information density: {info_density(len(codewords), n)}")
    for c in list(codewords)[:5]:
        print(c)

if __name__ == "__main__":
    main()