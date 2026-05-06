"""
D-LOCO code utilities (no bridging/concatenation yet).

Paper reference:
  "Protecting the Future of Information: LOCO Coding With Error Detection for DNA Data Storage"
  (İrimağzı, Uslan, Hareedy)

D-LOCO code D_{m, ell}: length-m strings over {A,T,G,C} that do NOT contain any
run of length (ell + 1) of the same symbol (no Λ^{ell + 1} for any Λ in the alphabet).
The paper also orders codewords lexicographically with A < T < G < C.
"""

from __future__ import annotations

from functools import lru_cache
from fractions import Fraction
from typing import List, Sequence

from tqdm import tqdm


from src.utils.eval import info_density
from src.utils.constraints import DEFAULT_ALPHABET, has_valid_alphabet, is_run_length_controlled, is_valid_codeword


# A <-> C and T <-> G
_COMPLEMENT_TRANS = str.maketrans({"A": "C", "C": "A", "T": "G", "G": "T"})


def complement_codeword(codeword: str, *, alphabet: Sequence[str] = DEFAULT_ALPHABET) -> str:
    """
    Return the D-LOCO "complement" c̄ used for balancing in the paper.

    Mapping:
      A ↔ C
      T ↔ G

    Properties (as used in the paper for odd m):
    - p(c̄) = -p(c) where p is disparity (#GC - #AT)
    - Complement preserves the run-length constraint (it only renames symbols).
    """
    assert has_valid_alphabet(codeword, alphabet)
    return codeword.translate(_COMPLEMENT_TRANS)


def disparity(codeword: str) -> int:
    """
    Disparity p(c) = #G + #C - #A - #T (paper Definition 2).
    """
    # No alphabet assertion here; keep it lightweight.
    return (codeword.count("G") + codeword.count("C")) - (codeword.count("A") + codeword.count("T"))


def _make_suffix_counter(m: int, ell: int, alphabet: Sequence[str] = DEFAULT_ALPHABET):
    """
    Create a memoized function count(pos, last_idx, run_len) -> number of valid suffixes.

    - pos: next position to fill in [0..m]
    - last_idx: index into alphabet of previous symbol, or -1 for "none" (pos==0)
    - run_len: current run length of last symbol (0 if last_idx==-1)
    """
    assert m >= 0
    assert ell >= 1
    assert alphabet

    # Convert to tuple to ensure stable ordering + safe closure capture.
    alphabet = tuple(alphabet)
    q = len(alphabet)

    @lru_cache(maxsize=None)
    def count(pos: int, last_idx: int, run_len: int) -> int:
        if pos == m:
            return 1

        total = 0
        for sym_idx in range(q):
            if sym_idx == last_idx:
                if run_len >= ell:
                    continue
                total += count(pos + 1, sym_idx, run_len + 1)
            else:
                total += count(pos + 1, sym_idx, 1)
        return total

    return count


def N(m: int, ell: int) -> int:
    """
    Compute N(m)=|D_{m,ell}| using Proposition 1's recurrence (q=4 alphabet).

    Proposition 1 (paper):
      - For 1 ≤ r ≤ ell: N(r) = 4 ^ r
      - For r ≥ ell: N(r) = 3N(r - 1) + 3N(r - 2)+ ... + 3N(r - ell)
      - With the convention N(0) = 4/3 to make the recurrence also hold at r = ell.

    We follow the convention internally using fractions and return the true integer N(m).
    """
    assert m >= 0
    assert ell >= 1

    if m == 0:
        return 1

    n: List[Fraction] = [Fraction(4, 3)]  # paper convention
    for r in range(1, min(m, ell) + 1):
        n.append(Fraction(4 ** r, 1))

    for r in range(ell + 1, m + 1):
        n_r = 3 * sum(n[r - k] for k in range(1, ell + 1))
        n.append(n_r)

    out = n[m]
    assert out.denominator == 1
    return int(out)


def g(codeword: str, ell: int, *, alphabet: Sequence[str] = DEFAULT_ALPHABET) -> int:
    """
    Lexicographic rank g(c) of `codeword` within D_{m, ell} under the given `alphabet` order.

    This is the LOCO idea (lexicographic indexing) implemented via DP counting.
    """
    assert ell >= 1
    assert alphabet
    assert has_valid_alphabet(codeword, alphabet)
    assert is_run_length_controlled(codeword, ell)

    m = len(codeword)
    count = _make_suffix_counter(m, ell, alphabet)
    sym_to_idx = {sym: i for i, sym in enumerate(alphabet)}

    rank = 0
    last_idx = -1
    run_len = 0

    for pos, ch in enumerate(codeword):
        ch_idx = sym_to_idx[ch]

        # Add counts for all valid choices smaller than ch at this position.
        for sym_idx in range(ch_idx):
            if sym_idx == last_idx:
                if run_len >= ell:
                    continue
                rank += count(pos + 1, sym_idx, run_len + 1)
            else:
                rank += count(pos + 1, sym_idx, 1)

        # Advance state with the actual symbol ch.
        if ch_idx == last_idx:
            run_len += 1
            assert run_len <= ell
        else:
            last_idx = ch_idx
            run_len = 1

    return rank


def g_inverse(
    m: int,
    ell: int,
    index: int,
    *,
    alphabet: Sequence[str] = DEFAULT_ALPHABET,
    count=None,
) -> str:
    """
    Inverse of g: map an index in [0, |D_{m,ell}|) to the corresponding codeword.
    """
    assert m >= 0
    assert ell >= 1
    assert alphabet
    assert index >= 0

    if count is None:
        count = _make_suffix_counter(m, ell, alphabet)
    total = count(0, -1, 0)
    assert index < total

    out: List[str] = []
    last_idx = -1
    run_len = 0

    for pos in range(m):
        for sym_idx, sym in enumerate(alphabet):
            if sym_idx == last_idx:
                if run_len >= ell:
                    continue
                cnt = count(pos + 1, sym_idx, run_len + 1)
                if index >= cnt:
                    index -= cnt
                    continue
                out.append(sym)
                run_len += 1
                break
            else:
                cnt = count(pos + 1, sym_idx, 1)
                if index >= cnt:
                    index -= cnt
                    continue
                out.append(sym)
                last_idx = sym_idx
                run_len = 1
                break
        else:  # pragma: no cover
            raise RuntimeError("unreachable: no valid symbol choice found")

    return "".join(out)


def encode_b2c(b: str, m: int, ell: int) -> str:
    """
    Encode a binary message into a single D-LOCO codeword of length m (no bridging).

    Requirements:
    - len(bits) must equal s = floor(log2(N(m,ell))) for maximum info density.
    - The resulting index is in [0, 2^s) ⊆ [0, N(m,ell)).
    """
    assert m >= 0
    assert ell >= 1
    assert all(ch in "01b" for ch in b)

    cardinality = N(m, ell)
    assert 0 <= int(b, 2) < cardinality

    index = int(b, 2) if b else 0
    assert index < N(m, ell)
    return g_inverse(m, ell, index)


def decode_c2b(c: str, ell: int) -> str:
    """
    Decode a D-LOCO codeword into a binary string by computing its lexicographic rank.
    """
    assert ell >= 1
    m = len(c)
    assert is_valid_codeword(c, ell)

    index = g(c, ell)
    return format(index, f"0{bin(index)}b") if index > 0 else ""


def build_codebook(m: int, ell: int) -> List[str]:
    """
    Build the codebook for D-LOCO code D_{m,ell} as a list of codewords.
    """
    assert m >= 0
    assert ell >= 1
    return [encode_b2c(bin(idx), m, ell) for idx in tqdm(range(0, N(m, ell)), desc="building codebook...")]

def main() -> None:
    """
    Main function to test the implementation.
    """
    ell = 3
    m = 8

    codebook = build_codebook(m, ell)
    valid_codebook = [c for c in tqdm(codebook, desc="filtering valid codewords...") if is_valid_codeword(c, ell)]
    cardinality = len(valid_codebook)
    print(f"number of valid codewords: {cardinality}")
    print(f"information density: {info_density(cardinality, m + 2)}")



if __name__ == "__main__":
    main()