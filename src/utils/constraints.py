from typing import Iterable, Sequence

DEFAULT_ALPHABET: Sequence[str] = ("A", "T", "G", "C")


def has_valid_alphabet(codeword: str, alphabet: Iterable[str] = DEFAULT_ALPHABET) -> bool:
    """
    Check if `codeword` uses only symbols from `alphabet`.
    """
    assert alphabet
    assert all(isinstance(sym, str) and len(sym) == 1 for sym in alphabet)
    return all(ch in alphabet for ch in codeword)

def is_run_length_controlled(codeword: str, ell: int) -> bool:
    """
    Check if `codeword` is run-length controlled for length `ell`, i.e., it:
    - has no run of identical symbols longer than `ell`
    """
    assert ell >= 1

    last = None
    run_len = 0
    for ch in codeword:
        if ch == last:
            run_len += 1
            if run_len > ell:
                return False
        else:
            last = ch
            run_len = 1
    return True

def is_gc_balanced(codeword: str, lower_bound: float = 0.45, upper_bound: float = 0.55) -> bool:
    """
    Check if `codeword` is GC-balanced, i.e., it has between `lower_bound` and `upper_bound` fraction of G's and C's.
    """
    assert lower_bound >= 0
    assert upper_bound <= 1
    assert lower_bound <= upper_bound
    gc_fraction = (codeword.count("G") + codeword.count("C")) / len(codeword)
    return lower_bound <= gc_fraction <= upper_bound

def is_valid_codeword(codeword: str, ell: int, *, alphabet: Iterable[str] = DEFAULT_ALPHABET) -> bool:
    """
    Check if `codeword` is a valid DNA codeword, i.e., it:
    - uses only symbols from `alphabet`
    - is run-length controlled for length `ell`
    - is GC-balanced
    """
    return has_valid_alphabet(codeword, alphabet) and is_run_length_controlled(codeword, ell) and is_gc_balanced(codeword)
