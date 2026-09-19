"""
Context-independent "edge cap" encoding built on top of CRLLCodeGenerator.

Idea: add one base at each end so that
  - caps differ from the first/last base of the core word (avoid length-4 runs)
  - GC content of the capped word stays within [gc_lower, gc_upper]
"""

from __future__ import annotations

import math
from typing import Sequence

from src.dependent.c_rll.dp import CRLLCodeGenerator
from src.utils.eval import info_density
from src.utils.constraints import is_gc_balanced, is_run_length_controlled
# from src.independent.edgecap.rs_code import rs_encode, rs_decode


COMPLEMENT = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
}

class EdgeCapCRLLGenerator(CRLLCodeGenerator):
    """
    CRLL generator with deterministic edge caps for context-independent concatenation.
    """

    @property
    def capped_length(self) -> int:
        return self.length + 2

    def cap_codeword(self, core: str) -> str:
        """
        Add one cap at each end of the core codeword.
        """
        if len(core) != self.length:
            raise ValueError(f"Core length {len(core)} does not match length {self.length}")

        alphabet: Sequence[str] = self.alphabet
        first = core[0]
        last = core[-1]

        core_gc = sum(1 for ch in core if ch in self.gc_symbols)
        min_gc = math.ceil(self.gc_lower * (self.length + 2))
        max_gc = math.floor(self.gc_upper * (self.length + 2))

        def is_gc(ch: str) -> int:
            return 1 if ch in self.gc_symbols else 0

        for start in alphabet:
            if start == first:
                continue
            for end in alphabet:
                if end == last:
                    continue
                gc_total = core_gc + is_gc(start) + is_gc(end)
                if min_gc <= gc_total <= max_gc:
                    return f"{start}{core}{end}"

        raise ValueError("No valid caps found to satisfy GC bounds and homopolymer constraint")

    def decap_codeword(self, capped: str) -> str:
        """
        Remove the two caps, returning the core codeword.
        """
        if len(capped) != self.length + 2:
            raise ValueError(
                f"Capped length {len(capped)} does not match length {self.length + 2}"
            )
        return capped[1:-1]

    def encode_capped(self, binary_value: int) -> str:
        """
        Encode a binary value and apply edge caps.
        """
        core = super().encode(binary_value)
        return self.cap_codeword(core)

    def decode_capped(self, capped: str, *, validate_caps: bool = True) -> int:
        """
        Decode a capped codeword back to its binary value.
        """
        if validate_caps:
            core = self.decap_codeword(capped)
            first = core[0]
            last = core[-1]
            if capped[0] == first or capped[-1] == last:
                raise ValueError("Invalid caps: cap matches core edge base")
        else:
            core = self.decap_codeword(capped)
        return super().decode(core)

    def capped_information_density(self) -> float:
        """
        Information density computed over the capped codeword length.
        """
        capacity = self.get_capacity()
        if capacity == 0:
            return 0.0
        return info_density(capacity, self.capped_length)

    def get_primer(self, binary_value: int) -> str:
        """
        Get the primer for a binary value.
        """
        codeword = self.encode_capped(binary_value)
        primer = reverse_complement(codeword)
        return primer

    def decode_seq(self, dna_sequence: str, *, bit_length: int) -> list[str]:
        """
        Decode a concatenated capped DNA sequence into fixed-width bitstrings.
        """
        if bit_length <= 0:
            raise ValueError("bit_length must be > 0")
        if len(dna_sequence) % self.capped_length != 0:
            raise ValueError(
                "DNA sequence length must be a multiple of capped codeword length"
            )

        bits: list[str] = []
        for i in range(0, len(dna_sequence), self.capped_length):
            chunk = dna_sequence[i : i + self.capped_length]
            bits.append(self.decode_capped(chunk, validate_caps=True))

        return [format(int(b, 2) if b else 0, f"0{bit_length}b") for b in bits]

    def encode_seq(self, bits: list[str]) -> str:
        """
        Encode a list of fixed-width bitstrings into a concatenated capped DNA sequence.
        """
        return "".join([self.encode_capped(b) for b in bits])


def reverse_complement(sequence: str) -> str:
    """
    Return the reverse complement of a DNA sequence.
    """
    return "".join(COMPLEMENT[ch] for ch in reversed(sequence))


def find_primer_binding_indices(dna_sequence: str, primer: str) -> list[int]:
    """
    Find all locations where a primer binds on the DNA strand.

    A primer binds to the reverse complement of its own sequence on the
    template strand. Returns nucleotide indices for all matches.
    """
    binding_target = reverse_complement(primer)
    indices: list[int] = []
    start = 0

    while True:
        index = dna_sequence.find(binding_target, start)
        if index == -1:
            break
        indices.append(index)
        start = index + 1

    return indices

def test_edgecap_information_density():
    gen = EdgeCapCRLLGenerator(length=10, max_run=3, gc_lower=0.4, gc_upper=0.6)
    print("capped information density: ", gen.capped_information_density())
    binary_values = [i for i in range(10)]
    for binary_value in binary_values:
        capped = gen.encode_capped(binary_value)
        print("capped: ", capped)
        print("binary value: {:010b}".format(binary_value))
        print("--------------------------------")


def simulate_edgecap_feasibility(
    *,
    length: int,
    max_run: int = 3,
    gc_lower: float = 0.4,
    gc_upper: float = 0.6,
    limit: int | None = None,
) -> dict:
    """
    Enumerate core codewords and check whether capping can always succeed.
    """
    gen = EdgeCapCRLLGenerator(
        length=length,
        max_run=max_run,
        gc_lower=gc_lower,
        gc_upper=gc_upper,
    )

    capacity = gen.get_capacity()
    total = capacity if limit is None else min(capacity, limit)
    failures: list[int] = []
    for idx in range(total):
        core = gen.encode(idx)
        try:
            capped = gen.cap_codeword(core)
        except ValueError:
            failures.append(idx)
            continue

        if not is_run_length_controlled(capped, max_run):
            failures.append(idx)
            continue

        if not is_gc_balanced(capped, gc_lower, gc_upper):
            failures.append(idx)
            continue

    return {
        "length": length,
        "checked": total,
        "capacity": capacity,
        "failures": len(failures),
        "failure_indices": failures,
    }


def print_ascii_codebook(gen: EdgeCapCRLLGenerator) -> None:
    """
    Print codebook entries for all 7-bit ASCII characters (0-127).
    """
    print("ASCII codebook:")
    print("dec\thex\tchar\tbits\t\tcodeword")
    print("-" * 72)
    for value in range(128):
        bits = format(value, "08b")
        codeword = gen.encode_capped(bits)
        char_label = ascii(chr(value))[1:-1]
        print(f"{value:3d}\t0x{value:02X}\t{char_label:<4}\t{bits}\t{codeword}")


def print_cache_values_for_n(gen: EdgeCapCRLLGenerator, n: int) -> None:
    """
    Print DP cache values for entries with remaining length n.
    """
    if n < 0 or n > gen.length:
        raise ValueError(f"n must be between 0 and {gen.length}")

    rows = [
        (state, value)
        for (state, remaining), value in gen.cache.items()
        if remaining == n
    ]
    rows.sort(key=lambda item: ((item[0].last_sym or ""), item[0].run_len, item[0].gc_count))

    print(f"DP cache values for n = {n}:")
    print("state\t\t\tvalue")
    print("-" * 48)
    for state, value in rows:
        print(f"{state!r:<24}\t{value}")
    print(f"total entries printed: {len(rows)}")


def main() -> None:
    encoder = EdgeCapCRLLGenerator(length=5, max_run=3, gc_lower=0.4, gc_upper=0.6)
    print("capped info density: ", encoder.capped_information_density())


    sentence = "i am very happy to have this conversation with you guys!"
    bits = [format(ord(ch), '08b') for ch in sentence]
    codewords = encoder.encode_seq(bits)
    print("Encoded DNA Strand: ", codewords)

    target = "conversation"

    primer = ""
    for ch in reversed(target):
        primer += encoder.get_primer(format(ord(ch), '08b'))
    print("primer: ", primer)
    target_nt_indices = find_primer_binding_indices(codewords, primer)
    print("target binding nucleotide indices: ", target_nt_indices)
    print(
        "target binding codeword indices: ",
        [index // encoder.capped_length for index in target_nt_indices],
    )

    decoded_bits = encoder.decode_seq(codewords, bit_length=8)
    # print("decoded bits: ", decoded_bits)
    print("decoded bits == bits: ", decoded_bits == bits)


if __name__ == "__main__":
    main()