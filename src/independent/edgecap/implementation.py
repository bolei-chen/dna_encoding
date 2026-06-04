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
        complement = {
            "A": "T",
            "T": "A",
            "C": "G",
            "G": "C",
        }
        primer = "".join(complement[ch] for ch in reversed(codeword))
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


def main() -> None:
    # result = simulate_edgecap_feasibility(length=8, max_run=3, gc_lower=0.4, gc_upper=0.6, limit=None)
    # print(result)
    gen = EdgeCapCRLLGenerator(length=12, max_run=3, gc_lower=0.4, gc_upper=0.6)
    print("capped info density: ", gen.capped_information_density())

    sentence = "This is a cat"
    print("sentence: ", sentence)
    bits = [format(ord(ch), '08b') for ch in sentence]
    print("sentence as 8-bit binaries: ", bits)
    codewords = gen.encode_seq(bits)

    target = "cat"
    primer = ""
    for ch in reversed(target):
        primer += gen.get_primer(format(ord(ch), '08b'))
    print("primer: ", primer)

    decoded_bits = gen.decode_seq(codewords, bit_length=8)
    print("decoded bits: ", decoded_bits)
    print("decoded bits == bits: ", decoded_bits == bits)


if __name__ == "__main__":
    main()