"""
Alternating code implementation (Smith et al., as summarized in the survey).

Codeword form:
  XYXYXY
where X is chosen from {A, G} and Y from {C, T}. This yields 64 symbols.

To fit the repo's 8-bit character workflow, one byte is represented as two
alternating symbols (total 12 nt) via a reversible mixed-radix map.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List


class AlternatingCodeGenerator:
    """
    Alternating-code generator with deterministic ranking/unranking.
    """

    x_alphabet: tuple[str, ...] = ("A", "G")
    y_alphabet: tuple[str, ...] = ("C", "T")
    symbol_length: int = 6
    symbols_per_byte: int = 2

    def __init__(self) -> None:
        symbols = self._build_symbol_table()
        self._symbols: List[str] = symbols
        self._symbol_to_index: Dict[str, int] = {sym: idx for idx, sym in enumerate(symbols)}

    def _build_symbol_table(self) -> List[str]:
        out: List[str] = []
        for choices in itertools.product((0, 1), repeat=self.symbol_length):
            codeword_chars: List[str] = []
            for pos, choice in enumerate(choices):
                if pos % 2 == 0:
                    codeword_chars.append(self.x_alphabet[choice])
                else:
                    codeword_chars.append(self.y_alphabet[choice])
            out.append("".join(codeword_chars))
        return sorted(out)

    def get_capacity(self) -> int:
        return len(self._symbols)

    @property
    def codeword_length_nt(self) -> int:
        return self.symbol_length * self.symbols_per_byte

    def information_density_per_symbol(self) -> float:
        return math.floor(math.log2(self.get_capacity())) / self.symbol_length

    def information_density_per_byte_mapping(self) -> float:
        return 8.0 / self.codeword_length_nt

    def encode_symbol(self, index: int) -> str:
        if index < 0 or index >= self.get_capacity():
            raise ValueError(f"symbol index out of range: {index}")
        return self._symbols[index]

    def decode_symbol(self, codeword: str) -> int:
        if codeword not in self._symbol_to_index:
            raise ValueError(f"invalid alternating code symbol: {codeword}")
        return self._symbol_to_index[codeword]

    def encode_byte(self, byte_value: int | str) -> str:
        if isinstance(byte_value, str):
            if any(ch not in "01" for ch in byte_value):
                raise ValueError("byte bitstring must contain only '0' and '1'")
            byte_value = int(byte_value, 2) if byte_value else 0

        if byte_value < 0 or byte_value >= 256:
            raise ValueError(f"byte value out of range: {byte_value}")

        # Mixed-radix representation with 4 x 64 = 256 states.
        hi = byte_value // self.get_capacity()
        lo = byte_value % self.get_capacity()
        return f"{self.encode_symbol(hi)}{self.encode_symbol(lo)}"

    def decode_byte(self, encoded: str, *, bit_length: int = 8) -> str:
        if len(encoded) != self.codeword_length_nt:
            raise ValueError(
                f"encoded byte length {len(encoded)} does not match {self.codeword_length_nt}"
            )
        first = encoded[: self.symbol_length]
        second = encoded[self.symbol_length :]
        value = self.decode_symbol(first) * self.get_capacity() + self.decode_symbol(second)
        return format(value, f"0{bit_length}b")
