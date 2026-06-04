"""
Comma-code evaluation entrypoint using shared eval utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.independent.comma.implementation import CommaCodeGenerator
from src.utils.eval import (
    CodewordEncoder,
    exhaustive_ascii_reverse_complement_report,
)


@dataclass(frozen=True)
class CommaCodeAdapter:
    generator: CommaCodeGenerator
    name: str = "comma_code"

    @property
    def codeword_length_nt(self) -> int:
        return self.generator.codeword_length_nt

    def encode_codeword(self, bits: str) -> str:
        return self.generator.encode_byte(bits)


def build_generator() -> CodewordEncoder:
    return CommaCodeAdapter(generator=CommaCodeGenerator())


def main() -> None:
    rc_report = exhaustive_ascii_reverse_complement_report(
        generator=build_generator(),
        bit_length=8,
    )
    print("=== reverse complement collision report ===")
    for key, value in rc_report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
