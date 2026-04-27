"""
Block length sweep for Edge-Cap evaluation.

Runs the DP generator across a range of block lengths and reports:
- Codebook cardinality
- Bits per codeword
- Core information density (bits/nt)
- Capped information density (bits/nt)
- Whether full ASCII coverage (>= 256 values) is achievable
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from src.independent.edgecap.implementation import EdgeCapCRLLGenerator


@dataclass
class BlockLengthResult:
    """
    Result for a single block length configuration.
    """
    n: int
    cardinality: int
    bits_per_codeword: int
    core_density_bpnt: float
    capped_density_bpnt: float
    capped_length_nt: int
    full_ascii_coverage: bool

    def __str__(self) -> str:
        ascii_str = "yes" if self.full_ascii_coverage else "no"
        return (
            f"n={self.n:2d} | "
            f"cardinality={self.cardinality:8d} | "
            f"bits/codeword={self.bits_per_codeword:2d} | "
            f"core density={self.core_density_bpnt:.4f} b/nt | "
            f"capped density={self.capped_density_bpnt:.4f} b/nt | "
            f"ASCII coverage={ascii_str}"
        )


def sweep(
    lengths: List[int],
    *,
    max_run: int = 3,
    gc_lower: float = 0.4,
    gc_upper: float = 0.6,
    ascii_threshold: int = 256,
) -> List[BlockLengthResult]:
    """
    Run the DP generator for each block length in lengths and collect results.

    Args:
        lengths: list of block lengths to evaluate
        max_run: maximum homopolymer run length
        gc_lower: minimum GC content fraction
        gc_upper: maximum GC content fraction
        ascii_threshold: minimum cardinality for full ASCII coverage (default 256)

    Returns:
        List of BlockLengthResult, one per block length
    """
    results: List[BlockLengthResult] = []
    for n in lengths:
        gen = EdgeCapCRLLGenerator(
            length=n,
            max_run=max_run,
            gc_lower=gc_lower,
            gc_upper=gc_upper,
        )
        cardinality = gen.get_capacity()
        bits_per_codeword = math.floor(math.log2(cardinality)) if cardinality > 0 else 0
        core_density = gen.information_density()
        capped_density = gen.capped_information_density()
        capped_length = gen.capped_length
        full_ascii = cardinality >= ascii_threshold

        results.append(BlockLengthResult(
            n=n,
            cardinality=cardinality,
            bits_per_codeword=bits_per_codeword,
            core_density_bpnt=core_density,
            capped_density_bpnt=capped_density,
            capped_length_nt=capped_length,
            full_ascii_coverage=full_ascii,
        ))
    return results


def print_table(results: List[BlockLengthResult]) -> None:
    """
    Print results as a formatted table.
    """
    header = (
        f"{'n':>4} | "
        f"{'Cardinality':>12} | "
        f"{'Bits/CW':>8} | "
        f"{'Core (b/nt)':>12} | "
        f"{'Capped (b/nt)':>14} | "
        f"{'Capped len (nt)':>16} | "
        f"{'ASCII coverage':>14}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        ascii_str = "yes" if r.full_ascii_coverage else "no"
        print(
            f"{r.n:>4} | "
            f"{r.cardinality:>12} | "
            f"{r.bits_per_codeword:>8} | "
            f"{r.core_density_bpnt:>12.4f} | "
            f"{r.capped_density_bpnt:>14.4f} | "
            f"{r.capped_length_nt:>16} | "
            f"{ascii_str:>14}"
        )
    print(sep)


def main() -> None:
    lengths = [4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50]
    print(f"\nEdge-Cap block length sweep")
    print(f"Parameters: max_run=3, gc_lower=0.4, gc_upper=0.6\n")
    results = sweep(lengths)
    print_table(results)

    print("\nNote: full ASCII coverage requires cardinality >= 256 (8 bits per codeword).")
    for r in results:
        if r.full_ascii_coverage:
            print(f"  Minimum block length for ASCII coverage: n = {r.n}")
            break


if __name__ == "__main__":
    main()