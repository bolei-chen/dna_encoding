"""
CRLL evaluation entrypoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.dependent.c_rll.dp import CRLLCodeGenerator
from src.utils.eval import (
    CodewordEncoder,
    PCRConditions,
    as_json_ready,
    default_payload_text,
    exhaustive_ascii_reverse_complement_report,
    simulate_word_primer_binding,
)


@dataclass(frozen=True)
class CRLLAdapter:
    """
    Adapter exposing CRLLCodeGenerator through the eval interface.
    """

    generator: CRLLCodeGenerator
    name: str = "crll"

    @property
    def codeword_length_nt(self) -> int:
        return self.generator.length

    def __post_init__(self) -> None:
        object.__setattr__(self, "_capacity", self.generator.get_capacity())
        object.__setattr__(self, "_prev_suffix2", "")

    def reset_for_new_strand(self) -> None:
        object.__setattr__(self, "_prev_suffix2", "")

    @staticmethod
    def _has_triple_suffix(codeword: str) -> bool:
        return len(codeword) >= 3 and codeword[-1] == codeword[-2] == codeword[-3]

    @staticmethod
    def _boundary_run_valid(prev_suffix2: str, next_codeword: str, max_run: int = 3) -> bool:
        window = f"{prev_suffix2}{next_codeword[:max_run]}"
        run = 1
        for i in range(1, len(window)):
            run = run + 1 if window[i] == window[i - 1] else 1
            if run > max_run:
                return False
        return True

    def encode_codeword(self, bits: str) -> str:
        base_idx = int(bits, 2) if bits else 0
        prev_suffix2: str = getattr(self, "_prev_suffix2")

        # Dynamic concatenation rule: select a context-compatible codeword based
        # on the previous codeword suffix, matching the paper's context-dependent
        # concatenation idea for long strands.
        for delta in range(self._capacity):
            idx = (base_idx + delta) % self._capacity
            candidate = self.generator.encode(idx)
            if self._has_triple_suffix(candidate):
                continue
            if prev_suffix2 and not self._boundary_run_valid(prev_suffix2, candidate):
                continue
            object.__setattr__(self, "_prev_suffix2", candidate[-2:])
            return candidate

        raise RuntimeError("failed to find a CRLL codeword satisfying dynamic bridge rule")


def build_generator(*, length: int = 5) -> CodewordEncoder:
    return CRLLAdapter(
        generator=CRLLCodeGenerator(
            length=length,
            max_run=3,
            gc_lower=0.4,
            gc_upper=0.6,
        )
    )


def main() -> None:
    # payload_text = default_payload_text()
    # target_word = "road"
    # pcr_conditions = PCRConditions(
    #     mv_conc_mM=50.0,
    #     dv_conc_mM=1.5,
    #     dntp_conc_mM=0.2,
    #     primer_conc_nM=250.0,
    #     annealing_temp_c=68.0,
    # )
    # result = simulate_word_primer_binding(
    #     payload_text=payload_text,
    #     target_word=target_word,
    #     generator=build_generator(length=5),
    #     pcr_conditions=pcr_conditions,
    #     bit_length=8,
    #     background_count=1000,
    #     background_seed=7,
    #     identity_threshold=0.70,
    #     dg_threshold_kcal=-9.0,
    #     dg_threshold_kcal_three_prime=-4.0,
    #     three_prime_len=8,
    # )
    # out = as_json_ready(result)
    # print("=== crll random-access simulation ===")
    # for key, value in out.items():
    #     print(f"{key}: {value}")
    rc_report = exhaustive_ascii_reverse_complement_report(
        generator=build_generator(length=5),
        bit_length=8,
    )
    print("=== reverse complement collision report ===")
    for key, value in rc_report.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
