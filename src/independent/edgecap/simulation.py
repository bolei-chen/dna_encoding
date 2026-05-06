"""
EdgeCap-CRLL evaluation entrypoint using shared eval utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.independent.edgecap.implementation import EdgeCapCRLLGenerator
from src.utils.eval import (
    CodewordEncoder,
    PCRConditions,
    as_json_ready,
    default_payload_text,
    exhaustive_ascii_reverse_complement_report,
    simulate_word_primer_binding,
)


@dataclass(frozen=True)
class EdgeCapAdapter:
    generator: EdgeCapCRLLGenerator
    name: str = "edgecap_crll"

    @property
    def codeword_length_nt(self) -> int:
        return self.generator.capped_length

    def encode_codeword(self, bits: str) -> str:
        return self.generator.encode_capped(bits)


def build_generator(*, length: int = 5) -> CodewordEncoder:
    return EdgeCapAdapter(
        generator=EdgeCapCRLLGenerator(
            length=length,
            max_run=3,
            gc_lower=0.4,
            gc_upper=0.6,
        )
    )


def main() -> None:
    # result = simulate_word_primer_binding(
    #     payload_text=default_payload_text(),
    #     target_word="road",
    #     generator=build_generator(length=5),
    #     pcr_conditions=PCRConditions(
    #         mv_conc_mM=50.0,
    #         dv_conc_mM=1.5,
    #         dntp_conc_mM=0.2,
    #         primer_conc_nM=250.0,
    #         annealing_temp_c=68.0,
    #     ),
    #     bit_length=8,
    #     background_count=1000,
    #     background_seed=7,
    #     identity_threshold=0.70,
    #     dg_threshold_kcal=-9.0,
    #     dg_threshold_kcal_three_prime=-4.0,
    #     three_prime_len=8,
    # )
    # print("=== edgecap random-access simulation ===")
    # for key, value in as_json_ready(result).items():
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
