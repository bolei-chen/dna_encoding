"""
D-LOCO evaluation entrypoint using shared eval utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.dependent.d_loco.implementation import N, complement_codeword, disparity, encode_b2c
from src.utils.eval import (
    CodewordEncoder,
    PCRConditions,
    as_json_ready,
    default_payload_text,
    simulate_word_primer_binding,
    exhaustive_ascii_reverse_complement_report,

)


@dataclass
class DLOCOAdapter:
    """
    Adapter exposing D-LOCO encoding through the eval interface.
    """

    length: int
    ell: int
    name: str = "dloco"
    running_disparity: int = 0

    @property
    def codeword_length_nt(self) -> int:
        return self.length

    def reset_for_new_strand(self) -> None:
        self.running_disparity = 0

    def encode_codeword(self, bits: str) -> str:
        try:
            codeword = encode_b2c(bits, self.length, self.ell)
        except AssertionError as exc:
            raise ValueError(
                f"cannot encode bitstring with D-LOCO (length={self.length}, ell={self.ell})"
            ) from exc

        comp = complement_codeword(codeword)
        disp = disparity(codeword)
        # Paper-style running-disparity balancing: pick c or c-bar to keep
        # cumulative AT/GC disparity as close to zero as possible.
        if abs(self.running_disparity + disp) <= abs(self.running_disparity - disp):
            chosen = codeword
            self.running_disparity += disp
        else:
            chosen = comp
            self.running_disparity -= disp
        return chosen


def build_generator(*, length: int = 5, ell: int = 3) -> CodewordEncoder:
    return DLOCOAdapter(length=length, ell=ell)


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
    # generator = DLOCOAdapter(length=5, ell=3)
    # if N(generator.length, generator.ell) < 2**8:
    #     raise ValueError("D-LOCO capacity is lower than required 8-bit space (256)")

    # result = simulate_word_primer_binding(
    #     payload_text=payload_text,
    #     target_word=target_word,
    #     generator=generator,
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
    # print("=== dloco random-access simulation ===")
    # for key, value in out.items():
    #     print(f"{key}: {value}")

    rc_report = exhaustive_ascii_reverse_complement_report(
        generator=build_generator(length=5, ell=3),
        bit_length=8,
    )
    print("=== reverse complement collision report ===")
    for key, value in rc_report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
