from __future__ import annotations

import unittest

from Bio.Seq import Seq

from src.independent.edgecap.simulation import build_generator
from src.utils.eval import PCRConditions, encode_text_to_strand
from src.utils.pool_specificity import (
    DuplexScore,
    EncodedPool,
    EncodedPoolStrand,
    _make_edit_distance,
    analyze_probe_against_pool,
    build_encoded_pool,
    generate_target_absent_payloads,
)


def fake_thermodynamic_scorer(
    probe: str,
    template: str,
    conditions: PCRConditions,
) -> DuplexScore:
    del conditions
    target = str(Seq(probe).reverse_complement())
    matches = sum(left == right for left, right in zip(target, template))
    return DuplexScore(dg_kcal_mol=-float(matches), tm_c=float(matches))


class PoolSpecificityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = build_generator(length=5)
        cls.conditions = PCRConditions()

    def test_target_absent_payload_generation_is_reproducible(self) -> None:
        kwargs = {
            "target_words": ("road", "wood"),
            "count": 5,
            "payload_length": 40,
            "seed": 19,
        }
        first = generate_target_absent_payloads(**kwargs)
        second = generate_target_absent_payloads(**kwargs)
        self.assertEqual(first, second)
        self.assertTrue(all("road" not in text and "wood" not in text for text in first))

    def test_bit_vector_edit_distance_matches_known_examples(self) -> None:
        distance = _make_edit_distance("GATTACA")
        self.assertEqual(distance("GATTACA"), 0)
        self.assertEqual(distance("GACTATA"), 2)
        self.assertEqual(distance("GCATGCU"), 4)

    def test_repeated_intended_sites_are_not_off_targets(self) -> None:
        pool = build_encoded_pool(
            generator=self.generator,
            positive_payload="road road",
            target_words=("road",),
            negative_count=2,
            negative_payload_length=9,
            seed=5,
        )
        result = analyze_probe_against_pool(
            pool=pool,
            generator=self.generator,
            target_word="road",
            conditions=self.conditions,
            thermodynamic_candidate_limit=10,
            thermodynamic_random_controls=2,
            thermodynamic_scorer=fake_thermodynamic_scorer,
        )
        self.assertEqual(result["intended_site_count"], 2)
        self.assertGreater(result["intended_locus_overlap_windows_excluded"], 0)
        self.assertEqual(result["exact_off_target_occurrences"], 0)
        self.assertFalse(result["thermodynamic_analysis"]["exhaustive"])

    def test_exact_sequence_on_negative_strand_is_an_off_target(self) -> None:
        target = "road"
        target_dna = encode_text_to_strand(target, generator=self.generator)
        positive = EncodedPoolStrand(
            name="positive",
            payload=target,
            dna=target_dna,
            is_target_positive=True,
        )
        negative = EncodedPoolStrand(
            name="negative_0000",
            payload="xxxx",
            dna=target_dna,
            is_target_positive=False,
        )
        pool = EncodedPool(
            generator_name=self.generator.name,
            codeword_length_nt=self.generator.codeword_length_nt,
            seed=1,
            target_words=(target,),
            strands=(positive, negative),
        )
        result = analyze_probe_against_pool(
            pool=pool,
            generator=self.generator,
            target_word=target,
            conditions=self.conditions,
            thermodynamic_scorer=fake_thermodynamic_scorer,
        )
        self.assertEqual(result["intended_site_count"], 1)
        self.assertEqual(result["exact_off_target_occurrences"], 1)
        self.assertEqual(result["minimum_off_target_hamming_distance"], 0)
        self.assertEqual(result["minimum_off_target_edit_distance"], 0)
        self.assertEqual(
            result["thermodynamic_analysis"][
                "off_target_sites_at_least_as_stable_as_target"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
