"""
Simulation utilities for Edge-Cap random-access retrieval.

This module demonstrates an in-silico workflow:
1) Encode text into concatenated Edge-Cap codewords
2) Build target primers and verify exact-match binding sites
3) Compute thermodynamic primer metrics
4) Evaluate off-target specificity on background strands
5) Verify search position accuracy
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Any, Dict, List

from Bio.Seq import Seq  # type: ignore[import-not-found]
from Bio.SeqUtils import MeltingTemp as mt  # type: ignore[import-not-found]
from Bio.SeqUtils import gc_fraction  # type: ignore[import-not-found]

from src.independent.edgecap.implementation import EdgeCapCRLLGenerator

try:
    import primer3  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    primer3 = None


@dataclass(frozen=True)
class PCRConditions:
    """
    Thermodynamic condition parameters for primer metric calculations.
    """
    mv_conc_mM: float = 50.0
    dv_conc_mM: float = 1.5
    dntp_conc_mM: float = 0.2
    primer_conc_nM: float = 250.0
    annealing_temp_c: float = 60.0


def _text_to_bits(text: str, *, bit_length: int = 8) -> List[str]:
    """
    Convert text into fixed-width bitstrings.
    """
    if bit_length <= 0:
        raise ValueError("bit_length must be > 0")
    return [format(ord(ch), f"0{bit_length}b") for ch in text]


def _encode_text_to_strand(
    text: str,
    *,
    generator: EdgeCapCRLLGenerator,
    bit_length: int = 8,
) -> str:
    """
    Encode a text string into a concatenated Edge-Cap DNA strand.
    """
    bits = _text_to_bits(text, bit_length=bit_length)
    return "".join(generator.encode_capped(b) for b in bits)


def _find_exact_subsequence_positions(sequence: str, subseq: str) -> List[int]:
    """
    Find all start indices of exact subsequence matches.
    """
    if not subseq:
        return []
    positions: List[int] = []
    start = 0
    while True:
        idx = sequence.find(subseq, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1
    return positions


def _count_primer_binding_sites(strand: str, primer: str) -> List[int]:
    """
    Return positions where a primer hybridises exactly on the forward strand.

    A primer hybridises to the reverse complement of its own sequence as it
    appears in the strand.
    """
    bound_target = str(Seq(primer).reverse_complement())
    return _find_exact_subsequence_positions(strand, bound_target)


def _primer_metrics_biopython(
    primer: str,
    conditions: PCRConditions,
) -> Dict[str, float]:
    """
    Compute condition-aware primer metrics.
    Reports length, GC content, Wallace-rule Tm, and nearest-neighbour Tm.
    """
    tm_nn = mt.Tm_NN(
        primer,
        dnac1=conditions.primer_conc_nM,
        dnac2=conditions.primer_conc_nM,
        Na=conditions.mv_conc_mM,
        Mg=conditions.dv_conc_mM,
        dNTPs=conditions.dntp_conc_mM,
    )
    return {
        "length_nt": float(len(primer)),
        "gc_percent": gc_fraction(primer) * 100.0,
        "tm_wallace_c": mt.Tm_Wallace(primer),
        "tm_nn_c": tm_nn,
    }


def _primer_metrics_primer3(
    primer: str,
    conditions: PCRConditions,
) -> Dict[str, float | str]:
    """
    Compute primer thermodynamic metrics with primer3.
    Reports Tm, hairpin formation temperature and free energy,
    and homodimer formation temperature and free energy.
    """
    if primer3 is None:
        return {"primer3": "not installed"}

    tm = primer3.calc_tm(
        primer,
        mv_conc=conditions.mv_conc_mM,
        dv_conc=conditions.dv_conc_mM,
        dntp_conc=conditions.dntp_conc_mM,
        dna_conc=conditions.primer_conc_nM,
    )
    hairpin = primer3.calc_hairpin(
        primer,
        mv_conc=conditions.mv_conc_mM,
        dv_conc=conditions.dv_conc_mM,
        dntp_conc=conditions.dntp_conc_mM,
        dna_conc=conditions.primer_conc_nM,
        temp_c=conditions.annealing_temp_c,
    )
    homodimer = primer3.calc_homodimer(
        primer,
        mv_conc=conditions.mv_conc_mM,
        dv_conc=conditions.dv_conc_mM,
        dntp_conc=conditions.dntp_conc_mM,
        dna_conc=conditions.primer_conc_nM,
        temp_c=conditions.annealing_temp_c,
    )
    return {
        "tm_c": float(tm),
        "hairpin_tm_c": float(hairpin.tm),
        "hairpin_dg": float(hairpin.dg),
        "homodimer_tm_c": float(homodimer.tm),
        "homodimer_dg": float(homodimer.dg),
    }


def _generate_random_payload(
    *,
    rng: random.Random,
    length: int,
    alphabet: str = string.ascii_lowercase + " ",
) -> str:
    """
    Generate a random text payload from the given alphabet.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    return "".join(rng.choice(alphabet) for _ in range(length))


def _generate_background_payloads_without_target(
    *,
    target_word: str,
    count: int,
    payload_length: int,
    seed: int,
) -> List[str]:
    """
    Generate random text payloads guaranteed not to contain the target word.
    Used for off-target specificity analysis.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    if payload_length < 1:
        raise ValueError("payload_length must be >= 1")
    if not target_word:
        raise ValueError("target_word must not be empty")

    rng = random.Random(seed)
    out: List[str] = []
    attempts = 0
    max_attempts = count * 100
    while len(out) < count and attempts < max_attempts:
        candidate = _generate_random_payload(rng=rng, length=payload_length)
        if target_word not in candidate:
            out.append(candidate)
        attempts += 1
    if len(out) != count:
        raise RuntimeError("failed to generate enough background payloads without target")
    return out


@dataclass
class WordPrimerSimulationResult:
    """
    Result of a single random-access hybridisation simulation.
    """
    payload_text: str
    target_word: str
    codeword_length_nt: int
    strand_length_nt: int
    encoded_strand: str
    target_segment: str
    target_primer: str
    binding_positions_nt: List[int]
    binding_positions_codeword_idx: List[int]
    expected_word_positions_char_idx: List[int]
    primer_metrics_biopython: Dict[str, float]
    primer_metrics_primer3: Dict[str, float | str]
    off_target_analysis: Dict[str, int]
    search_position_accurate: bool

    @property
    def binds(self) -> bool:
        return len(self.binding_positions_nt) > 0

    @property
    def exact_binding_site_count(self) -> int:
        return len(self.binding_positions_nt)


def simulate_word_primer_binding(
    *,
    payload_text: str,
    target_word: str,
    generator: EdgeCapCRLLGenerator,
    pcr_conditions: PCRConditions,
    bit_length: int = 8,
    background_count: int = 300,
    background_seed: int = 7,
) -> WordPrimerSimulationResult:
    """
    Simulate random-access hybridisation retrieval for a target word
    encoded within a concatenated Edge-Cap DNA strand.

    Reports:
    - Exact binding site positions (nt and codeword index)
    - Primer thermodynamic metrics (Tm, GC, hairpin, homodimer)
    - Off-target specificity against background strands
    - Search position accuracy (binding positions match expected character positions)
    """
    if not payload_text:
        raise ValueError("payload_text must not be empty")
    if not target_word:
        raise ValueError("target_word must not be empty")
    if len(target_word) > len(payload_text):
        raise ValueError("target_word length cannot exceed payload_text length")

    # Encode full payload into concatenated DNA strand
    encoded_strand = _encode_text_to_strand(
        payload_text, generator=generator, bit_length=bit_length
    )

    # Build target segment and reverse-complement primer
    target_bits = _text_to_bits(target_word, bit_length=bit_length)
    target_codewords = [generator.encode_capped(bits) for bits in target_bits]
    target_segment = "".join(target_codewords)
    target_primer = str(Seq(target_segment).reverse_complement())

    # Find exact hybridisation positions on the encoded strand
    binding_positions_nt = _count_primer_binding_sites(encoded_strand, target_primer)
    codeword_nt_len = generator.capped_length
    binding_positions_codeword_idx = [
        pos // codeword_nt_len for pos in binding_positions_nt
    ]

    # Find expected positions of target word in original text
    expected_word_positions_char_idx = _find_exact_subsequence_positions(
        payload_text, target_word
    )

    # Verify search position accuracy
    search_position_accurate = (
        binding_positions_codeword_idx == expected_word_positions_char_idx
    )

    # Compute primer thermodynamic metrics
    primer_metrics_biopython = _primer_metrics_biopython(target_primer, pcr_conditions)
    primer_metrics_primer3 = _primer_metrics_primer3(target_primer, pcr_conditions)

    # Off-target specificity: test primer against background strands
    background_payloads = _generate_background_payloads_without_target(
        target_word=target_word,
        count=background_count,
        payload_length=len(payload_text),
        seed=background_seed,
    )
    background_strands = [
        _encode_text_to_strand(text, generator=generator, bit_length=bit_length)
        for text in background_payloads
    ]
    off_target_binding_counts = [
        len(_count_primer_binding_sites(strand, target_primer))
        for strand in background_strands
    ]
    off_target_analysis = {
        "background_strands_tested": background_count,
        "strands_with_any_primer_hit": sum(
            1 for v in off_target_binding_counts if v > 0
        ),
        "total_off_target_primer_hits": sum(off_target_binding_counts),
        "max_primer_hits_in_single_background_strand": max(off_target_binding_counts),
    }

    return WordPrimerSimulationResult(
        payload_text=payload_text,
        target_word=target_word,
        codeword_length_nt=generator.capped_length,
        strand_length_nt=len(encoded_strand),
        encoded_strand=encoded_strand,
        target_segment=target_segment,
        target_primer=target_primer,
        binding_positions_nt=binding_positions_nt,
        binding_positions_codeword_idx=binding_positions_codeword_idx,
        expected_word_positions_char_idx=expected_word_positions_char_idx,
        primer_metrics_biopython=primer_metrics_biopython,
        primer_metrics_primer3=primer_metrics_primer3,
        off_target_analysis=off_target_analysis,
        search_position_accurate=search_position_accurate,
    )


def _as_json_ready(result: WordPrimerSimulationResult) -> Dict[str, Any]:
    """
    Convert simulation result to a compact JSON-ready dictionary.
    """
    return {
        "payload_text": result.payload_text,
        "target_word": result.target_word,
        "codeword_length_nt": result.codeword_length_nt,
        "strand_length_nt": result.strand_length_nt,
        "binds": result.binds,
        "exact_binding_site_count": result.exact_binding_site_count,
        "binding_positions_nt": result.binding_positions_nt,
        "binding_positions_codeword_idx": result.binding_positions_codeword_idx,
        "expected_word_positions_char_idx": result.expected_word_positions_char_idx,
        "search_position_accurate": result.search_position_accurate,
        "target_primer": result.target_primer,
        "target_segment": result.target_segment,
        "primer_metrics_biopython": result.primer_metrics_biopython,
        "primer_metrics_primer3": result.primer_metrics_primer3,
        "off_target_analysis": result.off_target_analysis,
    }


def main() -> None:
    """
    Example run for Section V evaluation.
    """
    payload_text = (
        "Two roads diverged in a yellow wood,\n"
        "And sorry I could not travel both\n"
        "And be one traveler, long I stood\n"
        "And looked down one as far as I could\n"
        "To where it bent in the undergrowth;\n"
        "\n"
        "Then took the other, as just as fair,\n"
        "And having perhaps the better claim,\n"
        "Because it was grassy and wanted wear;\n"
        "Though as for that the passing there\n"
        "Had worn them really about the same,\n"
        "\n"
        "And both that morning equally lay\n"
        "In leaves no step had trodden black.\n"
        "Oh, I kept the first for another day!\n"
        "Yet knowing how way leads on to way,\n"
        "I doubted if I should ever come back.\n"
        "\n"
        "I shall be telling this with a sigh\n"
        "Somewhere ages and ages hence:\n"
        "Two roads diverged in a wood, and I-\n"
        "I took the one less traveled by,\n"
        "And that has made all the difference."
    )
    target_word = "road"

    gen = EdgeCapCRLLGenerator(length=5, max_run=3, gc_lower=0.4, gc_upper=0.6)
    pcr_conditions = PCRConditions(
        mv_conc_mM=50.0,
        dv_conc_mM=1.5,
        dntp_conc_mM=0.2,
        primer_conc_nM=250.0,
        annealing_temp_c=60.0,
    )
    result = simulate_word_primer_binding(
        payload_text=payload_text,
        target_word=target_word,
        generator=gen,
        pcr_conditions=pcr_conditions,
        bit_length=8,
        background_count=300,
        background_seed=7,
    )

    out = _as_json_ready(result)
    print("=== edge-cap random-access simulation ===")
    for key, value in out.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()