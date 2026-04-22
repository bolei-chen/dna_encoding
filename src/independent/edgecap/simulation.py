"""
Simulation utilities for Edge-Cap random-access retrieval.

This module demonstrates an in-silico workflow:
1) Encode text into concatenated Edge-Cap codewords
2) Build a reverse-complement primer for a target word
3) Verify exact-match binding sites on clean strands
4) Report primer quality metrics (Tm, GC, simple secondary structure checks)
5) Estimate post-binding PCR amplification
"""

from __future__ import annotations

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


def _text_to_bits(text: str, *, bit_length: int = 8) -> List[str]:
    """
    Convert text into fixed-width bitstrings.
    """
    if bit_length <= 0:
        raise ValueError("bit_length must be > 0")
    return [format(ord(ch), f"0{bit_length}b") for ch in text]


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
    Return positions where a primer binds exactly on the forward strand.

    A primer binds to the complement of its own sequence, i.e., to the reverse complement
    of the primer sequence represented in the same 5'->3' direction as `strand`.
    """
    bound_target = str(Seq(primer).reverse_complement())
    return _find_exact_subsequence_positions(strand, bound_target)


def _primer_metrics_biopython(primer: str) -> Dict[str, float]:
    """
    Compute basic primer metrics with Biopython.
    """
    return {
        "length_nt": float(len(primer)),
        "gc_percent": gc_fraction(primer) * 100.0,
        "tm_wallace_c": mt.Tm_Wallace(primer),
        "tm_nn_c": mt.Tm_NN(primer),
    }


def _primer_metrics_primer3(primer: str) -> Dict[str, float | str]:
    """
    Compute optional primer quality metrics with primer3 (if installed).
    """
    if primer3 is None:
        return {"primer3": "not installed"}

    hairpin = primer3.calc_hairpin(primer)
    homodimer = primer3.calc_homodimer(primer)
    return {
        "tm_c": float(primer3.calc_tm(primer)),
        "hairpin_tm_c": float(hairpin.tm),
        "hairpin_dg": float(hairpin.dg),
        "homodimer_tm_c": float(homodimer.tm),
        "homodimer_dg": float(homodimer.dg),
    }


def _estimate_pcr_copies(
    *,
    initial_templates: int,
    cycles: int = 25,
    efficiency: float = 0.9,
) -> float:
    """
    Estimate PCR amplification using a simple exponential model.
    """
    if initial_templates < 0:
        raise ValueError("initial_templates must be >= 0")
    if cycles < 0:
        raise ValueError("cycles must be >= 0")
    if not (0.0 <= efficiency <= 1.0):
        raise ValueError("efficiency must be in [0.0, 1.0]")

    return float(initial_templates) * ((1.0 + efficiency) ** cycles)


@dataclass
class WordPrimerSimulationResult:
    payload_text: str
    target_word: str
    codeword_length_nt: int
    strand_length_nt: int
    encoded_strand: str
    target_segment: str
    primer: str
    binding_positions_nt: List[int]
    binding_positions_codeword_idx: List[int]
    expected_word_positions_char_idx: List[int]
    primer_metrics_biopython: Dict[str, float]
    primer_metrics_primer3: Dict[str, float | str]
    estimated_post_pcr_copies: float

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
    bit_length: int = 8,
    pcr_cycles: int = 25,
    pcr_efficiency: float = 0.9,
) -> WordPrimerSimulationResult:
    """
    Simulate exact-match random-access retrieval with a target-word primer.
    """
    if not payload_text:
        raise ValueError("payload_text must not be empty")
    if not target_word:
        raise ValueError("target_word must not be empty")
    if len(target_word) > len(payload_text):
        raise ValueError("target_word length cannot exceed payload_text length")

    payload_bits = _text_to_bits(payload_text, bit_length=bit_length)
    payload_codewords = [generator.encode_capped(bits) for bits in payload_bits]
    encoded_strand = "".join(payload_codewords)

    target_bits = _text_to_bits(target_word, bit_length=bit_length)
    target_codewords = [generator.encode_capped(bits) for bits in target_bits]
    target_segment = "".join(target_codewords)
    primer = str(Seq(target_segment).reverse_complement())

    binding_positions_nt = _count_primer_binding_sites(encoded_strand, primer)
    codeword_nt_len = generator.capped_length
    binding_positions_codeword_idx = [pos // codeword_nt_len for pos in binding_positions_nt]
    expected_word_positions_char_idx = _find_exact_subsequence_positions(payload_text, target_word)

    primer_metrics_biopython = _primer_metrics_biopython(primer)
    primer_metrics_primer3 = _primer_metrics_primer3(primer)
    estimated_post_pcr_copies = _estimate_pcr_copies(
        initial_templates=len(binding_positions_nt),
        cycles=pcr_cycles,
        efficiency=pcr_efficiency,
    )

    return WordPrimerSimulationResult(
        payload_text=payload_text,
        target_word=target_word,
        codeword_length_nt=generator.capped_length,
        strand_length_nt=len(encoded_strand),
        encoded_strand=encoded_strand,
        target_segment=target_segment,
        primer=primer,
        binding_positions_nt=binding_positions_nt,
        binding_positions_codeword_idx=binding_positions_codeword_idx,
        expected_word_positions_char_idx=expected_word_positions_char_idx,
        primer_metrics_biopython=primer_metrics_biopython,
        primer_metrics_primer3=primer_metrics_primer3,
        estimated_post_pcr_copies=estimated_post_pcr_copies,
    )


def _as_json_ready(result: WordPrimerSimulationResult) -> Dict[str, Any]:
    """
    Convert dataclass result to a compact JSON-ready dictionary.
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
        "primer": result.primer,
        "target_segment": result.target_segment,
        "primer_metrics_biopython": result.primer_metrics_biopython,
        "primer_metrics_primer3": result.primer_metrics_primer3,
        "estimated_post_pcr_copies": result.estimated_post_pcr_copies,
    }


def main() -> None:
    """
    Example run for Section V-style evaluation.
    """
    payload_text = "this is a cat and another cat"
    target_word = "cat"

    gen = EdgeCapCRLLGenerator(length=5, max_run=3, gc_lower=0.4, gc_upper=0.6)
    result = simulate_word_primer_binding(
        payload_text=payload_text,
        target_word=target_word,
        generator=gen,
        bit_length=8,
        pcr_cycles=25,
        pcr_efficiency=0.9,
    )

    out = _as_json_ready(result)
    print("=== edge-cap random-access simulation ===")
    for key, value in out.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
