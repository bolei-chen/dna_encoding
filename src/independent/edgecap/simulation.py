"""
Simulation utilities for Edge-Cap random-access retrieval.

This module demonstrates an in-silico workflow:
1) Encode text into concatenated Edge-Cap codewords
2) Build target primers and verify exact-match binding sites
3) Compute thermodynamic primer metrics
4) Evaluate off-target specificity using a thermodynamic binding model
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
    annealing_temp_c: float = 65.0


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


def _sequence_identity(seq_a: str, seq_b: str) -> float:
    """
    Compute fractional sequence identity between two equal-length strings.
    """
    if len(seq_a) != len(seq_b):
        raise ValueError("sequences must be the same length")
    matches = sum(a == b for a, b in zip(seq_a, seq_b))
    return matches / len(seq_a)


def _off_target_analysis(
    *,
    primer: str,
    background_strands: List[str],
    identity_threshold: float = 0.70,
) -> Dict[str, Any]:
    """
    Evaluate off-target binding specificity using a sequence identity model.

    For each background strand, a sliding window of primer length is scanned.
    A window is flagged as a potential off-target binding site if its sequence
    identity with the primer exceeds identity_threshold. A strand is considered
    an off-target hit if any such window is found.

    Args:
        primer: the target primer sequence
        background_strands: list of encoded DNA strands not containing the target
        identity_threshold: minimum fractional identity to flag a window as a
            hit (default 0.70, i.e. 70%)

    Returns:
        Dictionary summarising off-target binding events across all background
        strands.
    """
    primer_len = len(primer)
    strands_with_hit = 0
    total_hits = 0
    max_hits_single_strand = 0
    max_identity_observed = 0.0

    for strand_idx, strand in enumerate(background_strands):
        if (strand_idx + 1) % 50 == 0:
            print(f"  ...processed {strand_idx + 1}/{len(background_strands)} strands")

        hits_this_strand = 0
        for i in range(len(strand) - primer_len + 1):
            subseq = strand[i: i + primer_len]
            identity = _sequence_identity(primer, subseq)
            if identity > max_identity_observed:
                max_identity_observed = identity
            if identity >= identity_threshold:
                hits_this_strand += 1

        total_hits += hits_this_strand
        if hits_this_strand > 0:
            strands_with_hit += 1
        if hits_this_strand > max_hits_single_strand:
            max_hits_single_strand = hits_this_strand

    return {
        "model": "sequence_identity",
        "identity_threshold": identity_threshold,
        "background_strands_tested": len(background_strands),
        "strands_with_any_off_target_hit": strands_with_hit,
        "total_off_target_hits": total_hits,
        "max_hits_in_single_strand": max_hits_single_strand,
        "off_target_hit_rate": strands_with_hit / len(background_strands),
        "max_identity_observed": round(max_identity_observed, 4),
    }


def _thermodynamic_off_target_analysis(
    *,
    primer: str,
    background_strands: List[str],
    conditions: PCRConditions,
    dg_threshold_kcal: float = -9.0,
) -> Dict[str, Any]:
    """
    Evaluate off-target binding specificity using a thermodynamic model.

    For each background strand, all subsequences of the same length as the
    primer are extracted. Primer3's heterodimer calculation computes the binding
    free energy (delta G) between the primer and each subsequence. A subsequence
    is flagged as a potential off-target binding site if its delta G is at or
    below dg_threshold_kcal, indicating thermodynamically stable binding under
    the given conditions.

    Requires primer3 to be installed; returns a skipped sentinel dict if not.

    Args:
        primer: the target primer sequence
        background_strands: list of encoded DNA strands not containing the target
        conditions: thermodynamic conditions for the heterodimer calculation
        dg_threshold_kcal: delta G threshold in kcal/mol (default -9.0,
            a standard threshold for stable hybridisation under typical
            lab conditions)

    Returns:
        Dictionary summarising off-target binding events across all background
        strands.
    """
    if primer3 is None:
        return {"model": "thermodynamic_heterodimer", "skipped": "primer3 not installed"}

    primer_len = len(primer)
    strands_with_hit = 0
    total_hits = 0
    max_hits_single_strand = 0
    min_dg_observed = 0.0

    for strand_idx, strand in enumerate(background_strands):
        if (strand_idx + 1) % 50 == 0:
            print(f"  ...processed {strand_idx + 1}/{len(background_strands)} strands")

        hits_this_strand = 0
        for i in range(len(strand) - primer_len + 1):
            subseq = strand[i: i + primer_len]
            result = primer3.calc_heterodimer(
                primer,
                subseq,
                mv_conc=conditions.mv_conc_mM,
                dv_conc=conditions.dv_conc_mM,
                dntp_conc=conditions.dntp_conc_mM,
                dna_conc=conditions.primer_conc_nM,
                temp_c=conditions.annealing_temp_c,
            )
            dg_kcal = float(result.dg) / 1000.0
            if dg_kcal < min_dg_observed:
                min_dg_observed = dg_kcal
            if dg_kcal <= dg_threshold_kcal:
                hits_this_strand += 1

        total_hits += hits_this_strand
        if hits_this_strand > 0:
            strands_with_hit += 1
        if hits_this_strand > max_hits_single_strand:
            max_hits_single_strand = hits_this_strand

    return {
        "model": "thermodynamic_heterodimer",
        "dg_threshold_kcal_mol": dg_threshold_kcal,
        "background_strands_tested": len(background_strands),
        "strands_with_any_off_target_hit": strands_with_hit,
        "total_off_target_hits": total_hits,
        "max_hits_in_single_strand": max_hits_single_strand,
        "off_target_hit_rate": strands_with_hit / len(background_strands),
        "min_dg_observed_kcal_mol": round(min_dg_observed, 4),
    }


def _three_prime_anchored_off_target_analysis(
    *,
    primer: str,
    background_strands: List[str],
    conditions: PCRConditions,
    three_prime_len: int = 8,
    dg_threshold_kcal: float = -6.0,
) -> Dict[str, Any]:
    """
    Evaluate off-target binding specificity using a 3'-anchored thermodynamic model.

    In PCR, spurious extension only occurs if the 3' end of the primer finds a
    stable binding site — the rest of the primer is irrelevant if the 3' end is
    dangling. This function therefore slides the 3' terminal sub-primer
    independently across the full length of each background strand, asking:
    can the 3' end land stably anywhere on this strand?

    A strand is flagged as a hit if the minimum ΔG observed across all 3'
    sub-primer windows is at or below dg_threshold_kcal. This is fully decoupled
    from the full-primer scan — it is the most conservative and biologically
    meaningful PCR specificity check.

    Requires primer3 to be installed; returns a skipped sentinel dict if not.

    Args:
        primer: the target primer sequence
        background_strands: list of encoded DNA strands not containing the target
        conditions: thermodynamic conditions
        three_prime_len: number of 3'-terminal bases to evaluate (default 8,
            roughly one codeword's worth)
        dg_threshold_kcal: ΔG threshold in kcal/mol for the 3' sub-primer
            (default -6.0; appropriate for a short fragment)

    Returns:
        Dictionary summarising 3'-anchored off-target binding events.
    """
    if primer3 is None:
        return {"model": "three_prime_anchored", "skipped": "primer3 not installed"}

    three_prime_sub = primer[-three_prime_len:]

    strands_with_hit = 0
    total_hits = 0
    max_hits_single_strand = 0
    min_dg_3p_observed = 0.0

    for strand_idx, strand in enumerate(background_strands):
        if (strand_idx + 1) % 50 == 0:
            print(f"  ...processed {strand_idx + 1}/{len(background_strands)} strands")

        # Slide the 3' sub-primer independently across the full strand
        strand_min_dg = 0.0
        for j in range(len(strand) - three_prime_len + 1):
            sub_window = strand[j: j + three_prime_len]
            result = primer3.calc_heterodimer(
                three_prime_sub,
                sub_window,
                mv_conc=conditions.mv_conc_mM,
                dv_conc=conditions.dv_conc_mM,
                dntp_conc=conditions.dntp_conc_mM,
                dna_conc=conditions.primer_conc_nM,
                temp_c=conditions.annealing_temp_c,
            )
            dg_3p = float(result.dg) / 1000.0
            if dg_3p < strand_min_dg:
                strand_min_dg = dg_3p

        if strand_min_dg < min_dg_3p_observed:
            min_dg_3p_observed = strand_min_dg
        if strand_min_dg <= dg_threshold_kcal:
            strands_with_hit += 1
            total_hits += 1

    return {
        "model": "three_prime_anchored",
        "three_prime_len_nt": three_prime_len,
        "dg_threshold_kcal_mol": dg_threshold_kcal,
        "background_strands_tested": len(background_strands),
        "strands_with_any_off_target_hit": strands_with_hit,
        "total_off_target_hits": total_hits,
        "off_target_hit_rate": strands_with_hit / len(background_strands),
        "min_dg_3p_observed_kcal_mol": round(min_dg_3p_observed, 4),
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
    off_target_analysis_identity: Dict[str, Any]
    off_target_analysis_thermodynamic: Dict[str, Any]
    off_target_analysis_three_prime: Dict[str, Any]
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
    identity_threshold: float = 0.70,
    dg_threshold_kcal: float = -9.0,
    three_prime_len: int = 8,
) -> WordPrimerSimulationResult:
    """
    Simulate random-access hybridisation retrieval for a target word
    encoded within a concatenated Edge-Cap DNA strand.

    Reports:
    - Exact binding site positions (nt and codeword index)
    - Primer thermodynamic metrics (Tm, GC, hairpin, homodimer)
    - Off-target specificity via thermodynamic heterodimer model
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

    # Generate background strands and run thermodynamic off-target analysis
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
    print(
        f"Running sequence-identity off-target analysis on {background_count} "
        f"background strands ({len(background_strands[0])} nt each, "
        f"identity threshold = {identity_threshold:.0%})..."
    )
    off_target_analysis_identity = _off_target_analysis(
        primer=target_primer,
        background_strands=background_strands,
        identity_threshold=identity_threshold,
    )

    print(
        f"Running thermodynamic off-target analysis on {background_count} "
        f"background strands ({len(background_strands[0])} nt each, "
        f"dG threshold = {dg_threshold_kcal} kcal/mol)..."
    )
    off_target_analysis_thermodynamic = _thermodynamic_off_target_analysis(
        primer=target_primer,
        background_strands=background_strands,
        conditions=pcr_conditions,
        dg_threshold_kcal=dg_threshold_kcal,
    )

    print(
        f"Running 3'-anchored thermodynamic off-target analysis on {background_count} "
        f"background strands (3' region = {three_prime_len} nt)..."
    )
    off_target_analysis_three_prime = _three_prime_anchored_off_target_analysis(
        primer=target_primer,
        background_strands=background_strands,
        conditions=pcr_conditions,
        three_prime_len=three_prime_len,
        dg_threshold_kcal=dg_threshold_kcal,
    )

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
        off_target_analysis_identity=off_target_analysis_identity,
        off_target_analysis_thermodynamic=off_target_analysis_thermodynamic,
        off_target_analysis_three_prime=off_target_analysis_three_prime,
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
        "off_target_analysis_identity": result.off_target_analysis_identity,
        "off_target_analysis_thermodynamic": result.off_target_analysis_thermodynamic,
        "off_target_analysis_three_prime": result.off_target_analysis_three_prime,
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
        annealing_temp_c=68.0,
    )
    result = simulate_word_primer_binding(
        payload_text=payload_text,
        target_word=target_word,
        generator=gen,
        pcr_conditions=pcr_conditions,
        bit_length=8,
        background_count=300,
        background_seed=7,
        identity_threshold=0.70,
        dg_threshold_kcal=-4.0,
        three_prime_len=8,
    )

    out = _as_json_ready(result)
    print("=== edge-cap random-access simulation ===")
    for key, value in out.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()