"""
Shared evaluation pipeline for random-access DNA retrieval simulations.
"""

from __future__ import annotations


import math
import random
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from Bio.Seq import Seq  # type: ignore[import-not-found]
from Bio.SeqUtils import MeltingTemp as mt  # type: ignore[import-not-found]
from Bio.SeqUtils import gc_fraction  # type: ignore[import-not-found]

try:
    import primer3  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    primer3 = None


def info_density(cardinality: int, codeword_length: int) -> float:
    """
    Calculate the information density of a codeword.
    The information density is the ratio of the number of codewords to the number of possible codewords.
    """
    return math.floor(math.log2(cardinality)) / codeword_length


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


class CodewordEncoder(Protocol):
    """
    Minimal interface required by the evaluation pipeline.
    """

    name: str
    codeword_length_nt: int

    def encode_codeword(self, bits: str) -> str:
        """
        Encode one fixed-width bitstring into one DNA codeword.
        """


def _maybe_reset_for_new_strand(generator: CodewordEncoder) -> None:
    """
    Reset encoder state when supported by the encoder implementation.
    """

    reset_fn = getattr(generator, "reset_for_new_strand", None)
    if callable(reset_fn):
        reset_fn()


@dataclass
class WordPrimerSimulationResult:
    """
    Result of a single random-access hybridisation simulation.
    """

    generator_name: str
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


def text_to_bits(text: str, *, bit_length: int = 8) -> List[str]:
    """
    Convert text into fixed-width bitstrings.
    """

    if bit_length <= 0:
        raise ValueError("bit_length must be > 0")
    return [format(ord(ch), f"0{bit_length}b") for ch in text]


def encode_text_to_strand(
    text: str,
    *,
    generator: CodewordEncoder,
    bit_length: int = 8,
) -> str:
    """
    Encode a text string into a concatenated DNA strand.
    """

    _maybe_reset_for_new_strand(generator)
    bits = text_to_bits(text, bit_length=bit_length)
    return "".join(generator.encode_codeword(b) for b in bits)


def find_exact_subsequence_positions(sequence: str, subseq: str) -> List[int]:
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


def count_primer_binding_sites(strand: str, primer: str) -> List[int]:
    """
    Return positions where a primer hybridises exactly on the forward strand.
    """

    bound_target = str(Seq(primer).reverse_complement())
    return find_exact_subsequence_positions(strand, bound_target)


def primer_metrics_biopython(
    primer: str,
    conditions: PCRConditions,
) -> Dict[str, float]:
    """
    Compute condition-aware primer metrics.
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


def primer_metrics_primer3(
    primer: str,
    conditions: PCRConditions,
) -> Dict[str, float | str]:
    """
    Compute primer thermodynamic metrics with primer3.
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
    if len(seq_a) != len(seq_b):
        raise ValueError("sequences must be the same length")
    matches = sum(a == b for a, b in zip(seq_a, seq_b))
    return matches / len(seq_a)


def off_target_analysis_identity(
    *,
    primer: str,
    background_strands: List[str],
    identity_threshold: float = 0.70,
) -> Dict[str, Any]:
    """
    Evaluate off-target binding specificity using a sequence identity model.
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
            subseq = strand[i : i + primer_len]
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


def off_target_analysis_thermodynamic(
    *,
    primer: str,
    background_strands: List[str],
    conditions: PCRConditions,
    dg_threshold_kcal: float = -9.0,
) -> Dict[str, Any]:
    """
    Evaluate off-target binding specificity using a thermodynamic model.
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
            subseq = strand[i : i + primer_len]
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


def off_target_analysis_three_prime(
    *,
    primer: str,
    background_strands: List[str],
    conditions: PCRConditions,
    three_prime_len: int = 8,
    dg_threshold_kcal: float = -6.0,
) -> Dict[str, Any]:
    """
    Evaluate off-target specificity using a 3'-anchored thermodynamic model.
    """

    if primer3 is None:
        return {"model": "three_prime_anchored", "skipped": "primer3 not installed"}

    three_prime_sub = primer[-three_prime_len:]
    strands_with_hit = 0
    total_hits = 0
    min_dg_3p_observed = 0.0

    for strand_idx, strand in enumerate(background_strands):
        if (strand_idx + 1) % 50 == 0:
            print(f"  ...processed {strand_idx + 1}/{len(background_strands)} strands")

        strand_min_dg = 0.0
        for j in range(len(strand) - three_prime_len + 1):
            sub_window = strand[j : j + three_prime_len]
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
    if length < 1:
        raise ValueError("length must be >= 1")
    return "".join(rng.choice(alphabet) for _ in range(length))


def generate_background_payloads_without_target(
    *,
    target_word: str,
    count: int,
    payload_length: int,
    seed: int,
) -> List[str]:
    """
    Generate random text payloads guaranteed not to contain the target word.
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


def simulate_word_primer_binding(
    *,
    payload_text: str,
    target_word: str,
    generator: CodewordEncoder,
    pcr_conditions: PCRConditions,
    bit_length: int = 8,
    background_count: int = 300,
    background_seed: int = 7,
    identity_threshold: float = 0.70,
    dg_threshold_kcal: float = -9.0,
    dg_threshold_kcal_three_prime: float = -4.0,
    three_prime_len: int = 8,
) -> WordPrimerSimulationResult:
    """
    Simulate random-access hybridisation retrieval for one generator.
    """

    if not payload_text:
        raise ValueError("payload_text must not be empty")
    if not target_word:
        raise ValueError("target_word must not be empty")
    if len(target_word) > len(payload_text):
        raise ValueError("target_word length cannot exceed payload_text length")

    encoded_strand = encode_text_to_strand(
        payload_text,
        generator=generator,
        bit_length=bit_length,
    )

    target_bits = text_to_bits(target_word, bit_length=bit_length)
    target_codewords = [generator.encode_codeword(bits) for bits in target_bits]
    target_segment = "".join(target_codewords)
    target_primer = str(Seq(target_segment).reverse_complement())

    binding_positions_nt = count_primer_binding_sites(encoded_strand, target_primer)
    codeword_nt_len = generator.codeword_length_nt
    binding_positions_codeword_idx = [
        pos // codeword_nt_len for pos in binding_positions_nt
    ]

    expected_word_positions_char_idx = find_exact_subsequence_positions(
        payload_text,
        target_word,
    )
    search_position_accurate = (
        binding_positions_codeword_idx == expected_word_positions_char_idx
    )

    bp_metrics = primer_metrics_biopython(target_primer, pcr_conditions)
    p3_metrics = primer_metrics_primer3(target_primer, pcr_conditions)

    background_payloads = generate_background_payloads_without_target(
        target_word=target_word,
        count=background_count,
        payload_length=len(payload_text),
        seed=background_seed,
    )
    background_strands = [
        encode_text_to_strand(text, generator=generator, bit_length=bit_length)
        for text in background_payloads
    ]

    print(
        f"Running sequence-identity off-target analysis on {background_count} "
        f"background strands ({len(background_strands[0])} nt each, "
        f"identity threshold = {identity_threshold:.0%})..."
    )
    id_off_target = off_target_analysis_identity(
        primer=target_primer,
        background_strands=background_strands,
        identity_threshold=identity_threshold,
    )

    print(
        f"Running thermodynamic off-target analysis on {background_count} "
        f"background strands ({len(background_strands[0])} nt each, "
        f"dG threshold = {dg_threshold_kcal} kcal/mol)..."
    )
    thermo_off_target = off_target_analysis_thermodynamic(
        primer=target_primer,
        background_strands=background_strands,
        conditions=pcr_conditions,
        dg_threshold_kcal=dg_threshold_kcal,
    )

    print(
        f"Running 3'-anchored thermodynamic off-target analysis on {background_count} "
        f"background strands (3' region = {three_prime_len} nt, "
        f"dG threshold = {dg_threshold_kcal_three_prime} kcal/mol)..."
    )
    three_prime_off_target = off_target_analysis_three_prime(
        primer=target_primer,
        background_strands=background_strands,
        conditions=pcr_conditions,
        three_prime_len=three_prime_len,
        dg_threshold_kcal=dg_threshold_kcal_three_prime,
    )

    return WordPrimerSimulationResult(
        generator_name=generator.name,
        payload_text=payload_text,
        target_word=target_word,
        codeword_length_nt=codeword_nt_len,
        strand_length_nt=len(encoded_strand),
        encoded_strand=encoded_strand,
        target_segment=target_segment,
        target_primer=target_primer,
        binding_positions_nt=binding_positions_nt,
        binding_positions_codeword_idx=binding_positions_codeword_idx,
        expected_word_positions_char_idx=expected_word_positions_char_idx,
        primer_metrics_biopython=bp_metrics,
        primer_metrics_primer3=p3_metrics,
        off_target_analysis_identity=id_off_target,
        off_target_analysis_thermodynamic=thermo_off_target,
        off_target_analysis_three_prime=three_prime_off_target,
        search_position_accurate=search_position_accurate,
    )


def as_json_ready(result: WordPrimerSimulationResult) -> Dict[str, Any]:
    """
    Convert simulation result to a compact JSON-ready dictionary.
    """

    return {
        "generator_name": result.generator_name,
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


def default_payload_text() -> str:
    """
    Default payload for Section V simulation runs.
    """

    return (
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


def exhaustive_ascii_reverse_complement_report(
    *,
    generator: CodewordEncoder,
    bit_length: int = 8,
) -> Dict[str, Any]:
    """
    Check RC collisions exhaustively across ASCII code points 0..127.
    """

    unique_chars = [chr(i) for i in range(128)]
    bits_by_char = {ch: format(ord(ch), f"0{bit_length}b") for ch in unique_chars}

    codeword_by_char: Dict[str, str] = {}
    char_by_codeword: Dict[str, str] = {}

    for ch in unique_chars:
        codeword = encode_text_to_strand(
            ch,
            generator=generator,
            bit_length=bit_length,
        )
        codeword_by_char[ch] = codeword
        char_by_codeword[codeword] = ch

    collisions: List[Dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for ch, codeword in codeword_by_char.items():
        rc = str(Seq(codeword).reverse_complement())
        if rc not in char_by_codeword:
            continue

        other = char_by_codeword[rc]
        pair = tuple(sorted((ch, other)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        collisions.append(
            {
                "char_a": ch,
                "char_b": other,
                "ord_a": ord(ch),
                "ord_b": ord(other),
                "char_a_repr": repr(ch),
                "char_b_repr": repr(other),
                "codeword_a": codeword,
                "codeword_b": codeword_by_char[other],
                "is_self_reverse_complement": ch == other,
            }
        )

    return {
        "symbol_set": "ascii_0_to_127",
        "alphabet_size": len(unique_chars),
        "collision_count": len(collisions),
        "has_collision": len(collisions) > 0,
        "collisions": collisions,
    }
