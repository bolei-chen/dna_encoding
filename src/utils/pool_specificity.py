"""
Pool-wide in-silico specificity analysis for hybridisation probes.

The analysis treats a probe as the reverse complement of a known encoded
payload substring.  It scans every same-length window in an encoded pool,
separates intended repeated sites from unintended competitors, and reports
exact matches, Hamming/edit-distance near matches, and duplex thermodynamics.

This module models probe hybridisation.  It does not model PCR amplification,
which would require a primer pair and amplicon geometry.
"""

from __future__ import annotations

import random
import string
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Sequence

from Bio.Seq import Seq  # type: ignore[import-not-found]
from tqdm.auto import tqdm

from src.utils.eval import (
    CodewordEncoder,
    PCRConditions,
    encode_text_to_strand,
    find_exact_subsequence_positions,
)

try:
    import primer3  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional at import, required for default scoring
    primer3 = None


PRINTABLE_ASCII = string.ascii_letters + string.digits + string.punctuation + " "


@dataclass(frozen=True)
class EncodedPoolStrand:
    """
    One payload and its encoded DNA representation.
    """

    name: str
    payload: str
    dna: str
    is_target_positive: bool


@dataclass(frozen=True)
class EncodedPool:
    """
    Reproducible target-positive and target-absent encoded strands.
    """

    generator_name: str
    codeword_length_nt: int
    seed: int
    target_words: tuple[str, ...]
    strands: tuple[EncodedPoolStrand, ...]


@dataclass(frozen=True)
class WindowSite:
    """
    Representative location and distances for one unique competitor sequence.
    """

    sequence: str
    strand_name: str
    start_nt: int
    hamming_distance: int
    edit_distance: int
    occurrence_count: int
    codeword_aligned: bool


@dataclass(frozen=True)
class DuplexScore:
    """
    Nearest-neighbour heterodimer score for a probe/window pair.
    """

    dg_kcal_mol: float
    tm_c: float


ThermodynamicScorer = Callable[[str, str, PCRConditions], DuplexScore]


def _validate_target_words(target_words: Sequence[str]) -> tuple[str, ...]:
    words = tuple(target_words)
    if not words:
        raise ValueError("target_words must not be empty")
    if any(not word for word in words):
        raise ValueError("target words must not be empty")
    if len(set(words)) != len(words):
        raise ValueError("target words must be unique")
    return words


def generate_target_absent_payloads(
    *,
    target_words: Sequence[str],
    count: int,
    payload_length: int,
    seed: int,
    alphabet: str = PRINTABLE_ASCII,
) -> list[str]:
    """
    Generate fixed-seed printable-ASCII payloads containing none of the targets.
    """

    words = _validate_target_words(target_words)
    if count < 1:
        raise ValueError("count must be >= 1")
    if payload_length < 1:
        raise ValueError("payload_length must be >= 1")
    if not alphabet:
        raise ValueError("alphabet must not be empty")

    rng = random.Random(seed)
    payloads: list[str] = []
    max_attempts = count * 100
    for _ in range(max_attempts):
        candidate = "".join(rng.choice(alphabet) for _ in range(payload_length))
        if all(word not in candidate for word in words):
            payloads.append(candidate)
            if len(payloads) == count:
                return payloads
    raise RuntimeError("failed to generate enough target-absent payloads")


def build_encoded_pool(
    *,
    generator: CodewordEncoder,
    positive_payload: str,
    target_words: Sequence[str],
    negative_count: int,
    seed: int,
    bit_length: int = 8,
    negative_payload_length: int | None = None,
    alphabet: str = PRINTABLE_ASCII,
) -> EncodedPool:
    """
    Build one target-positive strand and fixed-seed target-absent strands.
    """

    words = _validate_target_words(target_words)
    if not positive_payload:
        raise ValueError("positive_payload must not be empty")
    missing = [word for word in words if word not in positive_payload]
    if missing:
        raise ValueError(f"target words absent from positive payload: {missing}")

    payload_length = (
        len(positive_payload)
        if negative_payload_length is None
        else negative_payload_length
    )
    negatives = generate_target_absent_payloads(
        target_words=words,
        count=negative_count,
        payload_length=payload_length,
        seed=seed,
        alphabet=alphabet,
    )

    strands = [
        EncodedPoolStrand(
            name="positive",
            payload=positive_payload,
            dna=encode_text_to_strand(
                positive_payload,
                generator=generator,
                bit_length=bit_length,
            ),
            is_target_positive=True,
        )
    ]
    for index, payload in enumerate(negatives):
        strands.append(
            EncodedPoolStrand(
                name=f"negative_{index:04d}",
                payload=payload,
                dna=encode_text_to_strand(
                    payload,
                    generator=generator,
                    bit_length=bit_length,
                ),
                is_target_positive=False,
            )
        )

    return EncodedPool(
        generator_name=generator.name,
        codeword_length_nt=generator.codeword_length_nt,
        seed=seed,
        target_words=words,
        strands=tuple(strands),
    )


def hamming_distance(left: str, right: str) -> int:
    """
    Return substitution distance between equal-length sequences.
    """

    if len(left) != len(right):
        raise ValueError("Hamming distance requires equal-length sequences")
    return sum(a != b for a, b in zip(left, right))


def _make_edit_distance(pattern: str) -> Callable[[str], int]:
    """
    Build a Myers bit-vector Levenshtein scorer for patterns up to 63 symbols.

    Pool probes in this study are 21-28 nt.  The bit-vector implementation makes
    exhaustive same-window-length edit-distance scans practical.
    """

    if not pattern:
        raise ValueError("pattern must not be empty")
    if len(pattern) > 63:
        raise ValueError("bit-vector edit distance supports patterns up to 63 nt")

    masks: dict[str, int] = {}
    for index, symbol in enumerate(pattern):
        masks[symbol] = masks.get(symbol, 0) | (1 << index)
    high_bit = 1 << (len(pattern) - 1)
    all_bits = (1 << len(pattern)) - 1

    def distance(text: str) -> int:
        positive = all_bits
        negative = 0
        score = len(pattern)

        for symbol in text:
            equal = masks.get(symbol, 0)
            xv = equal | negative
            xh = (((equal & positive) + positive) ^ positive) | equal
            ph = negative | ~(xh | positive)
            mh = positive & xh
            if ph & high_bit:
                score += 1
            elif mh & high_bit:
                score -= 1
            ph = ((ph << 1) | 1) & all_bits
            mh = (mh << 1) & all_bits
            positive = (mh | ~(xv | ph)) & all_bits
            negative = ph & xv
        return score

    return distance


def _default_thermodynamic_scorer(
    probe: str,
    template_window: str,
    conditions: PCRConditions,
) -> DuplexScore:
    if primer3 is None:
        raise RuntimeError(
            "primer3-py is required for thermodynamic scoring; "
            "install it or provide a thermodynamic_scorer"
        )
    result = primer3.calc_heterodimer(
        probe,
        template_window,
        mv_conc=conditions.mv_conc_mM,
        dv_conc=conditions.dv_conc_mM,
        dntp_conc=conditions.dntp_conc_mM,
        dna_conc=conditions.primer_conc_nM,
        temp_c=conditions.annealing_temp_c,
    )
    return DuplexScore(
        dg_kcal_mol=float(result.dg) / 1000.0,
        tm_c=float(result.tm),
    )


def _weighted_percentile(
    values_and_weights: Sequence[tuple[float, int]],
    fraction: float,
) -> float | None:
    if not values_and_weights:
        return None
    ordered = sorted(values_and_weights)
    total_weight = sum(weight for _, weight in ordered)
    threshold = fraction * max(0, total_weight - 1)
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative > threshold:
            return value
    return ordered[-1][0]


def _select_thermodynamic_candidates(
    competitors: dict[str, dict[str, Any]],
    *,
    limit: int | None,
    random_control_count: int,
    seed: int,
) -> tuple[list[str], bool, str]:
    sequences = list(competitors)
    if limit is None or limit >= len(sequences):
        return sequences, True, "all_unique_off_target_windows"
    if limit < 1:
        raise ValueError("thermodynamic_candidate_limit must be >= 1 or None")

    ranked = sorted(
        sequences,
        key=lambda seq: (
            competitors[seq]["edit_distance"],
            competitors[seq]["hamming_distance"],
            seq,
        ),
    )
    close_count = max(0, limit - random_control_count)
    selected = ranked[:close_count]
    remaining = ranked[close_count:]
    rng = random.Random(seed)
    sample_count = min(random_control_count, limit - len(selected), len(remaining))
    if sample_count:
        selected.extend(rng.sample(remaining, sample_count))
    return (
        selected,
        False,
        "nearest_edit_hamming_candidates_plus_seeded_random_controls",
    )


def analyze_probe_against_pool(
    *,
    pool: EncodedPool,
    generator: CodewordEncoder,
    target_word: str,
    conditions: PCRConditions,
    bit_length: int = 8,
    thermodynamic_candidate_limit: int | None = None,
    thermodynamic_random_controls: int = 100,
    thermodynamic_scorer: ThermodynamicScorer | None = None,
    representative_site_limit: int = 10,
    show_progress: bool = False,
) -> dict[str, Any]:
    """
    Exhaustively scan one target probe against all same-length pool windows.

    Exact and distance analyses are always exhaustive.  Thermodynamic analysis
    is exhaustive when ``thermodynamic_candidate_limit`` is ``None``; otherwise
    the output records the candidate-selection procedure and coverage.
    """

    if target_word not in pool.target_words:
        raise ValueError("target_word was not declared when the pool was built")
    if representative_site_limit < 1:
        raise ValueError("representative_site_limit must be >= 1")
    if thermodynamic_random_controls < 0:
        raise ValueError("thermodynamic_random_controls must be >= 0")

    target_segment = encode_text_to_strand(
        target_word,
        generator=generator,
        bit_length=bit_length,
    )
    probe = str(Seq(target_segment).reverse_complement())
    probe_len = len(target_segment)
    edit_distance = _make_edit_distance(target_segment)

    positive_payload = pool.strands[0].payload
    intended_starts = {
        position * pool.codeword_length_nt
        for position in find_exact_subsequence_positions(
            positive_payload,
            target_word,
        )
    }

    total_windows = 0
    intended_occurrences = 0
    intended_locus_overlap_windows = 0
    exact_off_target_occurrences = 0
    aligned_off_target_windows = 0
    unaligned_off_target_windows = 0
    hamming_histogram: Counter[int] = Counter()
    edit_histogram: Counter[int] = Counter()
    competitors: dict[str, dict[str, Any]] = {}

    possible_windows = sum(
        max(0, len(strand.dna) - probe_len + 1) for strand in pool.strands
    )
    with tqdm(
        total=possible_windows,
        desc=f"{target_word}: sequence scan",
        unit="window",
        disable=not show_progress,
    ) as scan_progress:
        for strand_index, strand in enumerate(pool.strands):
            for start in range(0, len(strand.dna) - probe_len + 1):
                total_windows += 1
                scan_progress.update()
                window = strand.dna[start : start + probe_len]
                is_intended = strand_index == 0 and start in intended_starts
                if is_intended:
                    intended_occurrences += 1
                    continue
                overlaps_intended_locus = strand_index == 0 and any(
                    start < intended_start + probe_len
                    and start + probe_len > intended_start
                    for intended_start in intended_starts
                )
                if overlaps_intended_locus:
                    intended_locus_overlap_windows += 1
                    continue

                aligned = start % pool.codeword_length_nt == 0
                if aligned:
                    aligned_off_target_windows += 1
                else:
                    unaligned_off_target_windows += 1

                hamming = hamming_distance(target_segment, window)
                edit = edit_distance(window)
                hamming_histogram[hamming] += 1
                edit_histogram[edit] += 1
                if hamming == 0:
                    exact_off_target_occurrences += 1

                existing = competitors.get(window)
                if existing is None:
                    competitors[window] = {
                        "strand_name": strand.name,
                        "start_nt": start,
                        "hamming_distance": hamming,
                        "edit_distance": edit,
                        "occurrence_count": 1,
                        "codeword_aligned": aligned,
                    }
                else:
                    existing["occurrence_count"] += 1

    expected_intended = len(intended_starts)
    if intended_occurrences != expected_intended:
        raise AssertionError(
            f"classified {intended_occurrences} intended sites; "
            f"expected {expected_intended}"
        )

    ranked_competitors = sorted(
        competitors.items(),
        key=lambda item: (
            item[1]["edit_distance"],
            item[1]["hamming_distance"],
            item[0],
        ),
    )
    representative_sites = [
        asdict(
            WindowSite(
                sequence=sequence,
                strand_name=details["strand_name"],
                start_nt=details["start_nt"],
                hamming_distance=details["hamming_distance"],
                edit_distance=details["edit_distance"],
                occurrence_count=details["occurrence_count"],
                codeword_aligned=details["codeword_aligned"],
            )
        )
        for sequence, details in ranked_competitors[:representative_site_limit]
    ]

    candidate_sequences, thermo_exhaustive, selection_method = (
        _select_thermodynamic_candidates(
            competitors,
            limit=thermodynamic_candidate_limit,
            random_control_count=thermodynamic_random_controls,
            seed=pool.seed,
        )
    )
    scorer = thermodynamic_scorer or _default_thermodynamic_scorer
    target_duplex = scorer(probe, target_segment, conditions)
    off_target_scores = []
    for sequence in tqdm(
        candidate_sequences,
        desc=f"{target_word}: thermodynamics",
        unit="unique window",
        disable=not show_progress,
    ):
        off_target_scores.append((sequence, scorer(probe, sequence, conditions)))
    off_target_tms = [score.tm_c for _, score in off_target_scores]
    scored_occurrences = sum(
        competitors[sequence]["occurrence_count"] for sequence in candidate_sequences
    )
    dg_occurrence_weights = [
        (score.dg_kcal_mol, competitors[sequence]["occurrence_count"])
        for sequence, score in off_target_scores
    ]
    strongest = (
        min(off_target_scores, key=lambda item: item[1].dg_kcal_mol)
        if off_target_scores
        else None
    )
    strongest_sequence = strongest[0] if strongest else None
    strongest_score = strongest[1] if strongest else None
    strongest_details = (
        competitors[strongest_sequence] if strongest_sequence is not None else None
    )

    target_preference_dg_margin = (
        strongest_score.dg_kcal_mol - target_duplex.dg_kcal_mol
        if strongest_score is not None
        else None
    )
    max_off_target_tm = max(off_target_tms) if off_target_tms else None
    tm_margin = (
        target_duplex.tm_c - max_off_target_tm
        if max_off_target_tm is not None
        else None
    )
    at_least_as_stable_count = sum(
        competitors[sequence]["occurrence_count"]
        for sequence, score in off_target_scores
        if score.dg_kcal_mol <= target_duplex.dg_kcal_mol
    )
    within_one_kcal_count = sum(
        competitors[sequence]["occurrence_count"]
        for sequence, score in off_target_scores
        if score.dg_kcal_mol <= target_duplex.dg_kcal_mol + 1.0
    )

    return {
        "target_word": target_word,
        "target_segment": target_segment,
        "probe": probe,
        "probe_length_nt": probe_len,
        "probe_gc_percent": round(
            100.0 * (probe.count("G") + probe.count("C")) / len(probe),
            4,
        ),
        "pool_seed": pool.seed,
        "pool_strands": len(pool.strands),
        "pool_total_nt": sum(len(strand.dna) for strand in pool.strands),
        "total_windows_scanned": total_windows,
        "intended_site_count": intended_occurrences,
        "intended_locus_overlap_windows_excluded": intended_locus_overlap_windows,
        "total_off_target_windows": (
            aligned_off_target_windows + unaligned_off_target_windows
        ),
        "exact_off_target_occurrences": exact_off_target_occurrences,
        "unique_off_target_windows": len(competitors),
        "aligned_off_target_windows": aligned_off_target_windows,
        "unaligned_off_target_windows": unaligned_off_target_windows,
        "minimum_off_target_hamming_distance": (
            min(hamming_histogram) if hamming_histogram else None
        ),
        "minimum_off_target_edit_distance": (
            min(edit_histogram) if edit_histogram else None
        ),
        "off_target_counts_by_hamming_distance": {
            str(distance): hamming_histogram.get(distance, 0)
            for distance in range(0, min(3, probe_len) + 1)
        },
        "off_target_counts_by_edit_distance": {
            str(distance): edit_histogram.get(distance, 0)
            for distance in range(0, min(3, probe_len) + 1)
        },
        "representative_nearest_sites": representative_sites,
        "thermodynamic_analysis": {
            "model": "primer3_heterodimer_nearest_neighbour",
            "interpretation": "hybridisation_probe_not_pcr",
            "unique_windows_scored": len(candidate_sequences),
            "unique_windows_available": len(competitors),
            "unique_sequence_coverage_fraction": (
                len(candidate_sequences) / len(competitors)
                if competitors
                else 1.0
            ),
            "window_occurrences_scored": scored_occurrences,
            "window_occurrences_available": (
                aligned_off_target_windows + unaligned_off_target_windows
            ),
            "window_occurrence_coverage_fraction": (
                scored_occurrences
                / (aligned_off_target_windows + unaligned_off_target_windows)
                if aligned_off_target_windows + unaligned_off_target_windows
                else 1.0
            ),
            "exhaustive": thermo_exhaustive,
            "selection_method": selection_method,
            "target_dg_kcal_mol": round(target_duplex.dg_kcal_mol, 4),
            "target_tm_c": round(target_duplex.tm_c, 4),
            "strongest_off_target_sequence": strongest_sequence,
            "strongest_off_target_dg_kcal_mol": (
                round(strongest_score.dg_kcal_mol, 4)
                if strongest_score is not None
                else None
            ),
            "strongest_off_target_tm_c": (
                round(strongest_score.tm_c, 4)
                if strongest_score is not None
                else None
            ),
            "strongest_off_target_hamming_distance": (
                strongest_details["hamming_distance"]
                if strongest_details is not None
                else None
            ),
            "strongest_off_target_edit_distance": (
                strongest_details["edit_distance"]
                if strongest_details is not None
                else None
            ),
            "strongest_off_target_strand": (
                strongest_details["strand_name"]
                if strongest_details is not None
                else None
            ),
            "strongest_off_target_start_nt": (
                strongest_details["start_nt"]
                if strongest_details is not None
                else None
            ),
            "strongest_off_target_occurrence_count": (
                strongest_details["occurrence_count"]
                if strongest_details is not None
                else None
            ),
            "strongest_off_target_codeword_aligned": (
                strongest_details["codeword_aligned"]
                if strongest_details is not None
                else None
            ),
            "target_preference_dg_margin_kcal_mol": (
                round(target_preference_dg_margin, 4)
                if target_preference_dg_margin is not None
                else None
            ),
            "target_minus_off_target_tm_margin_c": (
                round(tm_margin, 4) if tm_margin is not None else None
            ),
            "off_target_dg_percentiles_kcal_mol": {
                "p01": _weighted_percentile(dg_occurrence_weights, 0.01),
                "p05": _weighted_percentile(dg_occurrence_weights, 0.05),
                "p50": _weighted_percentile(dg_occurrence_weights, 0.50),
            },
            "off_target_sites_at_least_as_stable_as_target": (
                at_least_as_stable_count
            ),
            "off_target_fraction_at_least_as_stable_as_target": (
                at_least_as_stable_count / scored_occurrences
                if scored_occurrences
                else 0.0
            ),
            "off_target_sites_within_1_kcal_mol_of_target": (
                within_one_kcal_count
            ),
            "off_target_fraction_within_1_kcal_mol_of_target": (
                within_one_kcal_count / scored_occurrences
                if scored_occurrences
                else 0.0
            ),
        },
    }


def summarise_probe_analyses(
    analyses: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate the principal specificity outcomes across probes and seeds.
    """

    rows = list(analyses)
    if not rows:
        raise ValueError("analyses must not be empty")
    margins = [
        row["thermodynamic_analysis"][
            "target_preference_dg_margin_kcal_mol"
        ]
        for row in rows
        if row["thermodynamic_analysis"][
            "target_preference_dg_margin_kcal_mol"
        ]
        is not None
    ]
    return {
        "analysis_count": len(rows),
        "target_words": sorted({row["target_word"] for row in rows}),
        "pool_seeds": sorted({row["pool_seed"] for row in rows}),
        "total_windows_scanned": sum(row["total_windows_scanned"] for row in rows),
        "total_exact_off_target_occurrences": sum(
            row["exact_off_target_occurrences"] for row in rows
        ),
        "minimum_hamming_distance_observed": min(
            row["minimum_off_target_hamming_distance"] for row in rows
        ),
        "minimum_edit_distance_observed": min(
            row["minimum_off_target_edit_distance"] for row in rows
        ),
        "minimum_target_off_target_dg_margin_kcal_mol": (
            min(margins) if margins else None
        ),
        "all_thermodynamic_scans_exhaustive": all(
            row["thermodynamic_analysis"]["exhaustive"] for row in rows
        ),
        "all_targets_preferred_over_scored_off_targets": all(
            margin > 0 for margin in margins
        ),
    }
