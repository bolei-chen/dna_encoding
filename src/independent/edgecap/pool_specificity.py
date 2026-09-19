"""
Reproducible pool-wide hybridisation-probe specificity experiment for Edge-Cap.

Example:
    python -m src.independent.edgecap.pool_specificity \
        --core-length 5 \
        --negative-count 100 --seeds 7 17 29 \
        --thermodynamic-candidate-limit 5000

Use ``--thermodynamic-candidate-limit 0`` to score every unique off-target
window thermodynamically.  Exact, Hamming, and edit-distance scans are always
exhaustive regardless of this option.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from src.independent.edgecap.simulation import build_generator
from src.utils.eval import PCRConditions, default_payload_text
from src.utils.pool_specificity import (
    analyze_probe_against_pool,
    build_encoded_pool,
    summarise_probe_analyses,
)


DEFAULT_TARGET_WORDS = ("both", "road", "wood")
DEFAULT_SEEDS = (7, 17, 29)


def render_manuscript_update(result: dict[str, Any]) -> str:
    """
    Render manuscript-ready replacement text from a completed JSON result.
    """

    pool = result["pool"]
    analyses = result["analyses"]
    summary = result["summary"]
    conditions = result["thermodynamic_conditions"]
    unique_pool_nt = sum(
        next(row["pool_total_nt"] for row in analyses if row["pool_seed"] == seed)
        for seed in pool["seeds"]
    )
    scored_windows = sum(
        row["thermodynamic_analysis"]["window_occurrences_scored"]
        for row in analyses
    )
    stable_sites = sum(
        row["thermodynamic_analysis"][
            "off_target_sites_at_least_as_stable_as_target"
        ]
        for row in analyses
    )

    rows = []
    for target in pool["target_words"]:
        target_rows = [row for row in analyses if row["target_word"] == target]
        thermo_rows = [row["thermodynamic_analysis"] for row in target_rows]
        stable = sum(
            row["off_target_sites_at_least_as_stable_as_target"]
            for row in thermo_rows
        )
        represented = sum(row["window_occurrences_scored"] for row in thermo_rows)
        rows.append(
            "| {target} | {length} | {gc:.1f} | {exact} | {hamming} | "
            "{edit} | {target_dg:.2f} | {off_dg:.2f} | {margin:.2f} | "
            "{stable}/{represented} | {exhaustive} |".format(
                target=target,
                length=target_rows[0]["probe_length_nt"],
                gc=target_rows[0]["probe_gc_percent"],
                exact=sum(row["exact_off_target_occurrences"] for row in target_rows),
                hamming=min(
                    row["minimum_off_target_hamming_distance"] for row in target_rows
                ),
                edit=min(
                    row["minimum_off_target_edit_distance"] for row in target_rows
                ),
                target_dg=thermo_rows[0]["target_dg_kcal_mol"],
                off_dg=min(
                    row["strongest_off_target_dg_kcal_mol"]
                    for row in thermo_rows
                ),
                margin=min(
                    row["target_preference_dg_margin_kcal_mol"]
                    for row in thermo_rows
                ),
                stable=stable,
                represented=represented,
                exhaustive="yes" if all(row["exhaustive"] for row in thermo_rows) else "no",
            )
        )

    thermo_scope = (
        "all unique competing windows"
        if summary["all_thermodynamic_scans_exhaustive"]
        else "the explicitly reported distance-ranked and fixed-seed control subset"
    )
    specificity_sentence = (
        "No scored competing site was at least as thermodynamically stable as "
        "its intended target."
        if stable_sites == 0
        else (
            f"{stable_sites} scored competing-site occurrences were at least as "
            "thermodynamically stable as their intended target; these cases must "
            "not be described as specific without further probe redesign or "
            "physical validation."
        )
    )

    return f"""# Manuscript replacement: pool-wide probe specificity

## Methods: Pool-wide competing-site analysis

We evaluated reverse-complement probes for the known payload substrings
{", ".join(f"`{word}`" for word in pool["target_words"])} across
{len(pool["seeds"])} independently generated encoded pools. Each pool contained
one target-positive strand and {pool["target_absent_strands_per_seed"]}
same-length target-absent strands generated from printable ASCII with fixed
seed. Across the independent pools, the encoded sequences comprised
{unique_pool_nt:,} nt. Every same-length window was screened, including
codeword-unaligned windows and windows spanning codeword boundaries. Exact
intended occurrences were classified separately, and shifted windows
overlapping an intended physical locus were not counted as independent
off-target sites.

For every competing locus we computed Hamming distance and same-window-length
Levenshtein distance to the encoded target. Identical windows were deduplicated
for thermodynamic computation while retaining their occurrence multiplicity.
Probe-window heterodimer free energy and melting temperature were calculated
with Primer3 under {conditions["monovalent_cation_mM"]} mM monovalent cation,
{conditions["divalent_cation_mM"]} mM divalent cation,
{conditions["dntp_mM"]} mM dNTP, {conditions["probe_concentration_nM"]} nM
probe, and {conditions["temperature_c"]} °C. Thermodynamic scoring covered
{thermo_scope}. This analysis models in-silico probe hybridisation and does not
model PCR amplification.

## Results: Competing binding sites

Across {summary["analysis_count"]} probe-pool analyses, we screened
{summary["total_windows_scanned"]:,} same-length windows and
thermodynamically represented {scored_windows:,} competing-site occurrences.
The scan found {summary["total_exact_off_target_occurrences"]} exact unintended
matches. The minimum observed competing-site distances were
{summary["minimum_hamming_distance_observed"]} substitutions and
{summary["minimum_edit_distance_observed"]} edits. The minimum target-preference
free-energy margin was
{summary["minimum_target_off_target_dg_margin_kcal_mol"]:.2f} kcal/mol, where a
positive value indicates that the intended duplex was more favourable.
{specificity_sentence}

| Target | Probe nt | GC % | Exact off-targets | Min Hamming | Min edit | Target ΔG | Strongest off-target ΔG | Min preference margin | Sites at least as stable | Thermo exhaustive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
{chr(10).join(rows)}

These results are in-silico indicators of candidate-probe discrimination.
They do not establish physical specificity, which additionally depends on
strand accessibility, concentration, surface or solution assay design, and
experimental conditions.
"""


def run_experiment(
    *,
    target_words: Sequence[str] = DEFAULT_TARGET_WORDS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    core_length: int = 5,
    negative_count: int = 100,
    thermodynamic_candidate_limit: int | None = 5000,
    thermodynamic_random_controls: int = 500,
) -> dict[str, Any]:
    """
    Run the Edge-Cap pool-wide experiment and return JSON-ready results.
    """

    if core_length < 1:
        raise ValueError("core_length must be >= 1")
    generator = build_generator(length=core_length)
    payload = default_payload_text()
    conditions = PCRConditions(
        mv_conc_mM=50.0,
        dv_conc_mM=1.5,
        dntp_conc_mM=0.2,
        primer_conc_nM=250.0,
        annealing_temp_c=68.0,
    )

    analyses: list[dict[str, Any]] = []
    for seed in seeds:
        pool = build_encoded_pool(
            generator=generator,
            positive_payload=payload,
            target_words=target_words,
            negative_count=negative_count,
            seed=seed,
            bit_length=8,
        )
        for target_word in target_words:
            print(
                f"Scanning target={target_word!r}, seed={seed}, "
                f"pool_strands={len(pool.strands)}..."
            )
            analyses.append(
                analyze_probe_against_pool(
                    pool=pool,
                    generator=generator,
                    target_word=target_word,
                    conditions=conditions,
                    bit_length=8,
                    thermodynamic_candidate_limit=thermodynamic_candidate_limit,
                    thermodynamic_random_controls=thermodynamic_random_controls,
                    show_progress=True,
                )
            )

    return {
        "experiment": "edgecap_pool_wide_probe_specificity",
        "assay_model": "in_silico_hybridisation_probe",
        "generator": {
            "name": generator.name,
            "core_length_nt": core_length,
            "capped_codeword_length_nt": generator.codeword_length_nt,
            "max_homopolymer_run": 3,
            "gc_lower": 0.4,
            "gc_upper": 0.6,
        },
        "pool": {
            "positive_strands_per_seed": 1,
            "target_absent_strands_per_seed": negative_count,
            "payload_characters_per_strand": len(payload),
            "seeds": list(seeds),
            "target_words": list(target_words),
        },
        "thermodynamic_conditions": {
            "monovalent_cation_mM": conditions.mv_conc_mM,
            "divalent_cation_mM": conditions.dv_conc_mM,
            "dntp_mM": conditions.dntp_conc_mM,
            "probe_concentration_nM": conditions.primer_conc_nM,
            "temperature_c": conditions.annealing_temp_c,
        },
        "method_notes": [
            "All same-length windows are screened, including codeword-unaligned "
            "and boundary-spanning windows.",
            "Repeated intended occurrences are excluded from off-target counts.",
            "Exact, Hamming, and same-window-length Levenshtein scans are exhaustive.",
            "Thermodynamic coverage and candidate selection are reported per analysis.",
            "The experiment models probe hybridisation, not PCR amplification.",
        ],
        "summary": summarise_probe_analyses(analyses),
        "analyses": analyses,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGET_WORDS),
        help="Target substrings present in the positive payload",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Random seeds for independent target-absent pools",
    )
    parser.add_argument(
        "--core-length",
        type=int,
        default=5,
        help="Edge-Cap core codeword length before adding the two caps",
    )
    parser.add_argument(
        "--negative-count",
        type=int,
        default=100,
        help="Target-absent strands per seed",
    )
    parser.add_argument(
        "--thermodynamic-candidate-limit",
        type=int,
        default=5000,
        help="Unique windows scored per probe; 0 requests exhaustive scoring",
    )
    parser.add_argument(
        "--thermodynamic-random-controls",
        type=int,
        default=500,
        help="Seeded random controls included when thermodynamic scoring is limited",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/edgecap_pool_specificity.json"),
        help="JSON output path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.negative_count < 1:
        raise ValueError("--negative-count must be >= 1")
    if args.core_length < 1:
        raise ValueError("--core-length must be >= 1")
    if args.thermodynamic_candidate_limit < 0:
        raise ValueError("--thermodynamic-candidate-limit must be >= 0")
    thermo_limit = (
        None
        if args.thermodynamic_candidate_limit == 0
        else args.thermodynamic_candidate_limit
    )
    result = run_experiment(
        target_words=args.targets,
        seeds=args.seeds,
        core_length=args.core_length,
        negative_count=args.negative_count,
        thermodynamic_candidate_limit=thermo_limit,
        thermodynamic_random_controls=args.thermodynamic_random_controls,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    manuscript_path = args.output.with_suffix(".manuscript.md")
    manuscript_path.write_text(
        render_manuscript_update(result),
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {manuscript_path}")


if __name__ == "__main__":
    main()
