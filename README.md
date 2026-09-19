# dna_encoding

Research repo for developing DNA context-independent and context-dependent encoding schemes.

## Structure

- `src/independent/`: context-independent schemes
- `src/dependent/`: context-dependent schemes

## Setup

```bash
python -m pip install -r requirements.txt
```

Run commands from the repository root so the `src` package is importable.

## Pool-wide probe specificity

The Edge-Cap pool experiment performs exhaustive exact, Hamming-distance, and
same-window-length edit-distance screening across target-positive and
target-absent encoded strands. It includes codeword-unaligned and
boundary-spanning windows and separates repeated intended sites from competing
binding loci.

```bash
python -m src.independent.edgecap.pool_specificity \
  --core-length 5 \
  --targets both road wood \
  --seeds 7 17 29 \
  --negative-count 25 \
  --thermodynamic-candidate-limit 0
```

Set `--core-length 7` for the supplementary 9-nt capped-codeword analysis.
`--thermodynamic-candidate-limit 0` requests exhaustive Primer3 heterodimer
scoring of all unique off-target windows. A positive limit screens all windows
by sequence distance but thermodynamically scores only the nearest candidates
plus fixed-seed random controls; the JSON output records the resulting
coverage. The runner writes both machine-readable JSON and a sibling
`.manuscript.md` file containing replacement Methods, Results, and Table II
wording derived directly from those results. This is an in-silico
hybridisation-probe model, not a PCR amplification model.

Run the correctness checks with:

```bash
python -m unittest discover -s tests -v
```