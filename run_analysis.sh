#!/usr/bin/env bash
# run_analysis.sh
# Full analysis pipeline for the EU Values Survey LLM project.
#
# Usage:
#   bash run_analysis.sh [OPTIONS]
#
# Options:
#   --human-only     Run only the human responses analysis
#   --llm-only       Run only the LLM responses analysis
#   --no-pca         Skip PCA plots in the human analysis
#   --no-consensus   Skip consensus analysis in the human analysis
#   --sanity-check   Enable additional consistency checks before/after each step
#   --no-sanity-check Disable additional consistency checks
#   --help           Show this message and exit
#
# Outputs are written to:
#   src/data_preprocessing/output/
#
# Requirements:
#   A Python >=3.10 virtual environment with all dependencies installed.
#   By default the script looks for .venv/ at the repository root; override
#   with the VENV environment variable:
#       VENV=/path/to/venv bash run_analysis.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-${SCRIPT_DIR}/../.venv}"
PYTHON="${VENV}/bin/python"
OUTPUT_DIR="${SCRIPT_DIR}/src/data_preprocessing/output"
HUMAN_SURVEY_CSV="${HUMAN_SURVEY_CSV:-${SCRIPT_DIR}/Surveys_responses/Human_responses/moral_values_survey_with_languages.csv}"
TOPICS_CSV="${TOPICS_CSV:-${SCRIPT_DIR}/Surveys/Survey_metadata/questions_topic_matching.csv}"
LLM_DIR="${LLM_DIR:-${SCRIPT_DIR}/Surveys_responses}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "Required file not found: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "Required directory not found: $path"
}

require_nonempty_file() {
  local path="$1"
  [[ -s "$path" ]] || die "Expected non-empty output file was not generated: $path"
}

# ── Parse arguments ───────────────────────────────────────────────────────────
RUN_HUMAN=1
RUN_LLM=1
SANITY_CHECK=0
HUMAN_EXTRA_FLAGS="--pca --by-country --consensus"

for arg in "$@"; do
  case "$arg" in
    --human-only)  RUN_LLM=0 ;;
    --llm-only)    RUN_HUMAN=0 ;;
    --no-pca)      HUMAN_EXTRA_FLAGS="${HUMAN_EXTRA_FLAGS/--pca/}" ;;
    --no-consensus) HUMAN_EXTRA_FLAGS="${HUMAN_EXTRA_FLAGS/--by-country --consensus/}" ;;
    --sanity-check) SANITY_CHECK=1 ;;
    --no-sanity-check) SANITY_CHECK=0 ;;
    --help)
      sed -n '2,18p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *)
      echo "Unknown option: $arg  (use --help for usage)" >&2
      exit 1 ;;
  esac
done

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python interpreter not found at ${PYTHON}" >&2
  echo "       Create the virtual environment first:" >&2
  echo "       python -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── Optional sanity checks (pre-run) ─────────────────────────────────────────
if [[ $SANITY_CHECK -eq 1 ]]; then
  echo "[sanity] Running pre-run checks..."

  require_file "$HUMAN_SURVEY_CSV"
  require_file "$TOPICS_CSV"
  require_dir "$LLM_DIR"

  shopt -s nullglob
  llm_files=("${LLM_DIR}"/llm_survey_*.csv)
  shopt -u nullglob
  if [[ ${#llm_files[@]} -eq 0 && $RUN_LLM -eq 1 ]]; then
    die "No llm_survey_*.csv files found under ${LLM_DIR}"
  fi

  "$PYTHON" - <<'PY'
mods = ["pandas", "numpy", "matplotlib", "sklearn", "adjustText", "scienceplots", "scipy"]
missing = []
for mod in mods:
    try:
        __import__(mod)
    except Exception:
        missing.append(mod)
if missing:
    raise SystemExit(f"Missing Python dependencies required by pipeline: {missing}")
print("[sanity] Python dependencies: OK")
PY
fi

# ── Human responses analysis ──────────────────────────────────────────────────
if [[ $RUN_HUMAN -eq 1 ]]; then
  echo "================================================================"
  echo "  HUMAN RESPONSES ANALYSIS"
  echo "================================================================"
  # shellcheck disable=SC2086
  "$PYTHON" "${SCRIPT_DIR}/src/data_preprocessing/human_responses_analysis.py" \
    --survey "$HUMAN_SURVEY_CSV" \
    --topics "$TOPICS_CSV" \
    --output "$OUTPUT_DIR" \
    $HUMAN_EXTRA_FLAGS

  if [[ $SANITY_CHECK -eq 1 ]]; then
    echo "[sanity] Validating human-analysis outputs..."
    require_nonempty_file "${OUTPUT_DIR}/topic_averages.csv"
    require_nonempty_file "${OUTPUT_DIR}/normalized_survey_with_topics.csv"
    require_nonempty_file "${OUTPUT_DIR}/radar_chart_topics.png"
    require_nonempty_file "${OUTPUT_DIR}/radar_chart_topics_by_language_family.png"

    if [[ " $HUMAN_EXTRA_FLAGS " == *" --by-country "* ]]; then
      require_nonempty_file "${OUTPUT_DIR}/radar_chart_by_country.png"
    fi
    if [[ " $HUMAN_EXTRA_FLAGS " == *" --pca "* ]]; then
      require_nonempty_file "${OUTPUT_DIR}/pca_countries_by_language_family_1col.png"
      require_nonempty_file "${OUTPUT_DIR}/pca_countries_by_language_family_2col.png"
      require_nonempty_file "${OUTPUT_DIR}/pca_components_interpretation.csv"
    fi
    if [[ " $HUMAN_EXTRA_FLAGS " == *" --consensus "* ]]; then
      require_nonempty_file "${OUTPUT_DIR}/consensus_analysis.png"
    fi

    "$PYTHON" - <<PY
from pathlib import Path
import numpy as np
import pandas as pd

out = Path(r"${OUTPUT_DIR}")

topic = pd.read_csv(out / "topic_averages.csv")
required_cols = {"category", "average_score", "response_count"}
missing = required_cols - set(topic.columns)
if missing:
    raise SystemExit(f"topic_averages.csv missing columns: {sorted(missing)}")
if topic.empty:
    raise SystemExit("topic_averages.csv is empty")
if topic["average_score"].isna().any():
    raise SystemExit("topic_averages.csv contains NaN average_score values")
if ((topic["average_score"] < 0) | (topic["average_score"] > 1)).any():
    raise SystemExit("topic_averages.csv has average_score outside [0,1]")
if (topic["response_count"] <= 0).any():
    raise SystemExit("topic_averages.csv has non-positive response_count values")

norm = pd.read_csv(out / "normalized_survey_with_topics.csv", low_memory=False)
v_cols = [c for c in norm.columns if c.startswith("v") and len(c) > 1 and c[1].isdigit()]
if not v_cols:
    raise SystemExit("normalized_survey_with_topics.csv contains no v* survey columns")
vals = norm[v_cols].to_numpy(dtype=float)
finite = vals[np.isfinite(vals)]
if finite.size == 0:
    raise SystemExit("normalized survey data has no finite numeric values")
if finite.min() < -1 or finite.max() > 1:
    raise SystemExit(
    f"normalized survey values outside [-1,1]: min={finite.min():.4f}, max={finite.max():.4f}"
    )
share_minus_one = float((vals == -1).sum()) / float(np.isfinite(vals).sum())
if share_minus_one > 0.60:
  raise SystemExit(
    f"unexpectedly high share of -1 sentinel values in normalized survey data: {share_minus_one:.2%}"
  )
print("[sanity] Human outputs: OK")
PY
  fi
  echo
fi

# ── LLM responses analysis ────────────────────────────────────────────────────
if [[ $RUN_LLM -eq 1 ]]; then
  echo "================================================================"
  echo "  LLM RESPONSES ANALYSIS"
  echo "================================================================"
  "$PYTHON" "${SCRIPT_DIR}/src/llm_responses_analysis/llm_responses_analysis.py" \
    --llm-dir "$LLM_DIR" \
    --human "$HUMAN_SURVEY_CSV" \
    --topics "$TOPICS_CSV" \
    --output "$OUTPUT_DIR"

  if [[ $SANITY_CHECK -eq 1 ]]; then
    echo "[sanity] Validating llm-analysis outputs..."
    require_nonempty_file "${OUTPUT_DIR}/llm_pca_projection_1col.png"
    require_nonempty_file "${OUTPUT_DIR}/llm_pca_projection_2col.png"
    require_nonempty_file "${OUTPUT_DIR}/llm_pca_projection_by_model_1col.png"
    require_nonempty_file "${OUTPUT_DIR}/llm_pca_projection_by_model_2col.png"
    require_nonempty_file "${OUTPUT_DIR}/llm_pca_weighted_family_overlay_1col.png"
    require_nonempty_file "${OUTPUT_DIR}/llm_pca_weighted_family_overlay_2col.png"
    require_nonempty_file "${OUTPUT_DIR}/llm_pca_english_vs_non_english_mean_1col.png"
    require_nonempty_file "${OUTPUT_DIR}/llm_pca_english_vs_non_english_mean_2col.png"
    require_nonempty_file "${OUTPUT_DIR}/llm_pca_human_vs_llm_english_multilingual_1col.png"
    require_nonempty_file "${OUTPUT_DIR}/llm_pca_human_vs_llm_english_multilingual_2col.png"
    require_nonempty_file "${OUTPUT_DIR}/llm_radar_vs_human.png"
    require_nonempty_file "${OUTPUT_DIR}/ks_alignment_by_language_family.png"

    "$PYTHON" - <<PY
from pathlib import Path
import glob
import re
import sys
import numpy as np
import pandas as pd

llm_dir = Path(r"${LLM_DIR}")
out_dir = Path(r"${OUTPUT_DIR}")
files = sorted(glob.glob(str(llm_dir / "llm_survey_*.csv")))
if not files:
    raise SystemExit("No llm_survey_*.csv files found during post-run validation")

sample = pd.read_csv(files[0], nrows=10)
required = {"variable", "model_response"}
missing = required - set(sample.columns)
if missing:
    raise SystemExit(f"Sample LLM response file missing columns: {sorted(missing)}")

# ── Cross-model similarity sanity test ──────────────────────────────────────
sys.path.insert(0, str(Path(r"${SCRIPT_DIR}") / "src" / "llm_responses_analysis"))
from llm_responses_analysis import _parse_numeric  # noqa: E402

rx = re.compile(r"llm_survey_(.+)_responses_([a-z]+)\.csv")
rows = []
for fp in files:
    name = Path(fp).name
    m = rx.match(name)
    if not m:
        continue
    model = m.group(1)
    country = m.group(2)
    df = pd.read_csv(fp, usecols=lambda c: c in {"variable", "model_response"})
    if "variable" not in df.columns or "model_response" not in df.columns:
        continue
    df["numeric_response"] = df["model_response"].apply(_parse_numeric)
    df = df[df["numeric_response"].notna()].copy()
    if df.empty:
        continue
    g = df.groupby("variable", as_index=False)["numeric_response"].mean()
    g["model"] = model
    g["country"] = country
    rows.append(g)

if not rows:
    raise SystemExit("Could not parse numeric LLM responses for similarity sanity test")

parsed = pd.concat(rows, ignore_index=True)

# model-level vector: mean per variable across countries
model_vec = (parsed
             .groupby(["model", "variable"])["numeric_response"]
             .mean()
             .unstack("variable"))
models = sorted(model_vec.index.tolist())

sim_rows = []
for i, ma in enumerate(models):
    for mb in models[i + 1:]:
        va = model_vec.loc[ma]
        vb = model_vec.loc[mb]
        common = va.notna() & vb.notna()
        n_common = int(common.sum())
        if n_common < 30:
            continue
        a = va[common].to_numpy(dtype=float)
        b = vb[common].to_numpy(dtype=float)
        # Handle constant vectors safely
        if np.std(a) == 0 or np.std(b) == 0:
            corr = 1.0 if np.allclose(a, b, atol=1e-12, rtol=0.0) else np.nan
        else:
            corr = float(np.corrcoef(a, b)[0, 1])
        mad = float(np.mean(np.abs(a - b)))
        identical = bool(np.allclose(a, b, atol=1e-12, rtol=0.0))
        sim_rows.append({
            "model_a": ma,
            "model_b": mb,
            "pearson_r": corr,
            "mean_abs_diff": mad,
            "n_common_variables": n_common,
            "identical": identical,
        })

if not sim_rows:
    print("[sanity] LLM similarity: skipped (insufficient comparable model pairs)")
else:
    sim_df = pd.DataFrame(sim_rows).sort_values(
        ["identical", "pearson_r", "mean_abs_diff"],
        ascending=[False, False, True],
    )
    sim_csv = out_dir / "llm_model_similarity_sanity.csv"
    sim_df.to_csv(sim_csv, index=False)

    identical_df = sim_df[sim_df["identical"]]
    suspicious_df = sim_df[(~sim_df["identical"]) & (sim_df["pearson_r"] >= 0.999)]

    if not identical_df.empty:
        print(f"[sanity][ALERT] Identical model pairs detected: {len(identical_df)}")
        for _, r in identical_df.head(10).iterrows():
            print(
                f"  [IDENTICAL] {r['model_a']} == {r['model_b']}  "
                f"(n={int(r['n_common_variables'])}, MAD={r['mean_abs_diff']:.6f})"
            )

    if not suspicious_df.empty:
        print(f"[sanity][WARN] Highly similar model pairs (r>=0.999): {len(suspicious_df)}")
        for _, r in suspicious_df.head(10).iterrows():
            print(
                f"  [SIMILAR] {r['model_a']} ~ {r['model_b']}  "
                f"(r={r['pearson_r']:.6f}, n={int(r['n_common_variables'])}, "
                f"MAD={r['mean_abs_diff']:.6f})"
            )

    if identical_df.empty and suspicious_df.empty:
        print("[sanity] LLM similarity: OK (no identical or ultra-high-correlation pairs)")

    print(f"[sanity] LLM similarity report saved: {sim_csv}")

print(f"[sanity] LLM outputs: OK ({len(files)} input files detected)")
PY
  fi
  echo
fi

echo "All done. Figures saved to: ${OUTPUT_DIR}"
