"""
DK/NA and out-of-range response rate analysis — per variable × model.

Produces two figures:
  1. Heatmap: DK/NA rate (score 8 or 9) per variable × model
  2. Heatmap: out-of-range rate (parsed value outside the human-observed range
     for that variable) per variable × model

Only variables that are part of PCA_SELECTED_VARS are shown (sorted by overall
DK/NA rate descending so the most problematic variables appear at the top).
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]  # EUValues/

sys.path.insert(0, str(_ROOT / "src"))
from survey_variables import PCA_SELECTED_VARS  # noqa: E402
sys.path.insert(0, str(_HERE))
from llm_responses_analysis import _deduplicate_variable_rows, _parse_numeric  # noqa: E402
from scale_metadata import (  # noqa: E402
    infer_variable_scale_metadata,
    is_value_dkna,
    load_canonical_scale_metadata,
    keep_only_substantive_range,
)

# Human responses always come from the same location (not from scale_control)
HUMAN_LLM_DIR = Path(os.getenv("HUMAN_LLM_DIR", str(_ROOT / "Surveys_responses" / "Human_responses")))
DEFAULT_LLM_DIR    = Path(os.getenv("LLM_DIR", str(_ROOT / "Surveys_responses")))
DEFAULT_HUMAN_CSV  = HUMAN_LLM_DIR / "moral_values_survey_with_languages_v2.csv"
DEFAULT_SCALE_CATALOG = _ROOT / "Surveys" / "Survey_metadata" / "question_scale_catalog.csv"
DEFAULT_SCALES_DIR = _ROOT / "Surveys_parsed"
DEFAULT_OUTPUT_DIR = _HERE.parent / "data_preprocessing" / "output"

def _load_norm_params(human_csv: Path, scale_metadata: dict | None = None) -> dict:
    """Return baseline substantive ranges per variable.

    When scale metadata provides an inferred substantive range, prefer that
    range over the human observed support so valid but unseen response codes are
    not mislabeled as out-of-range.
    """
    df = pd.read_csv(human_csv, low_memory=False)
    v_cols = [c for c in df.columns if c.startswith("v") and len(c) > 1
              and c[1].isdigit()]
    params = {}
    for col in v_cols:
        meta = (scale_metadata or {}).get(col)
        s = keep_only_substantive_range(df[col], meta)
        if meta is not None and meta.get("min") is not None and meta.get("max") is not None:
            lo = float(meta["min"])
            hi = float(meta["max"])
            if hi != lo:
                params[col] = (lo, hi)
                continue
        valid = s.dropna()
        if len(valid) > 0 and valid.max() != valid.min():
            params[col] = (float(valid.min()), float(valid.max()))
    return params


def load_responses(llm_dir: Path) -> pd.DataFrame:
    files = sorted(llm_dir.glob("llm_survey_*.csv"))
    chunks = []
    for f in files:
        mo = re.match(r"llm_survey_(.+)_responses_([a-z]+)\.csv", f.name)
        if not mo:
            continue
        df = pd.read_csv(f)
        keep = [c for c in ["model_response", "variable", "response_scale"] if c in df.columns]
        df = df[keep]
        df["_model"] = mo.group(1)
        parsed = df["model_response"].apply(lambda r: _parse_numeric(r, return_strategy=True))
        df["num"] = parsed.apply(lambda t: float(t[0]) if pd.notna(t[0]) else np.nan)
        df["numeric_response"] = df["num"]
        df["_parse_strategy"] = parsed.apply(lambda t: t[1])

        at_pat = re.compile(r"@@(\d{2,})@@")
        def _was_collapsed(resp):
            m2 = at_pat.search(str(resp))
            if m2:
                d = m2.group(1)
                return len(set(d)) == 1
            return False

        df["_collapsed"] = df["model_response"].apply(_was_collapsed)
        df = _deduplicate_variable_rows(df, verbose=False)
        chunks.append(df)
    return pd.concat(chunks, ignore_index=True)


def compute_rates(df: pd.DataFrame, norm_params: dict,
                  pca_vars: list,
                  scale_metadata: dict) -> tuple:
    """
    Returns two DataFrames (rows=variable, cols=model):
    - dkna_rates  : fraction of parsed responses classified as DK/NA
      - oor_rates   : fraction of parsed responses outside the human range
    Only variables in pca_vars that appear in the data are included.
    """
    df_parsed = df[df["num"].notna()].copy()
    df_parsed = df_parsed[df_parsed["variable"].isin(pca_vars)]

    dkna_rows, oor_rows = [], []
    models = sorted(df_parsed["_model"].unique())

    for var in pca_vars:
        sub = df_parsed[df_parsed["variable"] == var]
        if len(sub) == 0:
            continue
        dkna_row = {"variable": var}
        oor_row  = {"variable": var}
        min_h, max_h = norm_params.get(var, (1.0, 10.0))
        meta = scale_metadata.get(var)
        for model in models:
            msub = sub[sub["_model"] == model]
            n = len(msub)
            if n == 0:
                dkna_row[model] = np.nan
                oor_row[model]  = np.nan
            else:
                dkna_mask = msub["num"].apply(lambda x, m=meta: is_value_dkna(x, m))
                dkna_row[model] = dkna_mask.sum() / n

                # out-of-range: parsed, not DK/NA, and outside the baseline
                # substantive range used for human/LLM alignment.
                in_valid = msub.loc[~dkna_mask, "num"]
                if len(in_valid) == 0:
                    oor_row[model] = np.nan
                else:
                    oor_row[model] = ((in_valid < min_h) | (in_valid > max_h)).sum() / n
        dkna_rows.append(dkna_row)
        oor_rows.append(oor_row)

    dkna_df = pd.DataFrame(dkna_rows).set_index("variable")
    oor_df  = pd.DataFrame(oor_rows).set_index("variable")
    return dkna_df, oor_df


def _sort_vars(df: pd.DataFrame) -> list:
    """Sort variables by descending overall mean rate."""
    return df.mean(axis=1).sort_values(ascending=False).index.tolist()


def _plot_heatmap(data: pd.DataFrame, title: str, label: str,
                  output_path: Path, cmap: str = "Reds",
                  vmax: float = 1.0) -> None:
    """
    Heatmap: rows = variables (sorted), cols = models.
    Variables with overall mean rate < 0.01 are hidden to keep the plot readable.
    """
    # Filter: keep only variables where at least one model has rate > threshold
    threshold = 0.01
    keep = data[data.max(axis=1) > threshold]
    if len(keep) == 0:
        print(f"  [SKIP] {title}: no variable exceeds threshold {threshold:.0%}")
        return

    sorted_vars = _sort_vars(keep)
    plot_data   = keep.loc[sorted_vars]

    n_vars, n_models = plot_data.shape
    # Adaptive figure height: ~0.26 inch per variable, min 6, max 40
    fig_h = max(6.0, min(40.0, n_vars * 0.26 + 2.0))

    with plt.style.context(["science", "no-latex"]):
        mpl.rcParams.update({"font.size": 9})
        fig, ax = plt.subplots(figsize=(max(6.0, n_models * 1.1 + 1.5), fig_h))

        im = ax.imshow(plot_data.values, aspect="auto", cmap=cmap,
                       vmin=0.0, vmax=vmax, interpolation="nearest")

        # Axes labels
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(plot_data.columns, rotation=40, ha="right",
                           fontsize=8.5)
        ax.set_yticks(range(n_vars))
        ax.set_yticklabels(sorted_vars, fontsize=7.0)

        # Annotate cells with percentage text where rate > 5%
        for i in range(n_vars):
            for j in range(n_models):
                val = plot_data.iloc[i, j]
                if np.isnan(val) or val < 0.05:
                    continue
                col = "white" if val > 0.55 else "#222222"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=6.0, color=col)

        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label(label, fontsize=9)
        cbar.ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Model", fontsize=9)
        ax.set_ylabel("Variable (PCA-selected, sorted by overall rate)", fontsize=9)

        plt.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"  Saved: {output_path}  ({n_vars} variables shown)")


def _plot_summary_bar(dkna_df: pd.DataFrame, oor_df: pd.DataFrame,
                      output_path: Path) -> None:
    """
    Grouped bar chart: overall DK/NA rate and out-of-range rate per model.
    """
    models = dkna_df.columns.tolist()
    dkna_mean = dkna_df.mean(axis=0)
    oor_mean  = oor_df.mean(axis=0)

    x = np.arange(len(models))
    w = 0.38

    with plt.style.context(["science", "no-latex"]):
        mpl.rcParams.update({"font.size": 9.5})
        fig, ax = plt.subplots(figsize=(10, 4.5))

        bars_dkna = ax.bar(x - w/2, dkna_mean.values, width=w,
                           color="#EE6677", label="DK/NA (8 or 9)", alpha=0.88,
                           edgecolor="white", linewidth=0.5)
        bars_oor  = ax.bar(x + w/2, oor_mean.values, width=w,
                           color="#4477AA", label="Out-of-range (excl. DK/NA)",
                           alpha=0.88, edgecolor="white", linewidth=0.5)

        for bar in list(bars_dkna) + list(bars_oor):
            h = bar.get_height()
            if h > 0.005:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                        f"{h:.1%}", ha="center", va="bottom",
                        fontsize=7.5, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8.5)
        ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.set_ylabel("Fraction of parsed responses excluded", fontsize=9.5)
        ax.set_title("Response exclusion rates per model — PCA-selected variables",
                     fontsize=10.5, fontweight="bold")
        ax.legend(fontsize=9, frameon=True, fancybox=False,
                  edgecolor="#CCCCCC", loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, min(1.0, max(dkna_mean.max(), oor_mean.max()) * 1.25 + 0.02))
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)

        plt.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"  Saved: {output_path}")


def main():
    out_dir = DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading LLM responses from {DEFAULT_LLM_DIR}…")
    df = load_responses(DEFAULT_LLM_DIR)
    scale_source = DEFAULT_SCALE_CATALOG if DEFAULT_SCALE_CATALOG.exists() else DEFAULT_SCALES_DIR
    print(f"Loading canonical scale metadata from: {scale_source}")
    scale_metadata = load_canonical_scale_metadata(scale_source)
    inferred_scale_metadata = infer_variable_scale_metadata(df)
    for var, meta in inferred_scale_metadata.items():
        scale_metadata.setdefault(var, meta)

    print("Loading human norm params…")
    norm_params = _load_norm_params(DEFAULT_HUMAN_CSV, scale_metadata=scale_metadata)

    print(f"  {len(df):,} rows loaded, "
          f"{df['_model'].nunique()} models, "
          f"{df['variable'].nunique()} variables")

    pca_vars_present = [v for v in PCA_SELECTED_VARS
                        if v in df["variable"].values]
    print(f"  {len(pca_vars_present)} / {len(PCA_SELECTED_VARS)} "
          f"PCA variables present in LLM data")

    print("Computing rates…")
    dkna_df, oor_df = compute_rates(df, norm_params, pca_vars_present, scale_metadata)

    # ── Summary bar chart ────────────────────────────────────────────────────
    print("Generating summary bar chart…")
    _plot_summary_bar(dkna_df, oor_df,
                      out_dir / "exclusion_rates_by_model.png")

    # ── DK/NA heatmap ────────────────────────────────────────────────────────
    print("Generating DK/NA heatmap…")
    _plot_heatmap(dkna_df,
                  title="DK/NA rate (variable-aware codes/range) per variable × model",
                  label="DK/NA rate",
                  output_path=out_dir / "dkna_rate_heatmap.png",
                  cmap="Reds", vmax=1.0)

    # ── Out-of-range heatmap ─────────────────────────────────────────────────
    print("Generating out-of-range heatmap…")
    _plot_heatmap(oor_df,
                  title="Out-of-range rate (excl. DK/NA) per variable × model",
                  label="Out-of-range rate",
                  output_path=out_dir / "oor_rate_heatmap.png",
                  cmap="Blues", vmax=0.5)

    # ── CSV export ───────────────────────────────────────────────────────────
    dkna_df.to_csv(out_dir / "dkna_rate_by_variable_model.csv")
    oor_df.to_csv(out_dir / "oor_rate_by_variable_model.csv")
    print(f"  CSVs saved to {out_dir}")

    # ── Top problematic variables ────────────────────────────────────────────
    print("\n── Top 20 variables by overall DK/NA rate ─────────────────────────")
    top_dkna = dkna_df.mean(axis=1).sort_values(ascending=False).head(20)
    for var, rate in top_dkna.items():
        print(f"  {var:8s}  {rate:.1%}")

    print("\n── Top 20 variables by overall out-of-range rate ───────────────────")
    top_oor = oor_df.mean(axis=1).sort_values(ascending=False).head(20)
    for var, rate in top_oor.items():
        print(f"  {var:8s}  {rate:.1%}")


if __name__ == "__main__":
    main()
