"""
LLM Responses Analysis — PCA projection, radar charts, and alignment analysis.

This script:
1. Loads all LLM response CSVs from Surveys_responses/
2. Parses numeric responses (@@N@@ format), normalises [0,1], discards DK/NA
3. Projects per-country LLM mean vectors onto the PCA space fitted on human data
4. Produces a PCA scatter plot (one point per model×country)
5. Produces a radar chart comparing per-model topic averages with human language-
   family averages
6. Produces a radar chart showing LLM aggregate response rates by topic
7. Computes Kolmogorov-Smirnov test statistics comparing human and LLM responses
   at the language family level
8. Produces a heatmap visualizing LLM-human alignment
"""

import os
import re
import sys
import glob
import argparse
import contextlib
import io
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — avoids GUI hang on macOS
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from matplotlib.colors import to_rgb
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text

# Shared variable list and normalization (single source of truth)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survey_variables import PCA_SELECTED_VARS, normalize_variable  # noqa: E402
from scale_metadata import load_canonical_scale_metadata  # noqa: E402

# ── Paths (defaults) ─────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]   # EUValues/

DEFAULT_LLM_DIR     = Path(os.getenv("LLM_DIR", str(_ROOT / "Surveys_responses_scale_control")))
DEFAULT_HUMAN_CSV   = _ROOT / "Surveys_responses" / "Human_responses" / \
                      "moral_values_survey_with_languages_v2.csv"
DEFAULT_TOPICS_CSV  = _ROOT / "Surveys" / "Survey_metadata" / \
                      "questions_topic_matching_v3.csv"
DEFAULT_SCALE_CATALOG = _ROOT / "Surveys" / "Survey_metadata" / "question_scale_catalog.csv"
DEFAULT_SCALES_DIR  = _ROOT / "Surveys_parsed"
DEFAULT_OUTPUT_DIR  = _HERE.parent / "data_preprocessing" / "output"

# PCA_SELECTED_VARS and normalize_variable imported from EUValues/src/survey_variables.py

# ── Colorblind-safe palette (Paul Tol) ────────────────────────────────────────
PALETTE = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44",
    "#66CCEE", "#AA3377", "#BBBBBB", "#332288",
    "#44BB99", "#FFAABB", "#99DDFF",
]
MARKER_STYLES = ["o", "s", "^", "D", "v", "p", "h", "*", "X", "<", ">"]
ORIGIN_ORDER = ["European", "American", "Chinese", "Other"]
ORIGIN_EDGE_COLOR = {
    "European": "#1B9E77",
    "American": "#D95F02",
    "Chinese": "#7570B3",
    "Other": "#666666",
}
ORIGIN_LINESTYLE = {
    "European": "-",
    "American": "--",
    "Chinese": ":",
    "Other": "-.",
}


def infer_llm_family(model_name: str) -> str:
    """Infer a compact model-family label from a concrete model name."""
    m = str(model_name).lower().strip()
    if m.startswith("apertus"):
        return "Apertus"
    if m.startswith("gemma"):
        return "Gemma"
    if m.startswith("llama"):
        return "Llama"
    if m.startswith("smollm"):
        return "SmollM"
    if m.startswith("phi"):
        return "Phi"
    if m.startswith("aya") or "aya" in m:
        return "Aya"
    if m.startswith("minimistral"):
        return "Ministral"
    if m.startswith("ministral"):
        return "Ministral"
    if m.startswith("qwen"):
        return "Qwen"
    if m.startswith("deepseek"):
        return "DeepSeek"
    if m.startswith("alia"):
        return "Alia"
    if m.startswith("eurollm"):
        return "EuroLLM"
    if m.startswith("occiglot"):
        return "Occiglot"
    if m.startswith("salamandra"):
        return "Salamandra"
    token = re.split(r"[-_.]", m)[0].strip()
    return token.title() if token else str(model_name)


def build_compact_family_mapping(
    models: list[str],
    min_family_size_for_dedicated: int = 2,
    max_dedicated_families: int | None = None,
) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    """
    Build compact family mapping to avoid repeated symbols.

    Small families are grouped into "Others" so only larger families receive
    dedicated symbols.
    """
    raw_families = [infer_llm_family(m) for m in models]
    family_counts = pd.Series(raw_families).value_counts().to_dict()

    if max_dedicated_families is None:
        # Keep one slot for "Others" whenever it exists.
        max_dedicated_families = max(1, len(MARKER_STYLES) - 1)

    ranked = sorted(family_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    dedicated = [
        fam for fam, cnt in ranked
        if cnt >= min_family_size_for_dedicated
    ][:max_dedicated_families]

    model_display_family = {
        m: (infer_llm_family(m) if infer_llm_family(m) in dedicated else "Others")
        for m in models
    }
    display_families = sorted(set(model_display_family.values()))

    family_marker: dict[str, str] = {}
    marker_idx = 0
    for fam in display_families:
        if fam == "Others":
            continue
        family_marker[fam] = MARKER_STYLES[marker_idx % len(MARKER_STYLES)]
        marker_idx += 1

    if "Others" in display_families:
        other_marker = "X" if "X" not in family_marker.values() else MARKER_STYLES[marker_idx % len(MARKER_STYLES)]
        family_marker["Others"] = other_marker

    return family_marker, model_display_family, family_counts


def infer_model_origin(model_name: str) -> str:
    """Map model family to geographic origin used for texture encoding."""
    fam = infer_llm_family(model_name)
    m = str(model_name).lower().strip()
    if m.startswith("villanova"):
        return "European"
    if fam in {"Apertus", "EuroLLM", "Ministral", "Alia", "Occiglot", "Salamandra"}:
        return "European"
    if fam in {"Gemma", "Phi", "Aya", "Llama", "SmollM"}:
        return "American"
    if fam in {"Qwen", "DeepSeek"}:
        return "Chinese"
    return "Other"


def build_origin_legend_handles(models: list[str], marker_color: str = "#666666") -> list:
    """Build legend handles for origin mapping using outline colour."""
    origins_present = {infer_model_origin(m) for m in models}
    ordered_origins = [o for o in ORIGIN_ORDER if o in origins_present]
    return [
        mpl.lines.Line2D([0], [0], marker="o", color="w",
                         markerfacecolor="none",
                         markeredgecolor=ORIGIN_EDGE_COLOR[o],
                         markeredgewidth=1.6,
                         markersize=8,
                         label=o)
        for o in ordered_origins
    ]


def draw_model_point_with_origin_outline(ax,
                                         x: float,
                                         y: float,
                                         marker: str,
                                         size: float,
                                         origin: str,
                                         linewidth: float = 1.4,
                                         alpha: float = 0.98,
                                         zorder: int = 6):
    """Draw a transparent model point with outline colour encoding origin."""
    return ax.scatter([x], [y],
                      s=size,
                      marker=marker,
                      facecolors="none",
                      edgecolors=ORIGIN_EDGE_COLOR.get(origin, ORIGIN_EDGE_COLOR["Other"]),
                      linewidths=linewidth,
                      alpha=alpha,
                      zorder=zorder)


def build_family_shape_legend_handles(models: list[str], marker_color: str = "#444444") -> list:
    """Build legend handles for model-family marker mapping."""
    family_marker, _, family_counts = build_compact_family_mapping(models)
    return [
        mpl.lines.Line2D([0], [0], marker=mk, color="w",
                         markerfacecolor="none", markeredgecolor="#222222",
                         markeredgewidth=1.2,
                         markersize=8.5,
                         label=(f"{fam} (n={family_counts.get(fam, 0)})" if fam != "Others" else "Others (small families)"))
        for fam, mk in sorted(family_marker.items())
    ]

# Shared visual style for human country-language background points
BG_LIGHTEN_AMOUNT = 0.68
BG_POINT_ALPHA = 0.18
BG_POINT_SIZE_MAIN = 82
BG_POINT_SIZE_SECONDARY = 26
BG_LABEL_COLOR = "#666666"


def _lighten_color(color: str, amount: float = 0.55) -> tuple[float, float, float]:
    """Return a lighter, less saturated version of a color by blending with white.

    amount in [0, 1]:
        0.0 -> original color
        1.0 -> white
    """
    amount = max(0.0, min(1.0, float(amount)))
    r, g, b = to_rgb(color)
    return (r + (1.0 - r) * amount,
            g + (1.0 - g) * amount,
            b + (1.0 - b) * amount)

# ── Country-code → display name ───────────────────────────────────────────────
CODE_TO_NAME = {
    "al": "Albania",   "at": "Austria",    "az": "Azerbaijan",
    "ba": "Bosnia",    "cz": "Czechia",    "ee": "Estonia",
    "es": "Spain",     "fr": "France",     "gb": "Gt Britain",
    "ge": "Georgia",   "gr": "Greece",     "hr": "Croatia",
    "hu": "Hungary",   "it": "Italy",      "lt": "Lithuania",
    "me": "Montenegro","mk": "N. Macedonia","no": "Norway",
    "pl": "Poland",    "pt": "Portugal",   "ro": "Romania",
    "rs": "Serbia",    "ru": "Russia",     "se": "Sweden",
    "si": "Slovenia",  "sk": "Slovakia",   "ua": "Ukraine",
}

# Country name → language family (mirrors human_responses_analysis remapping)
URALIC_COUNTRIES  = {"Finland", "Hungary", "Estonia"}
OTHERS_COUNTRIES  = {"Albania", "Armenia", "Greece", "Turkey", "Azerbaijan",
                     "Cyprus", "Georgia", "Liechtenstein", "San Marino"}
FAMILY_REMAP      = {"Italic": "Romance"}


# ── Helpers ───────────────────────────────────────────────────────────────────

# Ordered fallback strategies for extracting the numeric response.
# Each entry: (strategy_name, compiled_pattern)
# The first capturing group must be the numeric value.
_PARSE_STRATEGIES = [
    # 1. Standard @@N@@ delimiter
    ("@@N@@",   re.compile(r"@@(\d+)@@")),
    # 2. Partial delimiter on the right:  @@N (no closing @@)
    ("@@N",     re.compile(r"@@(\d+)")),
    # 3. Partial delimiter on the left:   N@@ (no opening @@)
    ("N@@",     re.compile(r"(\d+)@@")),
    # 4. "Final answer: N" — number (1-3 digits) directly after the colon,
    #    optionally followed by whitespace, parenthesis, dash, newline or EOS.
    #    Covers: "Final answer: 3", "Final answer: 3 (label)", "Final answer: 1\n...",
    #            "Final answer: 1 - nagyon fontos", "Final answer: 1@<...>"
    ("label:N", re.compile(
        r"[Ff]inal\s+[Aa]nswer\s*:\s*(\d{1,3})(?=\s*(?:\(|-|@|\n|$))"
    )),
    # 5. "Final answer: (NN)" style (common in translated categorical answers)
    #    Covers: "Final answer: (01) Yes", "Final answer: (96) None"
    ("label:(N)", re.compile(
        r"[Ff]inal\s+[Aa]nswer\s*:\s*\((\d{1,3})\)"
    )),
    # 5. Any "<word(s)>: N" pattern in any language
    #    (Válasz, Risposta finale, Odpowiedź, Atsakymas, პასუხი, etc.)
    #    Restrict to 1-3 digits to avoid matching years or large codes.
    ("any:N",   re.compile(
        r":\s*(\d{1,3})(?=\s*(?:\(|-|\n|$))"
    )),
    # 6. Bare integer — the entire (stripped) string is a 1-3 digit number
    ("bare",    re.compile(r"^\s*(\d{1,3})\s*$")),
]

_STRATEGY_SCORE = {
    "@@N@@": 6,
    "label:N": 5,
    "@@N": 4,
    "N@@": 3,
    "any:N": 2,
    "bare": 1,
    None: 0,
}


def _extract_scale_codes(scale_text: str) -> set[int]:
    """Extract numeric option codes from a response_scale text."""
    if scale_text is None or (isinstance(scale_text, float) and np.isnan(scale_text)):
        return set()
    txt = str(scale_text)
    # Typical format: "1 (label) - 2 (label) - 88 (DK) - 99 (NA)"
    codes = {int(x) for x in re.findall(r"\b(\d{1,3})\s*\(", txt)}
    if codes:
        return codes
    # Fallback for uncommon formats where parentheses are absent
    return {int(x) for x in re.findall(r"\b\d{1,3}\b", txt)}


def _contiguous_range_from_codes(codes: set[int]) -> tuple[float, float] | None:
    """
    Infer the valid response range from a set of integer codes by taking the
    first contiguous block. This typically separates valid values (e.g. 1..10)
    from DK/NA codes (e.g. 88/99).
    """
    if not codes:
        return None
    vals = sorted(v for v in codes if v >= 0)
    if not vals:
        return None

    # Build contiguous runs and keep the longest one.
    # This preserves valid scales such as 0..10 while excluding DK/NA tails
    # such as 77/88/99.
    runs: list[tuple[int, int]] = []
    run_start = vals[0]
    prev = vals[0]
    for x in vals[1:]:
        if x == prev + 1:
            prev = x
            continue
        runs.append((run_start, prev))
        run_start = x
        prev = x
    runs.append((run_start, prev))

    # Rank by run length (desc), then prefer runs that contain 1,
    # then lower start value for determinism.
    def _rank(run: tuple[int, int]) -> tuple[int, int, int]:
        a, b = run
        length = b - a + 1
        contains_one = 1 if (a <= 1 <= b) else 0
        return (length, contains_one, -a)

    best_start, best_end = max(runs, key=_rank)
    return float(best_start), float(best_end)


def infer_variable_valid_ranges(llm_df: pd.DataFrame) -> dict:
    """
    Return {variable: (valid_min, valid_max)} inferred from response_scale.
    Falls back to observed parsed values when response_scale is unavailable.
    """
    ranges = {}
    if "response_scale" in llm_df.columns:
        for var, grp in llm_df.groupby("variable"):
            codes = set()
            for s in grp["response_scale"].dropna().astype(str).unique():
                codes |= _extract_scale_codes(s)
            rng = _contiguous_range_from_codes(codes)
            if rng is not None:
                ranges[var] = rng

    # Fallback: infer from observed parsed values (integer-only series)
    for var, grp in llm_df.groupby("variable"):
        if var in ranges:
            continue
        s = pd.to_numeric(grp["numeric_response"], errors="coerce").dropna()
        if len(s) == 0:
            continue
        int_like = s[np.isclose(s, np.round(s))].astype(int)
        rng = _contiguous_range_from_codes(set(int_like.tolist()))
        if rng is not None:
            ranges[var] = rng
        else:
            ranges[var] = (float(s.min()), float(s.max()))
    return ranges


def _parse_numeric(response: str, return_strategy: bool = False):
    """
    Extract a numeric response value using a cascade of fallback strategies.

    Parameters
    ----------
    response : str
        Raw model_response string.
    return_strategy : bool
        If True, return (value, strategy_name) instead of just value.

    Returns
    -------
    int | np.nan   (or (int|np.nan, str|None) when return_strategy=True)
    """
    s = str(response)
    for name, pattern in _PARSE_STRATEGIES:
        m = pattern.search(s)
        if m:
            # IMPORTANT: keep the raw extracted integer (e.g. 88 stays 88).
            # DK/NA exclusion is done later using the per-variable valid range.
            val = int(m.group(1))
            return (val, name) if return_strategy else val
    return (np.nan, None) if return_strategy else np.nan


def _deduplicate_variable_rows(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Keep a single best row per variable within one LLM response file.

    Selection priority (highest first):
      1) Parseable numeric response
      2) Numeric response inside inferred valid range from response_scale
      3) Stronger parse strategy (_STRATEGY_SCORE)
      4) Not marked as collapsed repeated-digit pattern
      5) Earlier row order (stable tie-break)

    This addresses files where the same variable appears multiple times and
    one row is parseable while another is not.
    """
    if "variable" not in df.columns or len(df) == 0:
        return df

    vc = df["variable"].value_counts(dropna=False)
    dup_vars = vc[vc > 1]
    if len(dup_vars) == 0:
        return df

    work = df.copy().reset_index(drop=True)
    work["_orig_order"] = np.arange(len(work))
    keep_idx = []

    groups_exactly_one_parseable = 0
    groups_multiple_parseable = 0
    groups_none_parseable = 0

    for var, grp in work.groupby("variable", sort=False):
        if len(grp) == 1:
            keep_idx.append(grp.index[0])
            continue

        # Infer valid range from this duplicated variable block if possible
        codes = set()
        if "response_scale" in grp.columns:
            for s in grp["response_scale"].dropna().astype(str).unique():
                codes |= _extract_scale_codes(s)
        valid_range = _contiguous_range_from_codes(codes)

        parseable = grp["numeric_response"].notna()
        n_parse = int(parseable.sum())
        if n_parse == 1:
            groups_exactly_one_parseable += 1
        elif n_parse > 1:
            groups_multiple_parseable += 1
        else:
            groups_none_parseable += 1

        if valid_range is not None:
            lo, hi = valid_range
            in_range = parseable & grp["numeric_response"].between(lo, hi, inclusive="both")
        else:
            in_range = parseable

        strategy_score = grp["_parse_strategy"].map(_STRATEGY_SCORE).fillna(0).astype(int)
        not_collapsed = (~grp["_collapsed"]).astype(int)

        rank_df = pd.DataFrame({
            "_idx": grp.index,
            "_is_parseable": parseable.astype(int),
            "_in_range": in_range.astype(int),
            "_strategy": strategy_score,
            "_not_collapsed": not_collapsed,
            "_orig_order": grp["_orig_order"],
        })

        best = rank_df.sort_values(
            by=["_is_parseable", "_in_range", "_strategy", "_not_collapsed", "_orig_order"],
            ascending=[False, False, False, False, True],
        ).iloc[0]
        keep_idx.append(int(best["_idx"]))

    deduped = work.loc[sorted(keep_idx)].drop(columns=["_orig_order"]).reset_index(drop=True)

    if verbose:
        print("    [DEDUP] variable-level row selection:")
        print(f"      rows before/after: {len(df)} -> {len(deduped)}")
        print(f"      duplicated variable groups: {len(dup_vars)}")
        print(f"      groups with exactly 1 parseable row: {groups_exactly_one_parseable}")
        print(f"      groups with >1 parseable rows: {groups_multiple_parseable}")
        print(f"      groups with 0 parseable rows: {groups_none_parseable}")

    return deduped


# ── Data loading ──────────────────────────────────────────────────────────────

def _parse_llm_file(fpath: Path, verbose: bool = False):
    """
    Parse a single llm_survey_*.csv file.
    Returns a DataFrame with _model, _country_code, variable, numeric_response,
    or None if the file cannot be parsed.
    Emits verbose diagnostic prints when verbose=True.
    """
    m = re.match(r"llm_survey_(.+)_responses_([a-z]+)\.csv", fpath.name)
    if not m:
        if verbose:
            print(f"  [SKIP] filename does not match pattern: {fpath.name}")
        return None

    model_name   = m.group(1)
    country_code = m.group(2)

    try:
        df = pd.read_csv(fpath)
    except Exception as e:
        print(f"  [ERROR] could not read {fpath.name}: {e}")
        return None

    # ── Required columns ──────────────────────────────────────────────────────
    required = {"model_response", "variable"}
    missing  = required - set(df.columns)
    if missing:
        print(f"  [ERROR] {fpath.name} missing columns: {missing}")
        return None

    df["_model"]        = model_name
    df["_country_code"] = country_code

    # ── Parse numeric response ────────────────────────────────────────────────
    parsed = df["model_response"].apply(lambda r: _parse_numeric(r, return_strategy=True))
    df["numeric_response"] = parsed.apply(lambda t: t[0])
    df["_parse_strategy"]  = parsed.apply(lambda t: t[1])

    # Flag repeated-digit tokens (e.g. @@88@@, @@999@@) for diagnostics only.
    _at_pat = re.compile(r"@@(\d{2,})@@")
    def _was_collapsed(resp):
        m2 = _at_pat.search(str(resp))
        if m2:
            d = m2.group(1)
            return len(set(d)) == 1
        return False
    df["_collapsed"] = df["model_response"].apply(_was_collapsed)

    # Keep one best row per variable when duplicates exist in this file.
    df = _deduplicate_variable_rows(df, verbose=verbose)

    n_total   = len(df)
    n_parsed  = df["numeric_response"].notna().sum()
    n_failed  = n_total - n_parsed

    if verbose:
        pct_ok = 100 * n_parsed / n_total if n_total else 0
        print(f"  [{model_name} / {country_code}] "
              f"{n_parsed}/{n_total} responses parsed ({pct_ok:.1f}%)"
              + (f"  — {n_failed} FAILED" if n_failed else ""))

        # Strategy breakdown
        strat_counts = df["_parse_strategy"].value_counts(dropna=False)
        strat_str = "  ".join(
            f"{s}={c}" for s, c in strat_counts.items()
        )
        print(f"    Strategies: {strat_str}")

        # Rows where parsing still failed
        if n_failed:
            failed_rows   = df[df["numeric_response"].isna()]
            unique_failed = failed_rows["model_response"].unique()
            print(f"    Unparseable ({min(3, len(unique_failed))} examples):")
            for ex in unique_failed[:3]:
                preview = str(ex).replace("\n", "\\n")[:120]
                print(f"      {repr(preview)}")

        # ── Unexpected numeric values ─────────────────────────────────────────
        parsed_vals  = df["numeric_response"].dropna()
        out_of_range = parsed_vals[(parsed_vals < 1) | (parsed_vals > 10)]
        if len(out_of_range):
            print(f"    [WARN] {len(out_of_range)} values outside [1,10]: "
                  f"{sorted(out_of_range.unique())[:10]}")

        # ── Variable coverage ─────────────────────────────────────────────────
        present     = set(df["variable"].dropna().unique())
        missing_pca = set(PCA_SELECTED_VARS) - present
        if missing_pca:
            print(f"    [WARN] {len(missing_pca)} PCA variables absent: "
                  f"{sorted(missing_pca)[:10]}{'…' if len(missing_pca)>10 else ''}")

        # ── Duplicate variable rows ───────────────────────────────────────────
        dups = df["variable"].value_counts()
        dups = dups[dups > 1]
        if len(dups):
            print(f"    [WARN] {len(dups)} variables appear >1 time: "
                  f"{dups.index.tolist()[:5]}")

    return df


def _check_model_duplicates(combined: pd.DataFrame, verbose: bool = False):
    """
    Check whether any two models produce bit-for-bit identical response vectors
    for the same country.  Flags pairs that have suspiciously high correlation.
    Always checks the minimistral3 family explicitly.
    """
    models = sorted(combined["_model"].unique())

    # Build (model, country) → mean-per-variable pivot
    pivot = (combined
             .groupby(["_model", "_country_code", "variable"])["numeric_response"]
             .mean()
             .unstack("variable"))  # rows = (model, country)

    # ── minimistral3 family ───────────────────────────────────────────────────
    mini_models = [m for m in models if "minimistral3" in m]
    if len(mini_models) >= 2:
        print(f"\n[Duplicate check] minimistral3 family: {mini_models}")
        for i, ma in enumerate(mini_models):
            for mb in mini_models[i+1:]:
                sub_a = pivot.loc[ma] if ma in pivot.index.get_level_values(0) else None
                sub_b = pivot.loc[mb] if mb in pivot.index.get_level_values(0) else None
                if sub_a is None or sub_b is None:
                    print(f"  [WARN] one of {ma}, {mb} not in pivot")
                    continue
                common_countries = sub_a.index.intersection(sub_b.index)
                if len(common_countries) == 0:
                    print(f"  {ma} vs {mb}: no common countries")
                    continue
                a = sub_a.loc[common_countries].fillna(0).infer_objects(copy=False)
                b = sub_b.loc[common_countries].fillna(0).infer_objects(copy=False)
                are_identical = (a.values == b.values).all()
                diff_mean = float((a - b).abs().mean().mean())
                print(f"  {ma} vs {mb} — identical={are_identical}  "
                      f"mean |diff|={diff_mean:.6f}  "
                      f"({len(common_countries)} countries)")
                if are_identical:
                    print("  [ALERT] Both models produce EXACTLY the same responses!")
                elif diff_mean < 1e-6:
                    print("  [ALERT] Responses are numerically indistinguishable.")
    else:
        if verbose:
            print("[Duplicate check] minimistral3: fewer than 2 variants found — skipping")

    # ── General pairwise correlation (verbose only) ───────────────────────────
    if verbose and len(models) >= 2:
        print("\n[Duplicate check] Pairwise mean-response correlation (all models):")
        model_vecs = {}
        for mdl in models:
            if mdl in pivot.index.get_level_values(0):
                model_vecs[mdl] = pivot.loc[mdl].fillna(0).infer_objects(copy=False).values.flatten()
        pairs_flagged = 0
        for i, ma in enumerate(models):
            for mb in models[i+1:]:
                if ma not in model_vecs or mb not in model_vecs:
                    continue
                va, vb = model_vecs[ma], model_vecs[mb]
                min_len = min(len(va), len(vb))
                if min_len == 0:
                    continue
                corr = float(np.corrcoef(va[:min_len], vb[:min_len])[0, 1])
                if corr > 0.998:
                    print(f"  [WARN] {ma} vs {mb}: r={corr:.6f} (suspiciously high)")
                    pairs_flagged += 1
        if pairs_flagged == 0:
            print("  No suspiciously correlated model pairs found (threshold r>0.998).")


def load_llm_responses(llm_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """
    Load all llm_survey_*.csv files.
    Returns a DataFrame with columns:
        _model, _country_code, variable, numeric_response, norm_response
    """
    files = sorted(llm_dir.glob("llm_survey_*.csv"))
    if not files:
        raise FileNotFoundError(f"No llm_survey_*.csv files found in {llm_dir}")
    print(f"Found {len(files)} LLM response files.")

    chunks = []
    for fpath in files:
        df = _parse_llm_file(fpath, verbose=verbose)
        if df is not None:
            chunks.append(df)

    if not chunks:
        raise ValueError("No valid LLM response files could be parsed.")

    combined = pd.concat(chunks, ignore_index=True)

    # ── Global parsing summary ────────────────────────────────────────────────
    n_total  = len(combined)
    n_parsed = combined["numeric_response"].notna().sum()
    print(f"\n── Parse summary ────────────────────────────────────────────────")
    print(f"  Total rows   : {n_total:,}")
    print(f"  Parsed OK    : {n_parsed:,}  ({100*n_parsed/n_total:.1f}%)")
    print(f"  Failed       : {n_total - n_parsed:,}")
    print(f"  Models       : {combined['_model'].nunique()}  "
          f"— {sorted(combined['_model'].unique())}")
    print(f"  Countries    : {combined['_country_code'].nunique()}  "
          f"— {sorted(combined['_country_code'].unique())}")
    print(f"  Variables    : {combined['variable'].nunique()}")

    # ── Repeated-digit collapse summary ──────────────────────────────────────
    n_collapsed = combined["_collapsed"].sum()
    print(f"  Collapsed    : {n_collapsed:,}  "
          f"({100*n_collapsed/n_total:.1f}%)  "
          f"[repeated-digit @@ patterns, e.g. @@88@@ → 8]")
    if n_collapsed > 0:
        print("  Collapsed breakdown by model:")
        for _m in sorted(combined["_model"].unique()):
            _nc = combined.loc[combined["_model"] == _m, "_collapsed"].sum()
            _nt = (combined["_model"] == _m).sum()
            if _nc > 0:
                print(f"    {_m:25s}  {_nc:,}  ({100*_nc/_nt:.1f}%)")

    valid_ranges = infer_variable_valid_ranges(combined)

    # ── Invalid-by-range rates per model (DK/NA-like special codes) ─────────
    print(f"\n── Invalid-by-range rates per model (DK/NA-like) ───────────────────")
    for _m in sorted(combined["_model"].unique()):
        _sub = combined[combined["_model"] == _m]
        _n_m = len(_sub)
        bad = 0
        for _, r in _sub.iterrows():
            v = r["variable"]
            x = r["numeric_response"]
            if pd.isna(x):
                continue
            lo_hi = valid_ranges.get(v)
            if lo_hi is None:
                continue
            lo, hi = lo_hi
            if x < lo or x > hi:
                bad += 1
        print(f"  {_m:25s}  {100*bad/_n_m:.1f}%  ({bad:,} / {_n_m:,})")

    # ── Coverage matrix: models × countries ──────────────────────────────────
    if verbose:
        coverage = (combined.groupby(["_model", "_country_code"])
                    .size().unstack(fill_value=0))
        print(f"\n── Rows per (model × country) — 0 = missing ─────────────────")
        print(coverage.to_string())

        # Flag any (model, country) with zero parsed responses
        parsed_cov = (combined[combined["numeric_response"].notna()]
                      .groupby(["_model", "_country_code"])
                      .size().unstack(fill_value=0))
        zero_pairs = [(m, c) for m in parsed_cov.index
                      for c in parsed_cov.columns if parsed_cov.loc[m, c] == 0]
        if zero_pairs:
            print(f"\n  [WARN] {len(zero_pairs)} (model, country) pairs with 0 "
                  f"parsed responses:")
            for mc in zero_pairs[:20]:
                print(f"    {mc}")

    # ── Duplicate / identical model check ────────────────────────────────────
    _check_model_duplicates(combined, verbose=verbose)

    # ── Normalise per-variable ────────────────────────────────────────────────
    normed = []
    for var, grp in combined.groupby("variable"):
        grp = grp.copy()
        grp["norm_response"] = normalize_variable(grp["numeric_response"])
        normed.append(grp)
    combined = pd.concat(normed, ignore_index=True)

    print(f"\nLoaded {len(combined):,} rows — "
          f"{combined['_model'].nunique()} models × "
          f"{combined['_country_code'].nunique()} countries × "
          f"{combined['variable'].nunique()} variables")
    return combined


def load_human_data(human_csv: Path, topics_csv: Path, valid_ranges: dict | None = None):
    """Load and normalise human survey data.

    Returns
    -------
    df_norm      : DataFrame with v-columns normalised to [0, 1]
    topic_df     : variable → category mapping
    c2f          : country name → language family mapping
    country_col  : name of the country column
    v_cols       : list of numeric variable column names
    norm_params  : dict {variable: (min_val, max_val)} defining the baseline
                   substantive range for each variable. When available, ranges
                   inferred from the survey scale are preferred over the human
                   observed support so valid-but-unseen human codes (e.g. `3`
                   in a `1..3` scale where humans only answered `1..2`) remain
                   part of the normalisation baseline.
    """
    print(f"Loading human data from: {human_csv}")
    df = pd.read_csv(human_csv, low_memory=False)

    topic_df_raw = pd.read_csv(topics_csv, sep=";")

    # Backward/forward compatible topic schema:
    # - legacy: variable_name;variable_label;category
    # - current: EVS 2017 Variable Name;EVS 2017 Variable Label;Category;topic;source_url
    col_lookup = {str(c).strip().lower(): c for c in topic_df_raw.columns}

    var_col = (
        col_lookup.get("variable_name")
        or col_lookup.get("evs 2017 variable name")
        or (topic_df_raw.columns[0] if len(topic_df_raw.columns) > 0 else None)
    )
    label_col = (
        col_lookup.get("variable_label")
        or col_lookup.get("evs 2017 variable label")
        or (topic_df_raw.columns[1] if len(topic_df_raw.columns) > 1 else None)
    )
    # IMPORTANT: for radar plots per category, always prefer the "topic"
    # column when present. Fall back to "Category" only when "topic" is absent.
    category_col = col_lookup.get("topic")
    if category_col is None:
        category_col = col_lookup.get("category") or (topic_df_raw.columns[2] if len(topic_df_raw.columns) > 2 else None)

    if var_col is None or label_col is None or category_col is None:
        raise ValueError(
            "Topic mapping CSV must contain variable, label, and category/topic columns. "
            f"Found columns: {list(topic_df_raw.columns)}"
        )

    topic_df = topic_df_raw[[var_col, label_col, category_col]].copy()
    topic_df.columns = ["variable_name", "variable_label", "category"]
    topic_df["variable_name"] = topic_df["variable_name"].astype(str).str.strip()
    topic_df["variable_label"] = topic_df["variable_label"].astype(str).str.strip()
    topic_df["category"] = topic_df["category"].astype(str).str.strip()
    topic_df = topic_df[topic_df["variable_name"].ne("")]
    print(f"Using category source column: {category_col}")

    # Identify numeric variable columns
    v_cols = [c for c in df.columns if c.startswith("v") and len(c) > 1 and c[1].isdigit()]

    # ── Compute per-variable baseline normalisation params ────────────────────
    # Must be done BEFORE normalising df. Prefer the substantive survey range
    # when available, and only fall back to the human observed range when the
    # survey scale cannot be inferred.
    norm_params: dict = {}
    for col in v_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if valid_ranges and col in valid_ranges:
            lo, hi = valid_ranges[col]
            s = s.where((s >= lo) & (s <= hi), other=np.nan)
            if hi != lo:
                norm_params[col] = (float(lo), float(hi))
                continue
        valid = s.dropna()
        if len(valid) > 0 and valid.max() != valid.min():
            norm_params[col] = (float(valid.min()), float(valid.max()))
        # Variables with no variance or all-NaN are omitted — fallback to
        # LLM-only normalisation for those (edge case).

    # ── Normalise human v-columns to [0, 1] using the same baseline ───────────
    df_norm = df.copy()
    for col in v_cols:
        params = norm_params.get(col)
        raw = pd.to_numeric(df[col], errors="coerce")
        if valid_ranges and col in valid_ranges:
            lo, hi = valid_ranges[col]
            raw = raw.where((raw >= lo) & (raw <= hi), other=np.nan)
        if params is not None:
            min_h, max_h = params
            raw = raw.where((raw >= min_h) & (raw <= max_h), other=np.nan)
            df_norm[col] = ((raw - min_h) / (max_h - min_h)).clip(0.0, 1.0)
        else:
            df_norm[col] = normalize_variable(raw)

    # Build country → family mapping (mirrors human_responses_analysis.py)
    country_col = "country_str" if "country_str" in df.columns else "c_abrv"
    lang_col    = "country_group"

    c2f = {}
    for _, row in df[[country_col, lang_col]].drop_duplicates().iterrows():
        cname = row[country_col]
        fam   = FAMILY_REMAP.get(str(row[lang_col]), str(row[lang_col]))
        if cname in URALIC_COUNTRIES:
            fam = "Uralic"
        if cname in OTHERS_COUNTRIES:
            fam = "Others"
        c2f[cname] = fam

    return df_norm, topic_df, c2f, country_col, v_cols, norm_params


def _apply_human_normalization(llm_df: pd.DataFrame,
                               norm_params: dict,
                               valid_ranges: dict | None = None) -> pd.DataFrame:
    """
    Re-normalise LLM ``numeric_response`` values using human-derived min/max
    so that LLM and human values share the exact same [0, 1] scale.

    Why this is necessary
    ---------------------
    ``load_llm_responses`` normalises per-variable using only the LLM responses'
    own observed range.  Two problems arise:

    1. **Degenerate case**: when all LLM responses for a variable are identical
       (max == min), ``normalize_variable`` returns the raw value unchanged
       (e.g. 3.0, 88.0, 802.0) because a constant series cannot be scaled.

    2. **Scale mismatch**: even in the non-degenerate case the LLM-observed
       range usually differs from the human-observed range.  A human answer
       of "3" on a 1-4 scale normalises to 0.67; an LLM that only ever
       answers 3–4 would normalise "3" to 0.00.  The two 0.67 and 0.00
       values are not comparable, so the PCA projection is invalid.

    Fix
    ---
    Apply the per-variable (min_h, max_h) computed from the human dataset:

        norm = (llm_raw - min_h) / (max_h - min_h) , clipped to [0, 1]

    LLM values outside the baseline substantive range (e.g. 88, 99, 802 —
    likely DK/NA codes or parsing artefacts) are set to NaN so they do not bias
    the per-country mean vector.
    """
    result = []
    for var, grp in llm_df.groupby("variable"):
        grp = grp.copy()
        params = norm_params.get(var)
        raw = pd.to_numeric(grp["numeric_response"], errors="coerce")
        # Exclude values outside the variable-specific valid range first.
        if valid_ranges and var in valid_ranges:
            lo, hi = valid_ranges[var]
            raw = raw.where((raw >= lo) & (raw <= hi), other=np.nan)
        if params is not None:
            min_h, max_h = params
            # Values strictly outside the baseline substantive range are DK/NA
            # artefacts (88, 99, 802, …) — treat as missing.
            raw = raw.where((raw >= min_h) & (raw <= max_h), other=np.nan)
            grp["norm_response"] = ((raw - min_h) / (max_h - min_h)).clip(0.0, 1.0)
        else:
            # Variable not in human data — fall back to LLM-only normalisation
            grp["norm_response"] = normalize_variable(raw)
        result.append(grp)
    return pd.concat(result, ignore_index=True)


# ── PCA (fit on humans, project LLM) ─────────────────────────────────────────

def build_pca_data(df_human_norm: pd.DataFrame,
                   country_col: str,
                   v_cols: list,
                   c2f: dict):
    """
    Fit PCA on human country-mean vectors (same as human_responses_analysis.py).
    Returns fitted (pca, scaler, X_pca_human, countries_human, families_human).
    """
    EXCLUDE = {"Greece", "GR", "GRC", "Greek"}
    pca_cols = [c for c in PCA_SELECTED_VARS if c in v_cols]

    country_data, countries_h, families_h = [], [], []
    for country in sorted(df_human_norm[country_col].dropna().unique()):
        if country in EXCLUDE:
            continue
        sub = df_human_norm[df_human_norm[country_col] == country]
        # pandas mean() naturally skips NaN (DK/NA-excluded values).
        # NaN is kept if all responses for a variable are missing for this country.
        means = [float(sub[c].mean()) for c in pca_cols]
        country_data.append(means)
        countries_h.append(country)
        fam = c2f.get(country, "Unknown")
        families_h.append(fam)

    X = np.array(country_data, dtype=float)
    # Keep PCA basis equivalent to human_responses_analysis.plot_pca_country_language_points:
    # missing country-variable means are mapped to 0.0 before scaling/PCA.
    X = np.nan_to_num(X, nan=0.0)
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca     = PCA(n_components=2)
    X_pca   = pca.fit_transform(X_scaled)

    print(f"\nPCA fitted on {len(countries_h)} human country means "
          f"({len(pca_cols)} variables).")
    print(f"  PC1={pca.explained_variance_ratio_[0]:.2%}  "
          f"PC2={pca.explained_variance_ratio_[1]:.2%}  "
          f"Total={pca.explained_variance_ratio_.sum():.2%}")

    return pca, scaler, pca_cols, X_pca, countries_h, families_h


def project_llm_onto_pca(llm_df: pd.DataFrame,
                          pca, scaler, pca_cols: list):
    """
    For each (model, country_code) pair compute the mean normalised response
    vector and project it onto the fitted PCA space.

    Returns a DataFrame with columns:
        model, country_code, country_name, PC1, PC2
    """
    rows = []
    for (model, code), grp in llm_df.groupby(["_model", "_country_code"]):
        vec = []
        for var in pca_cols:
            sub = grp.loc[grp["variable"] == var, "norm_response"]
            val = float(sub.mean()) if len(sub) > 0 and not np.isnan(sub.mean()) \
                  else np.nan
            vec.append(val)
        vec_arr   = np.array(vec, dtype=float)
        available = ~np.isnan(vec_arr)
        if not available.any():
            # No valid responses at all for this (model, country) pair — skip.
            continue
        # Partial PCA projection: scale and project only available dimensions.
        # Variables with all-NaN responses (DK/NA-only or not asked) are excluded
        # entirely — not imputed — so they cannot bias the PC scores.
        X_scaled_avail   = ((vec_arr[available] - scaler.mean_[available])
                            / scaler.scale_[available])
        X_centered_avail = X_scaled_avail - pca.mean_[available]
        coords = pca.components_[:, available] @ X_centered_avail
        rows.append({
            "model":        model,
            "country_code": code,
            "country_name": CODE_TO_NAME.get(code, code.upper()),
            "PC1":          coords[0],
            "PC2":          coords[1],
        })
    return pd.DataFrame(rows)


def _infer_family_from_language_label(lang_value: str, fallback_family: str) -> str:
    """Infer language-family from language label; fallback to country family."""
    if lang_value is None:
        return fallback_family
    s = str(lang_value).strip().lower()
    if s in {"", "nan", "na", "n/a", "none", "null"}:
        return fallback_family

    # Germanic
    if any(k in s for k in ["deutsch", "german", "english", "norsk", "norwegian", "svenska", "swedish", "íslenska", "icelandic", "dutch"]):
        return "Germanic"
    # Romance
    if any(k in s for k in ["français", "french", "italiano", "español", "spanish", "català", "galego", "portugu", "român", "romana", "română"]):
        return "Romance"
    # Balto-Slavic
    if any(k in s for k in ["рус", "срп", "bosnian", "hrvats", "sloven", "slovak", "slovensk", "češt", "czech", "polski", "ukrain", "latvie", "lietu", "lithuan", "serbian", "croatian"]):
        return "Balto-Slavic"
    # Uralic
    if any(k in s for k in ["magyar", "finn", "eesti", "estonian"]):
        return "Uralic"
    # Others
    if any(k in s for k in ["euskara", "armenian", "azerbaijani", "georgian", "albanian", "turkish", "greek"]):
        return "Others"

    return fallback_family


def build_human_country_language_background(
    df_human_norm: pd.DataFrame,
    country_col: str,
    pca,
    scaler,
    pca_cols: list,
    c2f: dict,
) -> pd.DataFrame:
    """
    Build human PCA background points with the same country-language logic used
    in human_responses_analysis.py:
      - aggregate language subgroups into country-family weighted points
      - split a country only when >1 inferred language family is present
      - unknown-language rows kept only if unknown is the sole language group.
    """
    countries = sorted(df_human_norm[country_col].dropna().unique())
    countries = [c for c in countries if c not in {"Greece", "GR", "GRC", "Greek"}]

    language_col = None
    for cand in ["language_str", "language", "lang", "interview_language"]:
        if cand in df_human_norm.columns:
            language_col = cand
            break

    work = df_human_norm.copy()
    points = []

    if language_col is not None:
        lang = work[language_col].astype("string").str.strip()
        missing_like = {"", "nan", "na", "n/a", "none", "null"}
        lang = lang.mask(lang.str.lower().isin(missing_like), other=pd.NA)
        work["_lang_clean"] = lang

        keep_mask = pd.Series(False, index=work.index)
        for _, g in work.groupby(country_col, dropna=False):
            n_groups_total = len(g["_lang_clean"].drop_duplicates())
            if n_groups_total <= 1:
                keep_mask.loc[g.index] = True
            else:
                keep_mask.loc[g[g["_lang_clean"].notna()].index] = True
        work = work[keep_mask].copy()

        for country in countries:
            sub_country = work[work[country_col] == country]
            if sub_country.empty:
                continue

            fallback_family = c2f.get(country, "Unknown")
            lang_rows = []
            for lang_val, sub_lang in sub_country.groupby("_lang_clean", dropna=False):
                if sub_lang.empty:
                    continue
                vals = [float(sub_lang[col].mean()) if col in sub_lang.columns else np.nan for col in pca_cols]
                vec = np.nan_to_num(np.asarray(vals, dtype=float), nan=0.0)
                lang_label = str(lang_val) if pd.notna(lang_val) else "Unknown language"
                fam_from_lang = _infer_family_from_language_label(lang_label, fallback_family)
                lang_rows.append({
                    "country": country,
                    "language": lang_label,
                    "family": fam_from_lang,
                    "vec": vec,
                    "n": int(len(sub_lang)),
                })

            if not lang_rows:
                continue

            lang_df = pd.DataFrame(lang_rows)
            fam_rows = []
            for fam, gfam in lang_df.groupby("family", dropna=False):
                total_n_f = float(gfam["n"].sum())
                if total_n_f <= 0:
                    continue
                w = (gfam["n"].to_numpy(dtype=float) / total_n_f).reshape(-1, 1)
                mat = np.vstack(gfam["vec"].to_list())
                vec_f = (w * mat).sum(axis=0)
                fam_rows.append({
                    "country": country,
                    "family": str(fam),
                    "vec": vec_f,
                    "n_respondents": int(total_n_f),
                    "is_split_country": len(set(lang_df["family"].tolist())) > 1,
                    "languages_grouped": "; ".join(sorted(set(gfam["language"].astype(str).tolist()))),
                })

            for rr in fam_rows:
                vec_scaled = scaler.transform(np.asarray(rr["vec"], dtype=float).reshape(1, -1))
                coords = pca.transform(vec_scaled)[0]
                points.append({
                    "country": rr["country"],
                    "language": rr.get("languages_grouped", ""),
                    "family": rr["family"],
                    "PC1": float(coords[0]),
                    "PC2": float(coords[1]),
                    "n_respondents": int(rr["n_respondents"]),
                    "is_split_country": bool(rr["is_split_country"]),
                    "label": (
                        f"{rr['country']} [{rr.get('languages_grouped', '')}]"
                        if bool(rr["is_split_country"]) else rr["country"]
                    ),
                })
    else:
        # Fallback: one weighted country point
        for country in countries:
            sub_country = work[work[country_col] == country]
            if sub_country.empty:
                continue
            vals = [float(sub_country[col].mean()) if col in sub_country.columns else np.nan for col in pca_cols]
            vec = np.nan_to_num(np.asarray(vals, dtype=float), nan=0.0)
            vec_scaled = scaler.transform(vec.reshape(1, -1))
            coords = pca.transform(vec_scaled)[0]
            points.append({
                "country": country,
                "language": "Language unavailable",
                "family": c2f.get(country, "Unknown"),
                "PC1": float(coords[0]),
                "PC2": float(coords[1]),
                "n_respondents": int(len(sub_country)),
                "is_split_country": False,
                "label": country,
            })

    return pd.DataFrame(points)


def build_human_country_reference_points(human_bg: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse country-language background points into one canonical point per
    country using respondent-count weighted averages in PCA space.

    This guarantees that any country-level human anchor used in downstream PCA
    plots is derived from the *same* underlying human projection as the
    country-language background shown in the other PCA figures.
    """
    if human_bg.empty:
        return pd.DataFrame(columns=["country", "PC1", "PC2", "n_respondents"])

    work = human_bg.copy()
    work["n_respondents"] = pd.to_numeric(work.get("n_respondents", 1), errors="coerce").fillna(1.0)

    rows = []
    for country, g in work.groupby("country", dropna=False):
        ww = g["n_respondents"].to_numpy(dtype=float)
        if ww.sum() <= 0:
            ww = np.ones(len(g), dtype=float)
        pc1 = float(np.average(g["PC1"].to_numpy(dtype=float), weights=ww))
        pc2 = float(np.average(g["PC2"].to_numpy(dtype=float), weights=ww))
        rows.append({
            "country": country,
            "PC1": pc1,
            "PC2": pc2,
            "n_respondents": int(ww.sum()),
        })

    return pd.DataFrame(rows)


# ── Plot 1: PCA scatter ───────────────────────────────────────────────────────

def plot_pca(human_bg: pd.DataFrame,
             llm_proj: pd.DataFrame,
             pca,
             output_path=None,
             column_width: float = 3.5):
    """
    Scatter plot projecting LLM model centroids onto the human PCA space.
    Each model is represented by a single point (mean across all countries).
    Human country-language points are shown as family-coloured markers for context.
    
    Args:
        column_width: Figure width in inches (default 3.5 for single column, use 7.0 for two columns)
    """
    # ── Compute one centroid per model ────────────────────────────────────────
    centroids = (llm_proj
                 .groupby("model")[["PC1", "PC2"]]
                 .mean()
                 .reset_index())

    with plt.style.context(["science", "no-latex"]):
        mpl.rcParams.update({
            "font.size": 11, "axes.labelsize": 12,
            "xtick.labelsize": 10, "ytick.labelsize": 10,
            "legend.fontsize": 9,
        })

        # Map column_width to optimized figure dimensions
        # 1col: 5.8x4.1 (singlecol), 2col: 8.5x5.5 (improved readability)
        figsize_map = {3.5: (5.8, 4.1), 7.0: (8.5, 5.5)}
        figsize = figsize_map.get(column_width, (column_width, column_width * 0.65))
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axhline(0, color="#777777", lw=0.7, alpha=0.6)
        ax.axvline(0, color="#777777", lw=0.7, alpha=0.6)

        # ── Human reference: country-language background, coloured by family ───
        unique_fam = sorted(human_bg["family"].dropna().unique().tolist())
        fam_colors = {f: PALETTE[i % len(PALETTE)] for i, f in enumerate(unique_fam)}

        # Human background format mirrors pca_countries_by_language_points
        bg_texts = []
        for _, r in human_bg.iterrows():
            x, y = r["PC1"], r["PC2"]
            fam = r["family"]
            bg_col = _lighten_color(fam_colors[fam], amount=BG_LIGHTEN_AMOUNT)
            label = str(r.get("label", r.get("country", "")))
            ax.scatter(x, y, s=BG_POINT_SIZE_MAIN, color=bg_col, marker="o",
                       edgecolors="white", linewidth=0.6, alpha=BG_POINT_ALPHA, zorder=2)
            tbg = ax.text(x, y, label, fontsize=8.0, color=BG_LABEL_COLOR,
                          ha="center", va="bottom", zorder=3)
            bg_texts.append(tbg)

        # ── Family-legend handles (human) ─────────────────────────────────────
        family_handles = [
            mpl.lines.Line2D([0], [0], marker="o", color="w",
                             markerfacecolor=fam_colors[f], markersize=8,
                             label=f, alpha=0.7)
            for f in unique_fam
        ]

        # ── LLM model centroids ───────────────────────────────────────────────
        models = sorted(centroids["model"].unique())
        family_marker, model_display_family, _family_counts = build_compact_family_mapping(models)
        model_marker = {m: family_marker[model_display_family[m]] for m in models}
        model_origin = {m: infer_model_origin(m) for m in models}

        texts, pts = [], []
        for model in models:
            row = centroids[centroids["model"] == model].iloc[0]
            mk  = model_marker[model]
            origin = model_origin[model]
            draw_model_point_with_origin_outline(
                ax=ax,
                x=float(row["PC1"]),
                y=float(row["PC2"]),
                marker=mk,
                size=120,
                origin=origin,
                linewidth=1.8,
                alpha=0.95,
                zorder=6,
            )
            t = ax.text(row["PC1"], row["PC2"], model,
                        fontsize=8, color="#111111", fontweight="medium", zorder=7)
            texts.append(t)
            pts.append((row["PC1"], row["PC2"]))

        with contextlib.redirect_stdout(io.StringIO()):
            adjust_text(bg_texts, ax=ax,
                        expand=(1.18, 1.26),
                        force_text=(0.23, 0.34),
                        force_points=(0.10, 0.16),
                        ensure_inside_axes=True)

        with contextlib.redirect_stdout(io.StringIO()):
            adjust_text(texts, ax=ax,
                        expand=(1.4, 1.6),
                        force_text=(0.4, 0.6),
                        force_points=(0.15, 0.25),
                        ensure_inside_axes=False)

        # Connector lines for displaced labels
        xr = ax.get_xlim()[1] - ax.get_xlim()[0]
        yr = ax.get_ylim()[1] - ax.get_ylim()[0]
        for t, (px, py) in zip(texts, pts):
            tx, ty = t.get_position()
            if abs(tx - px) / xr > 0.015 or abs(ty - py) / yr > 0.015:
                ax.annotate("", xy=(px, py), xytext=(tx, ty),
                            arrowprops=dict(arrowstyle="-",
                                            color="#CCCCCC", lw=0.5),
                            zorder=5)

        ax.set_xlabel("Traditionalism vs Progressivism", fontsize=12)
        ax.set_ylabel("Universalism vs Particularism", fontsize=12)
        ax.tick_params(length=0, labelbottom=False, labelleft=False)

        shape_handles = build_family_shape_legend_handles(models, marker_color="#333333")
        origin_handles = build_origin_legend_handles(models, marker_color="#555555")

        leg_shape = ax.legend(handles=shape_handles,
                      loc="upper right", bbox_to_anchor=(0.985, 0.985),
                      fontsize=7.4, frameon=True, fancybox=False,
                      facecolor="white", framealpha=0.95,
                      edgecolor="#CCCCCC",
                      title="LLM family → symbol", title_fontsize=9)
        leg_shape.get_title().set_fontweight("bold")
        ax.add_artist(leg_shape)

        leg_origin = ax.legend(handles=origin_handles,
                       loc="lower right", bbox_to_anchor=(0.985, 0.05),
                               fontsize=8.3, frameon=True, fancybox=False,
                               facecolor="white", framealpha=0.95,
                               edgecolor="#CCCCCC",
                               title="LLM origin → outline colour", title_fontsize=9)
        leg_origin.get_title().set_fontweight("bold")
        ax.add_artist(leg_origin)

        fig.tight_layout()

        if output_path:
            out_path = Path(output_path)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
            fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
            print(f"PCA plot saved to: {out_path}")
            print(f"PCA plot saved to: {out_path.with_suffix('.pdf')}")
            print(f"PCA plot saved to: {out_path.with_suffix('.svg')}")
        plt.close(fig)


# ── Plot 1b: PCA — one coloured centroid per model + scatter per country ──────

def plot_pca_by_model(human_bg: pd.DataFrame,
                      llm_proj: pd.DataFrame,
                      pca,
                      code_to_family: dict,
                      output_path=None,
                      column_width: float = 3.5):
    """
    Alternative PCA plot.

    One point per (model, language-family) — the mean PCA projection of that
    model across all countries belonging to that family.

    Encoding:
      - Colour  → language family
      - Marker  → LLM model
      - Label   → "{model}" (short) placed next to each point
    Human country-language points are shown as small faded circles (colour = family)
    for spatial context.
    
    Args:
        column_width: Figure width in inches (default 3.5 for single column, use 7.0 for two columns)
    """
    # Attach language family to each (model, country) row
    llm_proj = llm_proj.copy()
    llm_proj["family"] = llm_proj["country_code"].map(code_to_family).fillna("Unknown")
    llm_proj = llm_proj[llm_proj["family"] != "Unknown"]

    # One centroid per (model, family)
    fam_centroids = (
        llm_proj.groupby(["model", "family"])[["PC1", "PC2"]]
        .mean()
        .reset_index()
    )

    models       = sorted(fam_centroids["model"].unique())
    all_fams     = sorted(fam_centroids["family"].unique())

    # Families drive colour; model families drive marker shape
    fam_colors   = {f: PALETTE[i % len(PALETTE)] for i, f in enumerate(all_fams)}
    llm_family_marker, model_display_family, _family_counts = build_compact_family_mapping(models)
    model_marker = {m: llm_family_marker[model_display_family[m]] for m in models}

    # Human background: same family colour mapping
    human_fam_colors = {f: PALETTE[i % len(PALETTE)]
                        for i, f in enumerate(sorted(human_bg["family"].dropna().unique().tolist()))}

    with plt.style.context(["science", "no-latex"]):
        mpl.rcParams.update({
            "font.size": 11, "axes.labelsize": 12,
            "xtick.labelsize": 10, "ytick.labelsize": 10,
            "legend.fontsize": 8.5,
        })

        # Map column_width to optimized figure dimensions
        # 1col: 5.8x4.1 (singlecol), 2col: 8.5x5.5 (improved readability)
        figsize_map = {3.5: (5.8, 4.1), 7.0: (8.5, 5.5)}
        figsize = figsize_map.get(column_width, (column_width, column_width * 0.65))
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axhline(0, color="#888888", lw=0.7, alpha=0.4)
        ax.axvline(0, color="#888888", lw=0.7, alpha=0.4)

        # ── Human country background (faded) ──────────────────────────────────
        for _, r in human_bg.iterrows():
            x, y = r["PC1"], r["PC2"]
            country, fam = r["country"], r["family"]
            base_col = human_fam_colors.get(fam, "#AAAAAA")
            col = _lighten_color(base_col, amount=BG_LIGHTEN_AMOUNT)
            ax.scatter(x, y, s=22, color=col, marker="o",
                       edgecolors="white", linewidth=0.3, alpha=BG_POINT_ALPHA, zorder=2)
            label = str(r.get("label", country))
            ax.text(x, y, label, fontsize=5.5, color="#CCCCCC",
                    ha="center", va="bottom", zorder=3)

        # ── (model, family) centroids ─────────────────────────────────────────
        pt_texts, pt_xy, pt_cols = [], [], []

        for _, row in fam_centroids.iterrows():
            model = row["model"]
            fam   = row["family"]
            cx, cy = row["PC1"], row["PC2"]
            col = fam_colors[fam]
            mk  = model_marker[model]

            origin = infer_model_origin(model)
            draw_model_point_with_origin_outline(
                ax=ax,
                x=float(cx),
                y=float(cy),
                marker=mk,
                size=100,
                origin=origin,
                linewidth=1.8,
                alpha=0.92,
                zorder=6,
            )

            # Short model name for label (strip common suffixes to save space)
            short = model.replace("-responses", "")
            t = ax.text(cx, cy, short, fontsize=6.5, color=col,
                        fontweight="medium", zorder=7)
            pt_texts.append(t)
            pt_xy.append((cx, cy))
            pt_cols.append(col)

        with contextlib.redirect_stdout(io.StringIO()):
            adjust_text(pt_texts, ax=ax,
                        expand=(1.4, 1.6),
                        force_text=(0.4, 0.6),
                        force_points=(0.15, 0.25),
                        ensure_inside_axes=False)

        xr = ax.get_xlim()[1] - ax.get_xlim()[0]
        yr = ax.get_ylim()[1] - ax.get_ylim()[0]
        for t, (px, py), col in zip(pt_texts, pt_xy, pt_cols):
            tx, ty = t.get_position()
            if abs(tx - px) / xr > 0.015 or abs(ty - py) / yr > 0.015:
                ax.annotate("", xy=(px, py), xytext=(tx, ty),
                            arrowprops=dict(arrowstyle="-", color=col,
                                            lw=0.45, alpha=0.45),
                            zorder=5)

        ax.set_xlabel("Traditionalism vs Progressivism", fontsize=12)
        ax.set_ylabel("Universalism vs Particularism", fontsize=12)
        ax.tick_params(length=0, labelbottom=False, labelleft=False)

        # Legend: model family → marker
        model_handles = build_family_shape_legend_handles(models, marker_color="#444444")
        origin_handles = build_origin_legend_handles(models, marker_color="#666666")
        leg2 = ax.legend(handles=model_handles,
                         loc="upper right", bbox_to_anchor=(0.985, 0.985),
                         fontsize=7.6, frameon=True, fancybox=False,
                         facecolor="white", framealpha=0.95,
                         edgecolor="#CCCCCC",
                         title="LLM family (shape)", title_fontsize=9)
        leg2.get_title().set_fontweight("bold")
        ax.add_artist(leg2)

        leg3 = ax.legend(handles=origin_handles,
                         loc="lower right", bbox_to_anchor=(0.985, 0.05),
                         fontsize=8.5, frameon=True, fancybox=False,
                         facecolor="white", framealpha=0.95,
                         edgecolor="#CCCCCC",
                         title="LLM origin (outline colour)", title_fontsize=9)
        leg3.get_title().set_fontweight("bold")

        fig.tight_layout()
        if output_path:
            out_path = Path(output_path)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
            print(f"PCA (by-model) plot saved to: {out_path}")
            print(f"PCA (by-model) plot saved to: {out_path.with_suffix('.pdf')}")
        plt.close(fig)


def plot_pca_weighted_family_overlay(
    human_bg: pd.DataFrame,
    llm_proj: pd.DataFrame,
    llm_country_weights: pd.DataFrame,
    pca,
    code_to_family: dict,
    output_path=None,
    compact: bool = False,
    compact_show_split_labels: bool = True,
    column_width: float = None,
):
    """
    New PCA view:
      - Background: human country-language points (same as human PCA figure)
      - Foreground: one weighted LLM point per language family
    
    Args:
        column_width: Figure width in inches. If provided, overrides compact sizing.
                     Default 3.5 for single column, use 7.0 for two columns.
    """
    if llm_proj.empty or human_bg.empty:
        print("Insufficient data for weighted-family PCA overlay — skipping")
        return

    work = llm_proj.copy()
    work["country_code"] = work["country_code"].astype(str).str.lower()
    work["family"] = work["country_code"].map(code_to_family).fillna("Unknown")

    w = llm_country_weights.copy()
    w["_country_code"] = w["_country_code"].astype(str).str.lower()

    work = work.merge(
        w,
        left_on=["model", "country_code"],
        right_on=["_model", "_country_code"],
        how="left",
    )
    work["weight"] = pd.to_numeric(work.get("weight", 1.0), errors="coerce").fillna(1.0)
    work = work[work["family"] != "Unknown"].copy()

    rows = []
    for fam, gf in work.groupby("family"):
        ww = gf["weight"].to_numpy(dtype=float)
        if ww.sum() <= 0:
            continue
        pc1 = float(np.average(gf["PC1"].to_numpy(dtype=float), weights=ww))
        pc2 = float(np.average(gf["PC2"].to_numpy(dtype=float), weights=ww))
        rows.append({
            "family": fam,
            "PC1": pc1,
            "PC2": pc2,
            "weight_sum": float(ww.sum()),
            "n_points": int(len(gf)),
        })

    fam_pts = pd.DataFrame(rows)
    if fam_pts.empty:
        print("No weighted family points available — skipping")
        return

    # Human family centroids in PCA space (weighted mean by language family;
    # respondent counts are used as weights when available).
    human_rows = []
    for fam, gf in human_bg.groupby("family"):
        ww = pd.to_numeric(gf.get("n_respondents", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=float)
        if ww.sum() <= 0:
            ww = np.ones(len(gf), dtype=float)
        human_rows.append({
            "family": fam,
            "PC1": float(np.average(gf["PC1"].to_numpy(dtype=float), weights=ww)),
            "PC2": float(np.average(gf["PC2"].to_numpy(dtype=float), weights=ww)),
            "n_points": int(len(gf)),
        })
    human_fam_pts = pd.DataFrame(human_rows)

    with plt.style.context(["science", "no-latex"]):
        mpl.rcParams.update({
            "font.size": 10 if compact else 11,
            "axes.labelsize": 10.5 if compact else 12,
            "xtick.labelsize": 8.5 if compact else 10,
            "ytick.labelsize": 8.5 if compact else 10,
            "legend.fontsize": 7.8 if compact else 9,
        })

        # Use column_width if provided, otherwise fall back to compact sizing
        if column_width is None:
            figsize = (5.8, 4.1) if compact else (11.5, 7.5)
        else:
            figsize = (column_width, column_width * 0.65)
        
        # Map column_width to optimized figure dimensions
        # 1col: 5.8x4.1 (singlecol), 2col: 8.5x5.5 (improved readability)
        figsize_map = {3.5: (5.8, 4.1), 7.0: (8.5, 5.5)}
        figsize = figsize_map.get(column_width, figsize)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axhline(0, color="#888888", lw=0.7, alpha=0.45)
        ax.axvline(0, color="#888888", lw=0.7, alpha=0.45)

        fams_bg = sorted(human_bg["family"].dropna().unique().tolist())
        fam_colors = {f: PALETTE[i % len(PALETTE)] for i, f in enumerate(fams_bg)}

        # Weighted LLM family points
        texts = []
        llm_family_xy = {}
        for _, r in fam_pts.iterrows():
            fam = r["family"]
            col = fam_colors.get(fam, "#444444")
            ax.scatter(r["PC1"], r["PC2"], s=122 if compact else 180, marker="D", color=col,
                       edgecolors="#000000", linewidth=1.1, alpha=0.95, zorder=7)
            llm_family_xy[fam] = (float(r["PC1"]), float(r["PC2"]))
            t = ax.text(r["PC1"], r["PC2"], fam, fontsize=8 if compact else 9, color="#111111",
                        fontweight="bold", zorder=8)
            texts.append(t)

        # Human family centroids (same family colours, different symbol)
        for _, r in human_fam_pts.iterrows():
            fam = r["family"]
            col = fam_colors.get(fam, "#444444")
            ax.scatter(r["PC1"], r["PC2"], s=95 if compact else 135, marker="o",
                       facecolors=col, edgecolors="#000000", linewidth=1.1,
                       alpha=0.98, zorder=6)

            # Optional visual linkage between human and LLM family centroids
            if fam in llm_family_xy:
                x_llm, y_llm = llm_family_xy[fam]
                ax.plot([r["PC1"], x_llm], [r["PC2"], y_llm],
                        color="#000000", lw=0.9, alpha=0.5, linestyle=":", zorder=5)

        with contextlib.redirect_stdout(io.StringIO()):
            adjust_text(texts, ax=ax,
                        expand=(1.25, 1.45),
                        force_text=(0.35, 0.55),
                        force_points=(0.1, 0.2),
                        ensure_inside_axes=False)

        if compact:
            ax.set_xlabel("Traditionalism vs Progressivism", fontsize=10.5)
            ax.set_ylabel("Universalism vs Particularism", fontsize=10.5)
        else:
            ax.set_xlabel("Traditionalism vs Progressivism", fontsize=12)
            ax.set_ylabel("Universalism vs Particularism", fontsize=12)
        ax.tick_params(length=0, labelbottom=False, labelleft=False)

        fam_handles = [
            mpl.lines.Line2D([0], [0], marker="o", color="w",
                             markerfacecolor=fam_colors[f], markeredgecolor="white",
                             markersize=7.5, label=f)
            for f in sorted(fam_pts["family"].tolist())
        ]
        llm_handle = mpl.lines.Line2D([0], [0], marker="D", color="w",
                          markerfacecolor="#999999", markeredgecolor="#000000",
                                  markersize=7.5, label="LLM centroid")
        human_centroid_handle = mpl.lines.Line2D([0], [0], marker="o", color="w",
                             markerfacecolor="#999999", markeredgecolor="#000000",
                             markeredgewidth=1.0,
                                      markersize=7.5, label="Human centroid")
        if compact:
            leg_symbol = ax.legend(handles=[llm_handle, human_centroid_handle],
                                   loc="upper center", bbox_to_anchor=(0.5, -0.10),
                                   ncol=2,
                                   frameon=True, fancybox=False, edgecolor="#CCCCCC",
                                   title="Symbol (source)", title_fontsize=8)
            leg_symbol.get_title().set_fontweight("bold")
            ax.add_artist(leg_symbol)

            leg_colour = ax.legend(handles=fam_handles,
                                   loc="upper center", bbox_to_anchor=(0.5, -0.24),
                                   ncol=3,
                                   frameon=True, fancybox=False, edgecolor="#CCCCCC",
                                   title="Colour (language family)", title_fontsize=8)
            leg_colour.get_title().set_fontweight("bold")
        else:
            leg_symbol = ax.legend(handles=[llm_handle, human_centroid_handle],
                                   loc="upper left", bbox_to_anchor=(1.02, 1.0),
                                   frameon=True, fancybox=False, edgecolor="#CCCCCC",
                                   title="Symbol (source)", title_fontsize=9)
            leg_symbol.get_title().set_fontweight("bold")
            ax.add_artist(leg_symbol)

            leg_colour = ax.legend(handles=fam_handles,
                                   loc="upper left", bbox_to_anchor=(1.02, 0.56),
                                   frameon=True, fancybox=False, edgecolor="#CCCCCC",
                                   title="Colour (language family)", title_fontsize=9)
            leg_colour.get_title().set_fontweight("bold")

        # Keep sufficient canvas margin for external legends
        if compact:
            fig.subplots_adjust(bottom=0.32)
        else:
            fig.subplots_adjust(right=0.78)

        legend_artists = [leg_symbol, leg_colour]
        if output_path:
            out_path = Path(output_path)
            fig.savefig(
                out_path,
                dpi=300,
                bbox_inches="tight",
                bbox_extra_artists=legend_artists,
                pad_inches=0.08,
            )
            fig.savefig(
                out_path.with_suffix(".pdf"),
                bbox_inches="tight",
                bbox_extra_artists=legend_artists,
                pad_inches=0.08,
            )
            print(f"PCA weighted-family overlay plot saved to: {out_path}")
            print(f"PCA weighted-family overlay plot saved to: {out_path.with_suffix('.pdf')}")
        plt.close(fig)


def plot_pca_english_vs_non_english_mean(llm_proj: pd.DataFrame,
                                         pca,
                                         output_path=None,
                                         english_code: str = "gb",
                                         column_width: float = 3.5):
    """
    PCA paired comparison by model:
      - English point:          (model, country_code == english_code)
      - Non-English mean point: average of all other country codes for model

    For each model, a connector links the non-English mean to the English point.
    
    Args:
        column_width: Figure width in inches (default 3.5 for single column, use 7.0 for two columns)
    """
    if llm_proj.empty:
        print("No projected LLM points available — skipping English vs non-English PCA plot.")
        return

    work = llm_proj.copy()
    work["country_code"] = work["country_code"].astype(str).str.lower()

    rows = []
    for model, grp in work.groupby("model"):
        en = grp[grp["country_code"] == english_code]
        non_en = grp[grp["country_code"] != english_code]

        if en.empty or non_en.empty:
            continue

        en_point = en[["PC1", "PC2"]].mean()
        non_en_point = non_en[["PC1", "PC2"]].mean()
        rows.append({
            "model": model,
            "en_PC1": float(en_point["PC1"]),
            "en_PC2": float(en_point["PC2"]),
            "non_en_PC1": float(non_en_point["PC1"]),
            "non_en_PC2": float(non_en_point["PC2"]),
            "delta_PC1": float(en_point["PC1"] - non_en_point["PC1"]),
            "delta_PC2": float(en_point["PC2"] - non_en_point["PC2"]),
        })

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        print("No model has both English and non-English projections — skipping plot.")
        return

    with plt.style.context(["science", "no-latex"]):
        mpl.rcParams.update({
            "font.size": 11, "axes.labelsize": 12,
            "xtick.labelsize": 10, "ytick.labelsize": 10,
            "legend.fontsize": 8.5,
        })

        # Map column_width to optimized figure dimensions
        # 1col: 5.8x4.1 (singlecol), 2col: 8.5x5.5 (improved readability)
        figsize_map = {3.5: (5.8, 4.1), 7.0: (8.5, 5.5)}
        figsize = figsize_map.get(column_width, (column_width, column_width * 0.65))
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axhline(0, color="#888888", lw=0.7, alpha=0.5)
        ax.axvline(0, color="#888888", lw=0.7, alpha=0.5)

        texts = []
        models = sorted(pairs["model"].unique())
        color_map = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}
        family_marker, model_display_family, _family_counts = build_compact_family_mapping(models)

        for _, row in pairs.iterrows():
            model = row["model"]
            col = color_map[model]
            mk = family_marker[model_display_family[model]]
            origin = infer_model_origin(model)
            ls = ORIGIN_LINESTYLE[origin]

            # Connector: non-English mean -> English
            ax.plot([row["non_en_PC1"], row["en_PC1"]],
                    [row["non_en_PC2"], row["en_PC2"]],
                color=col, lw=1.2, alpha=0.8, linestyle=ls, zorder=3)

            # Non-English mean
            draw_model_point_with_origin_outline(
                ax=ax,
                x=float(row["non_en_PC1"]),
                y=float(row["non_en_PC2"]),
                marker=mk,
                size=85,
                origin=origin,
                linewidth=1.7,
                alpha=0.92,
                zorder=4,
            )

            # English
            draw_model_point_with_origin_outline(
                ax=ax,
                x=float(row["en_PC1"]),
                y=float(row["en_PC2"]),
                marker=mk,
                size=95,
                origin=origin,
                linewidth=1.9,
                alpha=0.97,
                zorder=5,
            )

            # Label near English point
            t = ax.text(row["en_PC1"], row["en_PC2"], model,
                        fontsize=7.2, color="#111111", zorder=6)
            texts.append(t)

        with contextlib.redirect_stdout(io.StringIO()):
            adjust_text(texts, ax=ax,
                        expand=(1.3, 1.5),
                        force_text=(0.35, 0.55),
                        force_points=(0.1, 0.2),
                        ensure_inside_axes=False)

        marker_handles = [
            mpl.lines.Line2D([0], [0], marker="o", color="w",
                             markerfacecolor="#666666", markeredgecolor="#222222",
                             markersize=7, label="English (gb)"),
            mpl.lines.Line2D([0], [0], marker="o", color="w",
                             markerfacecolor="#666666", markeredgecolor="#666666",
                             markersize=7, label="Mean non-English"),
            mpl.lines.Line2D([0], [0], color="#666666", lw=1.2,
                             label="Shift: non-English → English"),
        ]
        shape_handles = build_family_shape_legend_handles(models, marker_color="#444444")
        origin_handles = build_origin_legend_handles(models, marker_color="#666666")

        leg1 = ax.legend(handles=shape_handles,
                 loc="upper right", bbox_to_anchor=(0.985, 0.985),
                 frameon=True, fancybox=False, edgecolor="#CCCCCC",
                 facecolor="white", framealpha=0.95,
                 title="LLM family → symbol", title_fontsize=9)
        leg1.get_title().set_fontweight("bold")
        ax.add_artist(leg1)

        leg2 = ax.legend(handles=origin_handles,
                         loc="lower right", bbox_to_anchor=(0.985, 0.05),
                 frameon=True, fancybox=False, edgecolor="#CCCCCC",
                 facecolor="white", framealpha=0.95,
                 title="LLM origin → outline colour", title_fontsize=9)
        leg2.get_title().set_fontweight("bold")
        ax.add_artist(leg2)

        leg3 = ax.legend(handles=marker_handles,
                         loc="upper right", bbox_to_anchor=(0.985, 0.42),
                 frameon=True, fancybox=False, edgecolor="#CCCCCC",
                 facecolor="white", framealpha=0.95,
                 title="Point meaning", title_fontsize=9)
        leg3.get_title().set_fontweight("bold")

        ax.set_xlabel("Traditionalism vs Progressivism", fontsize=12)
        ax.set_ylabel("Universalism vs Particularism", fontsize=12)
        ax.tick_params(length=0, labelbottom=False, labelleft=False)

        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"PCA English-vs-rest plot saved to: {output_path}")
        plt.close(fig)


def plot_pca_human_vs_llm_english_multilingual(
    human_country_ref: pd.DataFrame,
    llm_proj: pd.DataFrame,
    pca,
    output_path=None,
    english_code: str = "gb",
    column_width: float = 3.5,
):
    """
    PCA comparison with four conceptual points:
      - Human multilingual mean (all human countries except English)
      - Human English
      - LLM English (per model)
      - LLM multilingual mean (per model, all non-English countries)

    Human points are fixed reference anchors; each model contributes its two LLM
    points plus connectors showing its displacement relative to the human frame.
    
    Args:
        column_width: Figure width in inches (default 3.5 for single column, use 7.0 for two columns)
    """
    if llm_proj.empty or human_country_ref.empty:
        print("Insufficient PCA inputs — skipping human-vs-LLM English/multilingual plot.")
        return

    human_df = human_country_ref.rename(columns={"country": "country_name"}).copy()
    human_df["_country_norm"] = human_df["country_name"].astype(str).str.lower().str.strip()
    human_english_mask = human_df["_country_norm"].isin({
        "great britain", "gt britain", "britain", "united kingdom", "gb"
    }) | human_df["_country_norm"].str.contains("britain", regex=False)

    human_en = human_df[human_english_mask]
    human_non_en = human_df[~human_english_mask]
    if human_en.empty or human_non_en.empty:
        print("Could not derive both human English and human multilingual references — skipping plot.")
        return

    human_en_point = human_en[["PC1", "PC2"]].mean()
    human_non_en_point = human_non_en[["PC1", "PC2"]].mean()

    work = llm_proj.copy()
    work["country_code"] = work["country_code"].astype(str).str.lower()

    rows = []
    for model, grp in work.groupby("model"):
        en = grp[grp["country_code"] == english_code]
        non_en = grp[grp["country_code"] != english_code]
        if en.empty or non_en.empty:
            continue

        en_point = en[["PC1", "PC2"]].mean()
        non_en_point = non_en[["PC1", "PC2"]].mean()
        rows.append({
            "model": model,
            "llm_en_PC1": float(en_point["PC1"]),
            "llm_en_PC2": float(en_point["PC2"]),
            "llm_non_en_PC1": float(non_en_point["PC1"]),
            "llm_non_en_PC2": float(non_en_point["PC2"]),
        })

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        print("No model has both English and multilingual LLM projections — skipping plot.")
        return

    with plt.style.context(["science", "no-latex"]):
        mpl.rcParams.update({
            "font.size": 11, "axes.labelsize": 12,
            "xtick.labelsize": 10, "ytick.labelsize": 10,
            "legend.fontsize": 8.5,
        })

        # Map column_width to optimized figure dimensions
        # 1col: 5.8x4.1 (singlecol), 2col: 8.5x5.5 (improved readability)
        figsize_map = {3.5: (5.8, 4.1), 7.0: (8.5, 5.5)}
        figsize = figsize_map.get(column_width, (column_width, column_width * 0.65))
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axhline(0, color="#888888", lw=0.7, alpha=0.5)
        ax.axvline(0, color="#888888", lw=0.7, alpha=0.5)

        # Human reference anchors
        ax.plot([human_non_en_point["PC1"], human_en_point["PC1"]],
                [human_non_en_point["PC2"], human_en_point["PC2"]],
                color="#222222", lw=1.4, alpha=0.55, linestyle="--", zorder=2)
        ax.scatter(human_non_en_point["PC1"], human_non_en_point["PC2"],
                   s=150, marker="s", color="#FFFFFF", edgecolors="#222222",
                   linewidth=1.1, zorder=6)
        ax.scatter(human_en_point["PC1"], human_en_point["PC2"],
                   s=170, marker="*", color="#222222", edgecolors="#222222",
                   linewidth=0.8, zorder=7)

        ax.text(human_non_en_point["PC1"], human_non_en_point["PC2"],
                "Human multilingual", fontsize=8.2, color="#222222", zorder=8)
        ax.text(human_en_point["PC1"], human_en_point["PC2"],
                "Human English", fontsize=8.2, color="#222222", zorder=8)

        texts = []
        models = sorted(pairs["model"].unique())
        color_map = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}
        family_marker, model_display_family, _family_counts = build_compact_family_mapping(models)

        for _, row in pairs.iterrows():
            model = row["model"]
            col = color_map[model]
            mk = family_marker[model_display_family[model]]
            origin = infer_model_origin(model)
            ls = ORIGIN_LINESTYLE[origin]

            # LLM internal shift: multilingual -> English
            ax.plot([row["llm_non_en_PC1"], row["llm_en_PC1"]],
                    [row["llm_non_en_PC2"], row["llm_en_PC2"]],
                    color=col, lw=1.2, alpha=0.85, linestyle=ls, zorder=3)

            # Human-to-LLM connectors for each pole
            ax.plot([human_non_en_point["PC1"], row["llm_non_en_PC1"]],
                    [human_non_en_point["PC2"], row["llm_non_en_PC2"]],
                    color=col, lw=0.9, alpha=0.35, linestyle=":", zorder=2)
            ax.plot([human_en_point["PC1"], row["llm_en_PC1"]],
                    [human_en_point["PC2"], row["llm_en_PC2"]],
                    color=col, lw=0.9, alpha=0.35, linestyle=":", zorder=2)

            # LLM multilingual mean
            draw_model_point_with_origin_outline(
                ax=ax,
                x=float(row["llm_non_en_PC1"]),
                y=float(row["llm_non_en_PC2"]),
                marker=mk,
                size=88,
                origin=origin,
                linewidth=1.7,
                alpha=0.93,
                zorder=4,
                )

            # LLM English
            draw_model_point_with_origin_outline(
                ax=ax,
                x=float(row["llm_en_PC1"]),
                y=float(row["llm_en_PC2"]),
                marker=mk,
                size=98,
                origin=origin,
                linewidth=1.9,
                alpha=0.96,
                zorder=5,
                )

            t = ax.text(row["llm_en_PC1"], row["llm_en_PC2"], model,
                        fontsize=7.0, color="#111111", zorder=9)
            texts.append(t)

        with contextlib.redirect_stdout(io.StringIO()):
            adjust_text(texts, ax=ax,
                        expand=(1.3, 1.5),
                        force_text=(0.35, 0.55),
                        force_points=(0.1, 0.2),
                        ensure_inside_axes=False)

        marker_handles = [
            mpl.lines.Line2D([0], [0], marker="s", color="w",
                             markerfacecolor="#FFFFFF", markeredgecolor="#222222",
                             markersize=7, label="Human multilingual"),
            mpl.lines.Line2D([0], [0], marker="*", color="w",
                             markerfacecolor="#222222", markeredgecolor="#222222",
                             markersize=9, label="Human English"),
            mpl.lines.Line2D([0], [0], marker="o", color="w",
                             markerfacecolor="#666666", markeredgecolor="#666666",
                             markersize=7, label="LLM multilingual"),
            mpl.lines.Line2D([0], [0], marker="o", color="w",
                             markerfacecolor="#666666", markeredgecolor="#222222",
                             markersize=7, label="LLM English"),
            mpl.lines.Line2D([0], [0], color="#666666", lw=1.2,
                             label="LLM shift: multilingual → English"),
            mpl.lines.Line2D([0], [0], color="#666666", lw=0.9, linestyle=":",
                             label="Human ↔ LLM reference gap"),
        ]
        shape_handles = build_family_shape_legend_handles(models, marker_color="#444444")
        origin_handles = build_origin_legend_handles(models, marker_color="#666666")

        leg1 = ax.legend(handles=shape_handles,
                 loc="upper right", bbox_to_anchor=(0.985, 0.985),
                 frameon=True, fancybox=False, edgecolor="#CCCCCC",
                 facecolor="white", framealpha=0.95,
                 title="LLM family → symbol", title_fontsize=9)
        leg1.get_title().set_fontweight("bold")
        ax.add_artist(leg1)

        leg2 = ax.legend(handles=origin_handles,
                         loc="lower right", bbox_to_anchor=(0.985, 0.05),
                 frameon=True, fancybox=False, edgecolor="#CCCCCC",
                 facecolor="white", framealpha=0.95,
                 title="LLM origin → outline colour", title_fontsize=9)
        leg2.get_title().set_fontweight("bold")
        ax.add_artist(leg2)

        leg3 = ax.legend(handles=marker_handles,
                         loc="upper right", bbox_to_anchor=(0.985, 0.42),
                 frameon=True, fancybox=False, edgecolor="#CCCCCC",
                 facecolor="white", framealpha=0.95,
                 title="Point meaning", title_fontsize=9)
        leg3.get_title().set_fontweight("bold")

        ax.set_xlabel("Traditionalism vs Progressivism", fontsize=12)
        ax.set_ylabel("Universalism vs Particularism", fontsize=12)
        ax.tick_params(length=0, labelbottom=False, labelleft=False)

        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"PCA human-vs-LLM English/multilingual plot saved to: {output_path}")
        plt.close(fig)


# ── Plot 2: Radar chart ───────────────────────────────────────────────────────

def _category_averages(df_norm, variables, var_to_cat, cat_order=None):
    """
    Returns a dict {category: mean_normalised_score} for the given rows/variables.
    Only uses variables present in df_norm.
    """
    cat_vals: dict = {}
    for var in variables:
        cat = var_to_cat.get(var)
        if cat is None or cat == "Unknown":
            continue
        col_vals = df_norm[var].dropna() if var in df_norm.columns else pd.Series(dtype=float)
        if len(col_vals):
            cat_vals.setdefault(cat, []).extend(col_vals.tolist())
    return {c: float(np.mean(v)) for c, v in cat_vals.items()}


def _format_topic_label(label: str) -> str:
    """Convert topic labels to title-style capitalization.

    Examples:
      - "National identity"   -> "National Identity"
      - "Religion and morale" -> "Religion and Morale"
    """
    if label is None:
        return ""
    stop_words = {"and", "or", "of", "the", "in", "on", "to", "for", "vs"}
    words = [w for w in re.split(r"\s+", str(label).strip()) if w]
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        if i > 0 and lw in stop_words:
            out.append(lw)
        else:
            out.append(lw[:1].upper() + lw[1:])
    return " ".join(out)


def _build_family_style_map(families: list[str]) -> dict[str, dict[str, str]]:
    """Return deterministic visual styles for each language family."""
    fam_sorted = sorted(families)
    return {
        fam: {
            "color": PALETTE[i % len(PALETTE)],
            "marker": MARKER_STYLES[i % len(MARKER_STYLES)],
            "linestyle": "-",
        }
        for i, fam in enumerate(fam_sorted)
    }


def _radar_values_from_var_dict(avg_dict: dict,
                                var_to_cat: dict,
                                all_cats: list[str]) -> list[float]:
    """Build closed radar vector from per-variable averages."""
    vals = []
    for cat in all_cats:
        cat_vars = [v for v, c in var_to_cat.items() if c == cat and v in avg_dict]
        if cat_vars:
            vals.append(float(np.mean([avg_dict[v] for v in cat_vars])))
        else:
            vals.append(0.0)
    return vals + [vals[0]]


def _style_single_radar_axis(ax, angles: list[float], labels: list[str]):
    """Shared axis style for single radar figures."""
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8.2, fontweight="medium")
    ax.tick_params(axis="x", pad=5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=7.3, color="#555555")
    ax.grid(True, linestyle="-", alpha=0.20, color="#888888", linewidth=0.5)
    ax.spines["polar"].set_color("#CCCCCC")
    ax.spines["polar"].set_linewidth(0.7)


def _wrap_topic_label(label: str, width: int = 14) -> str:
    """Wrap long topic labels to keep strict single-column readability."""
    txt = str(label).strip()
    if not txt:
        return txt
    return "\n".join(textwrap.wrap(txt, width=width, break_long_words=False))


def plot_human_topics_radar(df_human_norm: pd.DataFrame,
                            topic_df: pd.DataFrame,
                            country_col: str,
                            c2f: dict,
                            v_cols: list,
                            output_path=None):
    """Human topic profiles by language family (single radar, no title text)."""
    var_to_cat = dict(zip(topic_df["variable_name"], topic_df["category"]))
    all_cats = sorted([c for c in topic_df["category"].unique() if c != "Unknown"])
    if not all_cats:
        print("No valid categories — skipping human topics radar chart.")
        return

    labels = [_wrap_topic_label(_format_topic_label(c), width=14) for c in all_cats]
    n = len(all_cats)
    angles = [k / n * 2 * np.pi for k in range(n)] + [0]

    families = sorted(set(c2f.values()))
    style_map = _build_family_style_map(families)

    human_avgs: dict = {}
    for fam in families:
        fam_countries = [c for c, f in c2f.items() if f == fam]
        sub = df_human_norm[df_human_norm[country_col].isin(fam_countries)]
        avg = {}
        for var in v_cols:
            if var in sub.columns:
                val = float(sub[var].mean())
                if not np.isnan(val):
                    avg[var] = val
        human_avgs[fam] = avg

    with plt.style.context(["science", "no-latex"]):
        mpl.rcParams.update({
            "font.size": 8.7,
            "axes.labelsize": 8.9,
            "legend.fontsize": 7.8,
        })
        fig, ax = plt.subplots(figsize=(3.35, 4.15), subplot_kw=dict(polar=True), facecolor="white")
        ax.set_facecolor("white")
        _style_single_radar_axis(ax, angles, labels)

        handles = []
        all_vals_for_stats = []
        for fam in families:
            vals = _radar_values_from_var_dict(human_avgs[fam], var_to_cat, all_cats)
            all_vals_for_stats.extend(vals[:-1])
            sty = style_map[fam]
            line, = ax.plot(
                angles,
                vals,
                linestyle=sty["linestyle"],
                linewidth=1.15,
                color=sty["color"],
                marker=sty["marker"],
                markersize=3.8,
                markerfacecolor="white",
                markeredgewidth=0.9,
                label=fam,
                alpha=0.92,
            )
            ax.fill(angles, vals, alpha=0.06, color=sty["color"])
            handles.append(line)

        if all_vals_for_stats:
            mean_score = float(np.mean(all_vals_for_stats))
            std_score = float(np.std(all_vals_for_stats))
            ax.text(
                0.02,
                1.12,
                f"Average ± Std: {mean_score:.3f} ± {std_score:.3f}",
                transform=ax.transAxes,
                fontsize=7.4,
                color="#222222",
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#CCCCCC", alpha=0.95),
            )

        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=min(3, max(2, len(handles))),
            frameon=True,
            fancybox=False,
            edgecolor="#CCCCCC",
            facecolor="white",
            fontsize=7.4,
            handlelength=1.25,
            columnspacing=0.8,
            borderpad=0.45,
        )

        fig.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.22)

        if output_path:
            out_path = Path(output_path)
            fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
            fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", edgecolor="none")
            print(f"Human topics radar saved to: {out_path}")
            print(f"Human topics radar saved to: {out_path.with_suffix('.pdf')}")
        plt.close(fig)


def plot_llm_topics_radar_weighted_by_family(llm_df: pd.DataFrame,
                                             topic_df: pd.DataFrame,
                                             code_to_family: dict,
                                             output_path=None):
    """LLM topic profiles by language family using weighted means."""
    var_to_cat = dict(zip(topic_df["variable_name"], topic_df["category"]))
    all_cats = sorted([c for c in topic_df["category"].unique() if c != "Unknown"])
    if not all_cats:
        print("No valid categories — skipping LLM topics radar chart.")
        return

    labels = [_wrap_topic_label(_format_topic_label(c), width=14) for c in all_cats]
    n = len(all_cats)
    angles = [k / n * 2 * np.pi for k in range(n)] + [0]

    llm_work = llm_df.copy()
    llm_work["_country_code"] = llm_work["_country_code"].astype(str).str.lower()
    llm_work["language_family"] = llm_work["_country_code"].map(code_to_family).fillna("Unknown")
    llm_work = llm_work[llm_work["language_family"] != "Unknown"].copy()
    llm_work = llm_work[llm_work["norm_response"].notna()].copy()

    if llm_work.empty:
        print("No valid LLM rows after family mapping — skipping LLM topics radar chart.")
        return

    # Weighted mean aggregation per family and variable:
    # 1) mean response per (family, model, country, variable)
    # 2) weight each cell by number of valid responses in that cell
    llm_cells = (
        llm_work
        .groupby(["language_family", "_model", "_country_code", "variable"])["norm_response"]
        .agg(cell_mean="mean", weight="size")
        .reset_index()
    )

    llm_cells["weighted_value"] = llm_cells["cell_mean"] * llm_cells["weight"]
    fam_var_weighted = (
        llm_cells
        .groupby(["language_family", "variable"], as_index=False)
        .agg(weighted_sum=("weighted_value", "sum"),
             weight_sum=("weight", "sum"))
    )
    fam_var_weighted["weighted_mean"] = (
        fam_var_weighted["weighted_sum"] / fam_var_weighted["weight_sum"]
    )

    family_avgs: dict[str, dict[str, float]] = {}
    for fam, g in fam_var_weighted.groupby("language_family"):
        family_avgs[str(fam)] = dict(zip(g["variable"], g["weighted_mean"]))

    families = sorted(family_avgs.keys())
    style_map = _build_family_style_map(families)

    with plt.style.context(["science", "no-latex"]):
        mpl.rcParams.update({
            "font.size": 8.7,
            "axes.labelsize": 8.9,
            "legend.fontsize": 7.8,
        })
        fig, ax = plt.subplots(figsize=(3.35, 4.15), subplot_kw=dict(polar=True), facecolor="white")
        ax.set_facecolor("white")
        _style_single_radar_axis(ax, angles, labels)

        handles = []
        all_vals_for_stats = []
        for fam in families:
            vals = _radar_values_from_var_dict(family_avgs[fam], var_to_cat, all_cats)
            all_vals_for_stats.extend(vals[:-1])
            sty = style_map[fam]
            line, = ax.plot(
                angles,
                vals,
                linestyle=sty["linestyle"],
                linewidth=1.15,
                color=sty["color"],
                marker=sty["marker"],
                markersize=3.8,
                markerfacecolor="white",
                markeredgewidth=0.9,
                label=fam,
                alpha=0.92,
            )
            ax.fill(angles, vals, alpha=0.06, color=sty["color"])
            handles.append(line)

        if all_vals_for_stats:
            mean_score = float(np.mean(all_vals_for_stats))
            std_score = float(np.std(all_vals_for_stats))
            ax.text(
                0.02,
                1.12,
                f"Average ± Std: {mean_score:.3f} ± {std_score:.3f}",
                transform=ax.transAxes,
                fontsize=7.4,
                color="#222222",
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#CCCCCC", alpha=0.95),
            )

        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=min(3, max(2, len(handles))),
            frameon=True,
            fancybox=False,
            edgecolor="#CCCCCC",
            facecolor="white",
            fontsize=7.4,
            handlelength=1.25,
            columnspacing=0.8,
            borderpad=0.45,
        )

        fig.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.22)

        if output_path:
            out_path = Path(output_path)
            fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
            fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", edgecolor="none")
            print(f"LLM topics radar saved to: {out_path}")
            print(f"LLM topics radar saved to: {out_path.with_suffix('.pdf')}")
        plt.close(fig)


def plot_llm_radar_topics_only(llm_df: pd.DataFrame,
                               topic_df: pd.DataFrame,
                               v_cols: list,
                               output_path=None):
    """
        Radar chart showing average response rates of LLMs for each topic,
        with one trace per model.

        Visual disambiguation for overlapping profiles
        ----------------------------------------------
        Some models can share near-identical topic means, producing perfectly
        overlapping traces. To keep them distinguishable in print:
            1) each model gets its own style (color + marker + line style), and
            2) traces that are numerically almost identical receive a tiny,
                 deterministic radial offset for display only.
    
    Args:
        llm_df: LLM responses dataframe with norm_response column
        topic_df: Topic matching dataframe
        v_cols: List of variable columns
        output_path: Path to save the chart
    """
    var_to_cat = dict(zip(topic_df["variable_name"], topic_df["category"]))
    all_cats   = sorted([c for c in topic_df["category"].unique() if c != "Unknown"])
    N          = len(all_cats)
    if N == 0:
        print("No valid categories — skipping LLM topics radar chart.")
        return

    angles = [n / N * 2 * np.pi for n in range(N)] + [0]

    llm_plot_df = llm_df.copy()

    # Average per variable for each LLM model (across countries/languages)
    llm_model_avgs: dict = {}
    for model, sub in llm_plot_df.groupby("_model"):
        llm_model_avgs[model] = sub.groupby("variable")["norm_response"].mean().to_dict()

    # Also keep the global aggregate as reference trace
    avg_llm_all = llm_plot_df.groupby("variable")["norm_response"].mean().to_dict()

    def _radar_values(avg_dict):
        vals = []
        for cat in all_cats:
            cat_vars = [v for v, c in var_to_cat.items() if c == cat and v in avg_dict]
            if cat_vars:
                vals.append(float(np.mean([avg_dict[v] for v in cat_vars])))
            else:
                vals.append(0.0)
        return vals + [vals[0]]

    # Build per-model radar vectors (closed curve) and detect overlaps.
    model_vectors = {
        model: np.array(_radar_values(avg_dict), dtype=float)
        for model, avg_dict in llm_model_avgs.items()
    }

    # Models with the same rounded profile are considered overlapping.
    # Use a tiny deterministic radial offset to separate them visually.
    overlap_groups: dict[tuple, list[str]] = {}
    for model, vec in model_vectors.items():
        key = tuple(np.round(vec, 4).tolist())
        overlap_groups.setdefault(key, []).append(model)

    model_display_shift = {m: 0.0 for m in model_vectors.keys()}
    for grp in overlap_groups.values():
        if len(grp) <= 1:
            continue
        grp_sorted = sorted(grp)
        center = (len(grp_sorted) - 1) / 2.0
        for idx, m in enumerate(grp_sorted):
            # Keep offsets very small so quantitative interpretation is preserved.
            model_display_shift[m] = (idx - center) * 0.006

    # Optional diagnostic print for reproducibility
    n_overlap_groups = sum(1 for grp in overlap_groups.values() if len(grp) > 1)
    if n_overlap_groups > 0:
        n_models_in_overlap = sum(len(grp) for grp in overlap_groups.values() if len(grp) > 1)
        print(f"[Radar topics] {n_models_in_overlap} models in {n_overlap_groups} overlapping profile group(s); applying tiny display offsets.")

    # ── Figure ────────────────────────────────────────────────────────────────
    with plt.style.context(["science", "no-latex"]):
        mpl.rcParams.update({
            "font.size": 10, "axes.labelsize": 11,
            "legend.fontsize": 9, "axes.titlesize": 12,
        })
        fig, ax = plt.subplots(figsize=(9, 8),
                               subplot_kw=dict(polar=True),
                               facecolor="white")
        ax.set_facecolor("white")

        # Global aggregate reference (dashed black)
        vals_all = _radar_values(avg_llm_all)
        ax.plot(angles, vals_all, linestyle="--", linewidth=2.2, color="#222222",
            marker="o", markersize=6, markerfacecolor="white",
            markeredgewidth=1.6, label="LLM (aggregate)", alpha=0.95)

        # One trace per LLM model
        line_styles = ["-", "--", ":", "-."]
        models = sorted(model_vectors.keys())
        family_marker, model_display_family, _family_counts = build_compact_family_mapping(models)
        for i, model in enumerate(models):
            vals = model_vectors[model].copy()
            shift = model_display_shift.get(model, 0.0)
            if shift != 0.0:
                vals = np.clip(vals + shift, 0.0, 1.0)

            col = PALETTE[i % len(PALETTE)]
            mk = family_marker[model_display_family[model]]
            ls = line_styles[(i // len(PALETTE)) % len(line_styles)]
            label = model if shift == 0.0 else f"{model} (offset)"

            ax.plot(angles, vals,
                    linestyle=ls, linewidth=1.25, color=col,
                    marker=mk, markersize=4.6,
                    markerfacecolor="white", markeredgewidth=1.0,
                    label=label, alpha=0.88)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(all_cats, size=11, fontweight="medium")
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(
            ["0.2", "0.4", "0.6", "0.8", "1.0"],
            size=9,
            color="#555555",
        )
        ax.grid(True, linestyle="-", alpha=0.2, color="#888888", linewidth=0.5)
        ax.spines["polar"].set_color("#CCCCCC")
        ax.spines["polar"].set_linewidth(0.8)

        legend = ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.06, 1.0),
            fontsize=8.5,
            frameon=True,
            fancybox=False,
            edgecolor="#CCCCCC",
            facecolor="white",
            title="LLM models",
            title_fontsize=9,
        )
        legend.get_title().set_fontweight("bold")

        plt.title(
            "LLM Average Response Rates by Topic (one trace per model)",
            size=13,
            y=1.06,
            fontweight="bold",
            color="#111111",
        )
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            print(f"LLM topics radar chart saved to: {output_path}")
        plt.close(fig)


def compute_ks_test_alignment(llm_df: pd.DataFrame,
                             df_human_norm: pd.DataFrame,
                             topic_df: pd.DataFrame,
                             country_col: str,
                             c2f: dict,
                             code_to_family: dict,
                             v_cols: list) -> pd.DataFrame:
    """
    Compute Kolmogorov-Smirnov test statistics between LLM and human responses
    at the language family level.

    Methodological note
    -------------------
        To avoid countries with very large sample sizes dominating family-level
        comparisons, this function uses balanced-by-country sampling within each
        language family before running KS tests.

        Statistical implementation details
        ----------------------------------
        KS is applied at the *single-variable* level (its standard 1D use case).
        For each (topic, family, model), we compute one KS statistic per variable
        and then aggregate:
            - ks_statistic = mean(variable-level KS D)
            - p_value = Fisher-combined p-value across variable-level tests
    
    Args:
        llm_df: LLM responses dataframe
        df_human_norm: Normalized human survey dataframe
        topic_df: Topic matching dataframe
        country_col: Name of the language family column in df_human_norm
        c2f: Country to language family mapping
        v_cols: List of variable columns
    
    Returns:
        DataFrame with KS test results: columns are topic, language_family,
        model, ks_statistic, p_value, n_variables_used, n_human, n_llm
    """
    from scipy.stats import ks_2samp, combine_pvalues
    
    var_to_cat = dict(zip(topic_df["variable_name"], topic_df["category"]))

    # Balance each family by capping per-country contribution.
    MAX_VALUES_PER_COUNTRY_VAR = 500
    MIN_VALUES_PER_GROUP_VAR = 5
    rng = np.random.default_rng(42)

    def _balanced_values_human_var(human_sub: pd.DataFrame,
                                   family_countries: list,
                                   var: str) -> np.ndarray:
        """Balanced human values for one variable across countries in a family."""
        vals_out = []
        if var not in human_sub.columns:
            return np.array([], dtype=float)
        for country in family_countries:
            csub = human_sub[human_sub[country_col] == country]
            if csub.empty:
                continue
            arr = csub[var].to_numpy(dtype=float).ravel()
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                continue
            k = min(MAX_VALUES_PER_COUNTRY_VAR, arr.size)
            if arr.size > k:
                arr = arr[rng.choice(arr.size, size=k, replace=False)]
            vals_out.extend(arr.tolist())
        return np.asarray(vals_out, dtype=float)

    def _balanced_values_llm_var(llm_sub_model_family: pd.DataFrame,
                                 family_codes: list,
                                 var: str) -> np.ndarray:
        """Balanced LLM values for one variable across countries in a family."""
        vals_out = []
        for code in family_codes:
            csub = llm_sub_model_family[
                llm_sub_model_family["_country_code"].astype(str).str.lower() == str(code).lower()
            ]
            if csub.empty:
                continue
            arr = csub.loc[csub["variable"] == var, "norm_response"].dropna().to_numpy(dtype=float)
            if arr.size == 0:
                continue
            arr = arr[~np.isnan(arr)]
            k = min(MAX_VALUES_PER_COUNTRY_VAR, arr.size)
            if arr.size > k:
                arr = arr[rng.choice(arr.size, size=k, replace=False)]
            vals_out.extend(arr.tolist())
        return np.asarray(vals_out, dtype=float)
    
    results = []
    
    # Get unique language families from human data
    families = sorted(set(c2f.values()))
    models = sorted(llm_df["_model"].unique())
    
    llm_work = llm_df.copy()
    llm_work["_country_code"] = llm_work["_country_code"].astype(str).str.lower()
    llm_work["language_family"] = llm_work["_country_code"].astype(str).str.lower().map(code_to_family)

    # For each language family and model (same-family human vs LLM only)
    for family in families:
        family_countries = [c for c, f in c2f.items() if f == family]
        family_codes = [cc for cc, f in code_to_family.items() if f == family]
        human_mask = df_human_norm[country_col].isin(family_countries)
        human_sub = df_human_norm[human_mask]
        
        for model in models:
            llm_mask = (llm_work["_model"] == model) & (llm_work["language_family"] == family)
            llm_sub = llm_work[llm_mask]
            
            # Compare by topic/category
            for topic in sorted(set(topic_df["category"].unique()) - {"Unknown"}):
                topic_vars = [v for v, c in var_to_cat.items() if c == topic]

                var_ks_stats = []
                var_p_values = []
                n_h_total = 0
                n_l_total = 0

                # Standard KS use: one variable at a time, then aggregate
                for var in topic_vars:
                    hvals = _balanced_values_human_var(human_sub, family_countries, var)
                    lvals = _balanced_values_llm_var(llm_sub, family_codes, var)
                    if len(hvals) < MIN_VALUES_PER_GROUP_VAR or len(lvals) < MIN_VALUES_PER_GROUP_VAR:
                        continue
                    ks_stat_var, p_val_var = ks_2samp(hvals, lvals)
                    var_ks_stats.append(float(ks_stat_var))
                    var_p_values.append(float(p_val_var))
                    n_h_total += int(len(hvals))
                    n_l_total += int(len(lvals))

                if var_ks_stats:
                    ks_stat = float(np.mean(var_ks_stats))
                    if len(var_p_values) == 1:
                        p_val = float(var_p_values[0])
                    else:
                        _, p_val = combine_pvalues(var_p_values, method="fisher")
                        p_val = float(p_val)
                    results.append({
                        "topic": topic,
                        "language_family": family,
                        "model": model,
                        "ks_statistic": ks_stat,
                        "p_value": p_val,
                        "n_variables_used": int(len(var_ks_stats)),
                        "n_human": int(n_h_total),
                        "n_llm": int(n_l_total),
                    })
    
    return pd.DataFrame(results)


def plot_ks_alignment(ks_df: pd.DataFrame, output_path=None):
    """
    Plot K-S test statistics showing alignment between human and LLM responses
    at language family level, averaged across topics and models.
    
    Creates a heatmap where:
    - Rows: language families
    - Columns: LLM models
    - Values: average K-S statistic (lower = better alignment)
    
    Args:
        ks_df: DataFrame from compute_ks_test_alignment
        output_path: Path to save the chart
    """
    # Aggregate K-S statistics: average across topics for each (family, model) pair
    ks_summary = ks_df.groupby(["language_family", "model"])["ks_statistic"].mean().reset_index()
    ks_pivot = ks_summary.pivot(index="language_family", 
                                columns="model", 
                                values="ks_statistic")
    
    # Create heatmap
    with plt.style.context(["science", "no-latex"]):
        mpl.rcParams.update({
            "font.size": 10, "axes.labelsize": 11,
            "legend.fontsize": 9, "axes.titlesize": 12,
        })
        fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
        
        # Create heatmap with scientific colormap (lower K-S = better = cooler colors)
        im = ax.imshow(ks_pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.5)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(ks_pivot.columns)))
        ax.set_yticks(np.arange(len(ks_pivot.index)))
        ax.set_xticklabels(ks_pivot.columns, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(ks_pivot.index, fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("K-S Statistic (lower = better alignment)", rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(ks_pivot.index)):
            for j in range(len(ks_pivot.columns)):
                val = ks_pivot.values[i, j]
                if not np.isnan(val):
                    text = ax.text(j, i, f"{val:.3f}",
                                  ha="center", va="center",
                                  color="white" if val > 0.25 else "black",
                                  fontsize=8, fontweight="bold")
        
        ax.set_xlabel("LLM Model", fontweight="bold", fontsize=11)
        ax.set_ylabel("Language Family", fontweight="bold", fontsize=11)
        ax.set_title("LLM-Human Alignment by K-S Test\n(Lower K-S = Better Alignment)", 
                    fontweight="bold", fontsize=13, pad=15)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            print(f"K-S alignment heatmap saved to: {output_path}")
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM responses analysis: PCA projection + radar chart vs human data"
    )
    parser.add_argument("-l", "--llm-dir",    default=str(DEFAULT_LLM_DIR),
                        help="Directory containing llm_survey_*.csv files")
    parser.add_argument("-u", "--human",      default=str(DEFAULT_HUMAN_CSV),
                        help="Human survey CSV")
    parser.add_argument("-t", "--topics",     default=str(DEFAULT_TOPICS_CSV),
                        help="Topic matching CSV (semicolon-separated)")
    parser.add_argument("-o", "--output",     default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for plots")
    parser.add_argument("--models",           nargs="*", default=None,
                        help="Restrict to these model names (default: all)")
    parser.add_argument("--countries",        nargs="*", default=None,
                        help="Restrict to these country codes (default: all)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    llm_df = load_llm_responses(Path(args.llm_dir))
    scale_source = DEFAULT_SCALE_CATALOG if DEFAULT_SCALE_CATALOG.exists() else DEFAULT_SCALES_DIR
    print(f"Loading canonical scale metadata from: {scale_source}")
    canonical_meta = load_canonical_scale_metadata(scale_source)
    valid_ranges = {
        var: (float(meta["min"]), float(meta["max"]))
        for var, meta in canonical_meta.items()
        if meta.get("min") is not None and meta.get("max") is not None
    }
    inferred_fallback = infer_variable_valid_ranges(llm_df)
    for var, rng in inferred_fallback.items():
        valid_ranges.setdefault(var, rng)

    # Human data is loaded after inferring valid per-variable bounds so DK/NA
    # removal respects each question's scale (e.g., v38 allows 8/9 but not 88/99).
    df_human_norm, topic_df, c2f, country_col, v_cols, norm_params = load_human_data(
        Path(args.human), Path(args.topics), valid_ranges=valid_ranges
    )

    if args.models:
        llm_df = llm_df[llm_df["_model"].isin(args.models)]
        print(f"Filtered to models: {args.models}")
    if args.countries:
        llm_df = llm_df[llm_df["_country_code"].isin(args.countries)]
        print(f"Filtered to countries: {args.countries}")

    # Re-normalise LLM responses using human-derived scales (see docstring
    # of _apply_human_normalization for the full explanation).
    print("\nRe-normalising LLM responses using human-derived scale…")
    llm_df = _apply_human_normalization(llm_df, norm_params, valid_ranges=valid_ranges)
    _pca_rows = llm_df[llm_df["variable"].isin(PCA_SELECTED_VARS)]["norm_response"]
    print(f"  PCA variables — "
          f"min={_pca_rows.min():.4f}  "
          f"max={_pca_rows.max():.4f}  "
          f"mean={_pca_rows.mean():.4f}  "
          f"NaN={_pca_rows.isna().sum():,}"
          f"  (non-PCA vars outside [0,1] are excluded from PCA projection)")

    # ── Per-model exclusion rates after renormalisation ───────────────────────
    print(f"\n── Excluded after renormalisation (DK/NA + out-of-range) per model ─")
    for _m in sorted(llm_df["_model"].unique()):
        _grp   = llm_df[llm_df["_model"] == _m]
        _n_nan = _grp["norm_response"].isna().sum()
        _n_tot = len(_grp)
        print(f"  {_m:25s}  {100*_n_nan/_n_tot:.1f}%  ({_n_nan:,} / {_n_tot:,})")

    # ── Build PCA on human data ───────────────────────────────────────────────
    _pca_fit = build_pca_data(df_human_norm, country_col, v_cols, c2f)
    pca, scaler, pca_cols = _pca_fit[0], _pca_fit[1], _pca_fit[2]

    # ── Build lowercase country-code → language-family mapping ─────────────
    code_to_family = {}
    for _, row in (df_human_norm[["c_abrv", country_col]]
                   .drop_duplicates().iterrows()):
        code  = str(row["c_abrv"]).lower()
        cname = row[country_col]
        code_to_family[code] = c2f.get(cname, "Unknown")

    # ── Human background points with country-language split logic ───────────
    human_bg = build_human_country_language_background(
        df_human_norm=df_human_norm,
        country_col=country_col,
        pca=pca,
        scaler=scaler,
        pca_cols=pca_cols,
        c2f=c2f,
    )
    human_country_ref = build_human_country_reference_points(human_bg)
    print(f"  Human PCA background points (country-language): {len(human_bg)}")
    print(f"  Canonical human country anchors: {len(human_country_ref)}")

    # ── Project LLM responses onto PCA ───────────────────────────────────────
    print("\nProjecting LLM responses onto PCA space…")
    llm_proj = project_llm_onto_pca(llm_df, pca, scaler, pca_cols)
    print(f"  {len(llm_proj)} (model, country) pairs projected.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    # Generate both one-column and two-column versions of PCA plots for paper layout flexibility
    pca_path_1col    = out_dir / "llm_pca_projection_1col.png"
    pca_path_2col    = out_dir / "llm_pca_projection_2col.png"
    pca_bymodel_path_1col = out_dir / "llm_pca_projection_by_model_1col.png"
    pca_bymodel_path_2col = out_dir / "llm_pca_projection_by_model_2col.png"
    pca_english_vs_rest_path_1col = out_dir / "llm_pca_english_vs_non_english_mean_1col.png"
    pca_english_vs_rest_path_2col = out_dir / "llm_pca_english_vs_non_english_mean_2col.png"
    pca_human_vs_llm_path_1col = out_dir / "llm_pca_human_vs_llm_english_multilingual_1col.png"
    pca_human_vs_llm_path_2col = out_dir / "llm_pca_human_vs_llm_english_multilingual_2col.png"
    human_topics_radar_path = out_dir / "human_topics_radar.png"
    llm_topics_radar_path = out_dir / "llm_topics_radar.png"
    ks_alignment_path = out_dir / "ks_alignment_by_language_family.png"

    print("\nGenerating PCA projection plot (centroids, black markers) — one and two column versions…")
    plot_pca(human_bg, llm_proj, pca, output_path=str(pca_path_1col), column_width=3.5)
    plot_pca(human_bg, llm_proj, pca, output_path=str(pca_path_2col), column_width=7.0)

    print("Generating PCA projection plot (family centroids per model) — one and two column versions…")
    plot_pca_by_model(human_bg, llm_proj, pca, code_to_family,
                      output_path=str(pca_bymodel_path_1col), column_width=3.5)
    plot_pca_by_model(human_bg, llm_proj, pca, code_to_family,
                      output_path=str(pca_bymodel_path_2col), column_width=7.0)

    # Weight for each (model, country) projection point: number of valid
    # PCA-variable responses used to compute that point.
    llm_country_weights = (
        llm_df[llm_df["variable"].isin(pca_cols) & llm_df["norm_response"].notna()]
        .groupby(["_model", "_country_code"])  # raw response support
        .size()
        .reset_index(name="weight")
    )

    print("Generating PCA plot (one weighted LLM point per language family) — one and two column versions…")
    plot_pca_weighted_family_overlay(
        human_bg=human_bg,
        llm_proj=llm_proj,
        llm_country_weights=llm_country_weights,
        pca=pca,
        code_to_family=code_to_family,
        output_path=str(out_dir / "llm_pca_weighted_family_overlay_1col.png"),
        column_width=3.5,
    )
    plot_pca_weighted_family_overlay(
        human_bg=human_bg,
        llm_proj=llm_proj,
        llm_country_weights=llm_country_weights,
        pca=pca,
        code_to_family=code_to_family,
        output_path=str(out_dir / "llm_pca_weighted_family_overlay_singlecol.png"),
        column_width=3.5,
    )
    plot_pca_weighted_family_overlay(
        human_bg=human_bg,
        llm_proj=llm_proj,
        llm_country_weights=llm_country_weights,
        pca=pca,
        code_to_family=code_to_family,
        output_path=str(out_dir / "llm_pca_weighted_family_overlay_2col.png"),
        column_width=7.0,
    )

    print("Generating PCA plot (English vs mean of non-English per model) — one and two column versions…")
    plot_pca_english_vs_non_english_mean(
        llm_proj, pca, output_path=str(pca_english_vs_rest_path_1col),
        english_code="gb", column_width=3.5)
    plot_pca_english_vs_non_english_mean(
        llm_proj, pca, output_path=str(pca_english_vs_rest_path_2col),
        english_code="gb", column_width=7.0)

    print("Generating PCA plot (human vs LLM, English vs multilingual) — one and two column versions…")
    plot_pca_human_vs_llm_english_multilingual(
        human_country_ref, llm_proj, pca,
        output_path=str(pca_human_vs_llm_path_1col),
        english_code="gb", column_width=3.5)
    plot_pca_human_vs_llm_english_multilingual(
        human_country_ref, llm_proj, pca,
        output_path=str(pca_human_vs_llm_path_2col),
        english_code="gb", column_width=7.0)

    print("Generating radar chart (human topic profiles by language family)…")
    plot_human_topics_radar(
        df_human_norm=df_human_norm,
        topic_df=topic_df,
        country_col=country_col,
        c2f=c2f,
        v_cols=v_cols,
        output_path=str(human_topics_radar_path),
    )

    print("Generating radar chart (LLM topic profiles, weighted by language family)…")
    plot_llm_topics_radar_weighted_by_family(
        llm_df=llm_df,
        topic_df=topic_df,
        code_to_family=code_to_family,
        output_path=str(llm_topics_radar_path),
    )

    print("Computing K-S test alignment statistics…")
    ks_df = compute_ks_test_alignment(llm_df, df_human_norm, topic_df,
                                       country_col, c2f, code_to_family, v_cols)
    print(f"  {len(ks_df)} (topic, family, model) comparisons computed.")

    print("Generating K-S alignment heatmap…")
    plot_ks_alignment(ks_df, output_path=str(ks_alignment_path))

    print("\nDone.")


if __name__ == "__main__":
    main()
