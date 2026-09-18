import re
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import numpy as np
import pandas as pd


# Multilingual DK/NA hints commonly seen in survey labels.
# Keep uncertainty phrases separate from explicit nonresponse phrases so that
# substantive options like "No sé qué pensar" or "don't know what to think"
# are not automatically treated as missing when encoded with regular response
# values (e.g. code 3 in v62).
_UNCERTAINTY_LABEL_PATTERNS = [
    r"\bdk\b",
    r"don't\s*know",
    r"dont\s*know",
    r"\bno\s+sabe\b",
    r"\bno\s+se\b",
    r"\bne\s*vem\b",
    r"\bnev[ií]m\b",
    r"не\s*знаю",
    r"\bvet\s*inte\b",
]
_EXPLICIT_NONRESPONSE_PATTERNS = [
    r"(?<!\w)n\s*/\s*a(?!\w)",
    r"\bnot\s+applicable\b",
    r"\bnon\s+applicable\b",
    r"\bnot\s+relevant\b",
    r"\bnon\s+concern[ée]?\b",
    r"\bsin\s+respuesta\b",
    r"\bno\s+aplica\b",
    r"no\s+contesta",
    r"no\s+answer",
    r"\brefused\b",
    r"\brefusal\b",
    r"\brefus\s+de\s+r[ée]ponse\b",
    r"\bmissing\b",
    r"\bnon\s+sa\b",
    r"primjenjiv",
    r"primenljiv",
    r"net[ýy]ka",
    r"nije\s+relevant",
    r"ikke\s+relevant",
    r"\bnsp\b",
    r"\bnr\b",
]
_UNCERTAINTY_RE = re.compile("|".join(_UNCERTAINTY_LABEL_PATTERNS), flags=re.IGNORECASE)
_EXPLICIT_NONRESPONSE_RE = re.compile("|".join(_EXPLICIT_NONRESPONSE_PATTERNS), flags=re.IGNORECASE)


def parse_scale_pairs(scale_text: object) -> list[tuple[int, str]]:
    """Extract pairs like 1 (label), 88 (DK), etc. from response_scale text."""
    s = str(scale_text or "")
    pairs = []
    for m in re.finditer(r"(?<!\d)(\d{1,3})(?!\d)\s*\(([^)]*)\)", s):
        pairs.append((int(m.group(1)), m.group(2).strip()))
    return pairs


def _extract_all_codes(scale_text: object) -> set[int]:
    """Extract all numeric codes appearing in a response scale text."""
    pairs = parse_scale_pairs(scale_text)
    if pairs:
        return {int(n) for n, _ in pairs}
    return {int(x) for x in re.findall(r"(?<!\d)(\d{1,3})(?!\d)", str(scale_text or ""))}


def _contiguous_range_from_codes(codes: set[int]) -> Optional[Tuple[int, int]]:
    """Select the most plausible substantive range as the best contiguous run."""
    vals = sorted(int(v) for v in codes if int(v) >= 0)
    if not vals:
        return None

    runs: list[tuple[int, int]] = []
    start = vals[0]
    prev = vals[0]
    for x in vals[1:]:
        if x == prev + 1:
            prev = x
            continue
        runs.append((start, prev))
        start = x
        prev = x
    runs.append((start, prev))

    def _rank(run: tuple[int, int]) -> tuple[int, int, int]:
        a, b = run
        length = b - a + 1
        contains_one = 1 if (a <= 1 <= b) else 0
        return (length, contains_one, -a)

    a, b = max(runs, key=_rank)
    return int(a), int(b)


def _allowed_values_from_meta(var_meta: dict) -> list[int]:
    """Return substantive allowed values inferred from min/max and DK/NA codes."""
    lo = var_meta.get("min")
    hi = var_meta.get("max")
    dkna = set(var_meta.get("dkna_codes", set()))

    if lo is not None and hi is not None and int(hi) >= int(lo):
        # EVS/WVS scales are compact; keep explicit allowed values for clarity.
        vals = [int(v) for v in range(int(lo), int(hi) + 1) if int(v) not in dkna]
        return vals

    codes = sorted(int(v) for v in set(var_meta.get("all_codes", [])) if int(v) not in dkna)
    return codes


def infer_bounds_from_scale_text(scale_text: object) -> Optional[Tuple[int, int]]:
    """
    Infer the substantive answer range from response_scale.

    Heuristic:
      - Parse code/label pairs when available.
      - Remove explicit DK/NA / non-applicable codes before inferring bounds.
      - For EVS/WVS ordinal scales, treat the remaining low codes as a
        continuous substantive range and return (min, max). This preserves
        valid `0..10` scales while excluding tails like 77/88/99.
      - If no low substantive codes remain, fallback to the full observed range.
    """
    nums = _extract_all_codes(scale_text)
    if not nums:
        return None

    # First try removing explicitly identified DK/NA codes.
    dkna_codes = infer_dkna_codes_from_scale_text(scale_text)
    substantive = {n for n in nums if n not in dkna_codes}

    # Endpoint-encoded scales are common in EVS/WVS, e.g. "1 (...) - 10 (...)"
    # with no explicit intermediate labels. Treat as closed interval.
    sub_sorted = sorted(substantive)
    if (
        len(sub_sorted) <= 3
        and len(sub_sorted) >= 2
        and 0 <= sub_sorted[0] <= 20
        and 0 <= sub_sorted[-1] <= 20
        and sub_sorted[-1] - sub_sorted[0] >= 4
        and sub_sorted[0] in {0, 1}
    ):
        return int(sub_sorted[0]), int(sub_sorted[-1])
    if len(sub_sorted) == 2 and 0 <= sub_sorted[0] <= 20 and 0 <= sub_sorted[1] <= 20:
        return int(sub_sorted[0]), int(sub_sorted[1])

    rng = _contiguous_range_from_codes(substantive) if substantive else None
    if rng is not None:
        return rng

    # Fallback: infer from all codes using contiguous-run structure.
    rng = _contiguous_range_from_codes(nums)
    if rng is not None:
        return rng

    return (min(nums), max(nums))


def infer_dkna_codes_from_scale_text(scale_text: object) -> set[int]:
    """Infer DK/NA codes from labels and common sentinel values."""
    pairs = parse_scale_pairs(scale_text)
    codes: set[int] = set()
    sentinel_codes = {77, 88, 99, 777, 888, 999}
    short_sentinel_codes = {7, 8, 9}
    for code, label in pairs:
        if _EXPLICIT_NONRESPONSE_RE.search(label):
            codes.add(code)
            continue
        if _UNCERTAINTY_RE.search(label) and code in short_sentinel_codes | sentinel_codes:
            codes.add(code)
            continue
        if code in sentinel_codes:
            codes.add(code)
    return codes


def infer_variable_scale_metadata(df: pd.DataFrame,
                                  variable_col: str = "variable",
                                  scale_col: str = "response_scale") -> Dict[str, dict]:
    """
    Build per-variable metadata:
      {
        var: {
          "min": int|None,
          "max": int|None,
          "dkna_codes": set[int],
        }
      }
    """
    if variable_col not in df.columns:
        return {}

    metadata: Dict[str, dict] = {}
    for var, grp in df.groupby(variable_col):
        bounds_candidates = []
        dkna_codes: set[int] = set()
        code_union: set[int] = set()

        if scale_col in grp.columns:
            for scale in grp[scale_col].dropna().astype(str).unique():
                b = infer_bounds_from_scale_text(scale)
                if b is not None:
                    bounds_candidates.append(b)
                dkna_codes |= infer_dkna_codes_from_scale_text(scale)
                code_union |= _extract_all_codes(scale)

        chosen_bounds = None
        if bounds_candidates:
            chosen_bounds = Counter(bounds_candidates).most_common(1)[0][0]

        base = {
            "min": None if chosen_bounds is None else int(chosen_bounds[0]),
            "max": None if chosen_bounds is None else int(chosen_bounds[1]),
            "dkna_codes": dkna_codes,
            "all_codes": sorted(code_union),
        }
        base["allowed_values"] = _allowed_values_from_meta(base)
        metadata[str(var)] = base
    return metadata


def build_scale_catalog_from_surveys_parsed(surveys_parsed_dir: Path) -> pd.DataFrame:
    """Build a canonical per-variable scale catalog from `Surveys_parsed/`.

    Output columns:
      - variable
      - valid_min
      - valid_max
      - allowed_values_json
      - dkna_codes_json
      - response_scale_text
      - question_number
      - question_text
      - response_options
      - source_language
      - source_file
    """
    surveys_parsed_dir = Path(surveys_parsed_dir)
    rows: list[pd.DataFrame] = []
    if not surveys_parsed_dir.exists():
        return pd.DataFrame()

    for csv_path in sorted(surveys_parsed_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue

        variable_col = next((c for c in ("Variable_Name", "variable", "variable_name")
                             if c in df.columns), None)
        scale_col = next((c for c in ("Response_Scale", "response_scale", "Scale")
                          if c in df.columns), None)
        if variable_col is None or scale_col is None:
            continue

        qn_col = "Question_Number" if "Question_Number" in df.columns else None
        qt_col = "Question_Text" if "Question_Text" in df.columns else None
        ro_col = "Response_Options" if "Response_Options" in df.columns else None

        sub = pd.DataFrame({
            "variable": df[variable_col].astype(str),
            "response_scale": df[scale_col].astype(str),
            "question_number": df[qn_col].astype(str) if qn_col else "",
            "question_text": df[qt_col].astype(str) if qt_col else "",
            "response_options": df[ro_col].astype(str) if ro_col else "",
            "source_file": csv_path.name,
        })
        m = re.search(r"_([a-z]{2})\.csv$", csv_path.name)
        sub["source_language"] = m.group(1) if m else ""
        rows.append(sub)

    if not rows:
        return pd.DataFrame()

    raw = pd.concat(rows, ignore_index=True)
    raw = raw.dropna(subset=["variable", "response_scale"])

    meta = infer_variable_scale_metadata(raw, variable_col="variable", scale_col="response_scale")

    lang_priority = {"es": 0, "gb": 1, "fr": 2, "it": 3}
    out_rows: list[dict] = []
    for var, grp in raw.groupby("variable", sort=True):
        grp2 = grp.copy()
        grp2["_prio"] = grp2["source_language"].map(lambda x: lang_priority.get(str(x), 99))
        grp2 = grp2.sort_values(by=["_prio", "source_file"])
        rep = grp2.iloc[0]

        var_meta = meta.get(str(var), {})
        allowed_values = [int(v) for v in var_meta.get("allowed_values", [])]
        dkna_codes = sorted(int(v) for v in set(var_meta.get("dkna_codes", set())))

        out_rows.append({
            "variable": str(var),
            "valid_min": var_meta.get("min"),
            "valid_max": var_meta.get("max"),
            "allowed_values_json": json.dumps(allowed_values, ensure_ascii=False),
            "dkna_codes_json": json.dumps(dkna_codes, ensure_ascii=False),
            "response_scale_text": str(rep.get("response_scale", "")),
            "question_number": str(rep.get("question_number", "")),
            "question_text": str(rep.get("question_text", "")),
            "response_options": str(rep.get("response_options", "")),
            "source_language": str(rep.get("source_language", "")),
            "source_file": str(rep.get("source_file", "")),
        })

    return pd.DataFrame(out_rows).sort_values("variable").reset_index(drop=True)


def write_scale_catalog(catalog_csv_path: Path, surveys_parsed_dir: Path) -> Path:
    """Generate and write the canonical scale catalog CSV."""
    catalog_csv_path = Path(catalog_csv_path)
    catalog_csv_path.parent.mkdir(parents=True, exist_ok=True)
    cat_df = build_scale_catalog_from_surveys_parsed(surveys_parsed_dir)
    if cat_df.empty:
        raise ValueError(f"No scale metadata could be built from {surveys_parsed_dir}")
    cat_df.to_csv(catalog_csv_path, index=False, encoding="utf-8")
    return catalog_csv_path


def _loads_json_int_list(raw: object) -> list[int]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    txt = str(raw).strip()
    if not txt:
        return []
    try:
        arr = json.loads(txt)
    except json.JSONDecodeError:
        return []
    if not isinstance(arr, list):
        return []
    out: list[int] = []
    for x in arr:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def load_canonical_scale_metadata(scale_source: Path) -> Dict[str, dict]:
    """Load canonical scale metadata from a CSV catalog or `Surveys_parsed/`.

    Parameters
    ----------
    scale_source : Path
        Either:
          - path to canonical catalog CSV (recommended), or
          - path to `Surveys_parsed/` directory (fallback/inference mode).
    """
    scale_source = Path(scale_source)
    if not scale_source.exists():
        return {}

    # Preferred: precomputed canonical catalog CSV
    if scale_source.is_file():
        try:
            cat = pd.read_csv(scale_source, low_memory=False)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            return {}

        if "variable" not in cat.columns:
            return {}

        out: Dict[str, dict] = {}
        for _, r in cat.iterrows():
            var = str(r.get("variable", "")).strip()
            if not var:
                continue

            lo = r.get("valid_min")
            hi = r.get("valid_max")
            try:
                lo = None if pd.isna(lo) else int(float(lo))
            except (TypeError, ValueError):
                lo = None
            try:
                hi = None if pd.isna(hi) else int(float(hi))
            except (TypeError, ValueError):
                hi = None

            allowed = _loads_json_int_list(r.get("allowed_values_json"))
            dkna = set(_loads_json_int_list(r.get("dkna_codes_json")))

            if not allowed and lo is not None and hi is not None and hi >= lo:
                allowed = [int(v) for v in range(lo, hi + 1) if int(v) not in dkna]

            out[var] = {
                "min": lo,
                "max": hi,
                "dkna_codes": dkna,
                "allowed_values": allowed,
                "response_scale_text": str(r.get("response_scale_text", "") or ""),
            }
        return out

    # Fallback: infer directly from Surveys_parsed directory
    catalog = build_scale_catalog_from_surveys_parsed(scale_source)
    if catalog.empty:
        return {}

    tmp_meta: Dict[str, dict] = {}
    for _, row in catalog.iterrows():
        var = str(row["variable"])
        lo = row.get("valid_min")
        hi = row.get("valid_max")
        try:
            lo = None if pd.isna(lo) else int(float(lo))
        except (TypeError, ValueError):
            lo = None
        try:
            hi = None if pd.isna(hi) else int(float(hi))
        except (TypeError, ValueError):
            hi = None
        dkna = set(_loads_json_int_list(row.get("dkna_codes_json")))
        allowed = _loads_json_int_list(row.get("allowed_values_json"))
        tmp_meta[var] = {
            "min": lo,
            "max": hi,
            "dkna_codes": dkna,
            "allowed_values": allowed,
            "response_scale_text": str(row.get("response_scale_text", "") or ""),
        }
    return tmp_meta


def is_value_dkna(value: object, var_meta: Optional[dict]) -> bool:
    """Classify value as DK/NA using explicit codes + repeated 77/88/99 style."""
    if pd.isna(value):
        return False

    try:
        x = int(float(value))
    except (TypeError, ValueError):
        return False

    if not var_meta:
        return x in {7, 8, 9, 77, 88, 99, 777, 888, 999}

    dkna_codes = set(var_meta.get("dkna_codes", set()))
    if x in dkna_codes:
        return True

    sx = str(abs(x))
    if len(sx) > 1 and len(set(sx)) == 1 and sx[0] in {"7", "8", "9"}:
        return True
    return False


def is_value_out_of_range(value: object, var_meta: Optional[dict]) -> bool:
    """Check if value is outside the substantive range for the variable."""
    if pd.isna(value):
        return False
    if not var_meta:
        return False
    lo = var_meta.get("min")
    hi = var_meta.get("max")
    if lo is None or hi is None:
        return False
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return bool((x < lo) or (x > hi))


def keep_only_substantive_range(series: pd.Series,
                                var_meta: Optional[dict]) -> pd.Series:
    """Mask out DK/NA and out-of-range values for one variable series."""
    s = pd.to_numeric(series, errors="coerce")
    if var_meta is None:
        return s.replace([7, 8, 9, 77, 88, 99], np.nan)

    lo = var_meta.get("min")
    hi = var_meta.get("max")

    s2 = s.copy()
    s2 = s2.mask(s2.apply(lambda x: is_value_dkna(x, var_meta)))
    if lo is not None and hi is not None:
        s2 = s2.where((s2 >= lo) & (s2 <= hi), other=np.nan)
    return s2
