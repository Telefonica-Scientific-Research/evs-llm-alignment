"""
Shared constants and utilities for EVS survey analysis.

Import from here to guarantee identical behaviour across all scripts:

    from survey_variables import PCA_SELECTED_VARS
    from survey_variables import normalize_variable, normalize_survey_data

Variables v174, v174a, v175, v175a are excluded from PCA_SELECTED_VARS because
they correspond to open-text / free-response questions that do not yield
comparable numeric scale values across languages and models.
"""

import numpy as np
import pandas as pd


# DK / NA codes that are replaced with NaN before normalisation
DK_NA_CODES: tuple = (8, 9)


def normalize_variable(
    series: pd.Series,
    exclude_values: tuple = DK_NA_CODES,
) -> pd.Series:
    """
    Min-max normalise a single survey variable to [0, 1].

    Steps
    -----
    1. Coerce to numeric (non-numeric → NaN).
    2. Replace DK/NA codes with NaN.
    3. Compute min/max on remaining valid values.
    4. Apply (x - min) / (max - min).

    If all values are NaN or all valid values are identical the original
    (NaN-replaced) series is returned unchanged.
    """
    s = pd.to_numeric(series, errors="coerce")
    for v in exclude_values:
        s = s.replace(v, np.nan)
    valid = s.dropna()
    if len(valid) == 0 or valid.max() == valid.min():
        return s
    return (s - valid.min()) / (valid.max() - valid.min())


def normalize_survey_data(
    df: pd.DataFrame,
    v_columns: list,
    exclude_values: tuple = DK_NA_CODES,
) -> pd.DataFrame:
    """
    Apply :func:`normalize_variable` to every column in *v_columns*.

    Returns a copy of *df* with the named columns normalised in-place.
    Columns not present in *df* are silently skipped.
    """
    df_norm = df.copy()
    for col in v_columns:
        if col in df_norm.columns:
            df_norm[col] = normalize_variable(df_norm[col], exclude_values)
    return df_norm

PCA_SELECTED_VARS = [
    "v1", "v2", "v3", "v4", "v5", "v6",
    "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
    "v32", "v33", "v34", "v35", "v36", "v37", "v38",
    "v40", "v41", "v42", "v43", "v44", "v45", "v45a",
    "v46", "v47", "v48", "v49", "v50",
    "v57", "v58", "v59", "v60", "v61", "v62", "v63",
    "v65", "v66", "v67", "v68", "v69", "v70", "v71", "v72", "v73", "v74",
    "v75", "v76", "v77", "v78", "v79", "v80", "v81", "v82", "v83", "v84",
    "v85", "v86", "v87", "v88", "v89", "v90", "v91", "v92", "v93", "v94",
    "v95", "v96", "v98", "v99", "v100", "v101", "v102", "v103",
    "v104", "v105", "v106", "v107", "v108",
    "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116", "v117",
    "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v125", "v126",
    "v127", "v128", "v129", "v130", "v131", "v132", "v133", "v134", "v135",
    "v136", "v137", "v138", "v139", "v140", "v141", "v142", "v143", "v144",
    "v145", "v146", "v147", "v148", "v149", "v150", "v151", "v152", "v153",
    "v154", "v155", "v156", "v157", "v158", "v159", "v160", "v161", "v162",
    "v163", "v164", "v165", "v166", "v167", "v168", "v169", "v170", "v171",
    "v172", "v173", "v176", "v177", "v178",
    "v179", "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187",
    "v188", "v189", "v190", "v191", "v192", "v193", "v194", "v195", "v196",
    "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205",
    "v206", "v207",
    "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220",
    "v221", "v222", "v223", "v224",
]
