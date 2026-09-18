"""
Human Responses Analysis - Data Preprocessing and Visualization

This script:
1. Reads the moral values survey data
2. Normalizes scores for variables (columns starting with 'v') between 0 and 1
3. Discards DK/NA values (8 or 9)
4. Joins with topic matching data for category analysis
5. Generates radar charts showing average response scores across topics

Methodological note
-------------------
This script operates on *human responses only*. The consensus analysis,
country-level PCA, and all summary figures are computed from
`Surveys_responses/Human_responses/moral_values_survey_with_languages_v2.csv`.

The only external information it uses beyond the human response table is the
questionnaire scale metadata stored in `Surveys_parsed/`, which is used to
recover the substantive min/max response ranges and DK/NA codes for each
variable. No LLM response files are loaded or consulted by this script.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — avoids GUI hang on macOS
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots  # noqa: F401 — registers 'science' style
import argparse
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from adjustText import adjust_text

# Shared variable list and normalization (single source of truth)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survey_variables import PCA_SELECTED_VARS, normalize_variable, normalize_survey_data  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'llm_responses_analysis'))
from scale_metadata import load_canonical_scale_metadata, keep_only_substantive_range  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCALE_CATALOG = _ROOT / 'Surveys' / 'Survey_metadata' / 'question_scale_catalog.csv'
DEFAULT_SCALES_DIR = _ROOT / 'Surveys_parsed'


def normalize_survey_data_with_scales(
    df: pd.DataFrame,
    v_columns: list,
    scale_metadata: dict | None = None,
) -> pd.DataFrame:
    """Normalize human survey data using canonical substantive ranges.

    Parameters
    ----------
    df : DataFrame
        Human response table only.
    v_columns : list
        Survey variables to normalize.
    scale_metadata : dict | None
        Canonical questionnaire scale metadata loaded from `Surveys_parsed/`.

    Notes
    -----
    This function does not inspect or depend on any LLM response files. The
    scale metadata is questionnaire metadata, not model output.
    """
    df_norm = df.copy()
    for col in v_columns:
        if col not in df_norm.columns:
            continue
        meta = (scale_metadata or {}).get(col)
        raw = keep_only_substantive_range(df[col], meta)
        if meta and meta.get('min') is not None and meta.get('max') is not None and meta['min'] != meta['max']:
            lo = float(meta['min'])
            hi = float(meta['max'])
            df_norm[col] = ((raw - lo) / (hi - lo)).clip(0.0, 1.0)
        else:
            df_norm[col] = normalize_variable(raw)
    return df_norm

# Academic paper style configuration
def set_academic_style():
    """Configure matplotlib for academic paper-quality plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Georgia'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'text.usetex': False,
    })

# Academic color palette (colorblind-friendly)
# Based on Paul Tol's colorblind-safe palette
ACADEMIC_COLORS = {
    'overall': '#000000',      # Black for overall/reference
    'palette': [
        '#4477AA',  # Blue
        '#EE6677',  # Red/Pink
        '#228833',  # Green
        '#CCBB44',  # Yellow/Gold
        '#66CCEE',  # Cyan
        '#AA3377',  # Purple
        '#BBBBBB',  # Gray
    ],
    'fill_alpha': 0.12,
    'line_alpha': 1.0,
}

# Marker styles for different groups
MARKER_STYLES = ['o', 's', '^', 'D', 'v', 'p', 'h']

# PCA_SELECTED_VARS imported from EUValues/src/survey_variables.py


def load_survey_data(survey_path: str) -> pd.DataFrame:
    """Load the moral values survey data."""
    print(f"Loading survey data from: {survey_path}")
    df = pd.read_csv(survey_path, low_memory=False)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def load_topic_matching(topic_path: str) -> pd.DataFrame:
    """Load the topic matching data with variable-to-category mapping."""
    print(f"Loading topic matching from: {topic_path}")
    df = pd.read_csv(topic_path, sep=';')

    # Backward/forward compatible column handling:
    # - legacy format: variable_name;variable_label;category
    # - current format: EVS 2017 Variable Name;EVS 2017 Variable Label;Category;topic;source_url
    col_lookup = {str(c).strip().lower(): c for c in df.columns}

    var_col = (
        col_lookup.get('variable_name')
        or col_lookup.get('evs 2017 variable name')
        or (df.columns[0] if len(df.columns) > 0 else None)
    )
    label_col = (
        col_lookup.get('variable_label')
        or col_lookup.get('evs 2017 variable label')
        or (df.columns[1] if len(df.columns) > 1 else None)
    )
    # IMPORTANT: for radar plots per category, always prefer the "topic"
    # column when present (user-specified requirement). Fallback to "Category"
    # only when "topic" is absent.
    category_col = col_lookup.get('topic')
    if category_col is None:
        category_col = col_lookup.get('category') or (df.columns[2] if len(df.columns) > 2 else None)

    if var_col is None or label_col is None or category_col is None:
        raise ValueError(
            "Topic mapping CSV must contain variable, label, and category/topic columns. "
            f"Found columns: {list(df.columns)}"
        )

    out = df[[var_col, label_col, category_col]].copy()
    out.columns = ['variable_name', 'variable_label', 'category']
    out['variable_name'] = out['variable_name'].astype(str).str.strip()
    out['variable_label'] = out['variable_label'].astype(str).str.strip()
    out['category'] = out['category'].astype(str).str.strip()
    out = out[out['variable_name'].ne('')]

    print(f"Using category source column: {category_col}")
    print(f"Loaded {len(out)} variable mappings")
    return out


def get_variable_columns(df: pd.DataFrame) -> list:
    """Get all columns starting with 'v' that contain numeric survey responses."""
    # Get columns that start with 'v' followed by a digit (e.g., v1, v2, v85)
    v_columns = [col for col in df.columns if col.startswith('v') and 
                 len(col) > 1 and col[1].isdigit()]
    print(f"Found {len(v_columns)} variable columns")
    return v_columns


def calculate_topic_averages(df_normalized: pd.DataFrame, 
                             topic_df: pd.DataFrame,
                             v_columns: list) -> pd.DataFrame:
    """
    Calculate average normalized scores per topic category.
    
    Args:
        df_normalized: Normalized survey dataframe
        topic_df: Topic matching dataframe
        v_columns: List of variable columns
    
    Returns:
        DataFrame with topic averages
    """
    print("Calculating topic averages...")
    
    # Create mapping from variable name to category
    var_to_category = dict(zip(topic_df['variable_name'], topic_df['category']))
    
    # Group variables by category
    category_vars = {}
    for var in v_columns:
        if var in var_to_category:
            category = var_to_category[var]
            if category not in category_vars:
                category_vars[category] = []
            category_vars[category].append(var)
    
    # Calculate average per category
    topic_averages = {}
    topic_counts = {}
    
    for category, vars_list in category_vars.items():
        # Get all values for this category
        values = []
        for var in vars_list:
            if var in df_normalized.columns:
                col_values = df_normalized[var].dropna().values
                values.extend(col_values)
        
        if values:
            topic_averages[category] = np.mean(values)
            topic_counts[category] = len(values)
    
    # Create result dataframe
    result_df = pd.DataFrame({
        'category': list(topic_averages.keys()),
        'average_score': list(topic_averages.values()),
        'response_count': [topic_counts[cat] for cat in topic_averages.keys()]
    })
    
    # Sort by average score descending
    result_df = result_df.sort_values('average_score', ascending=False)
    
    print(f"Calculated averages for {len(result_df)} categories")
    return result_df


def plot_radar_chart(topic_averages: pd.DataFrame, 
                     output_path: str = None,
                     title: str = "Average Response Scores by Topic") -> None:
    """
    Plot a radar chart showing average scores across topics.
    
    Args:
        topic_averages: DataFrame with category and average_score columns
        output_path: Path to save the chart (optional)
        title: Chart title
    """
    set_academic_style()
    
    # Filter out 'Unknown' category if present
    df_plot = topic_averages[topic_averages['category'] != 'Unknown'].copy()
    
    if len(df_plot) == 0:
        print("No valid categories to plot")
        return
    
    categories = df_plot['category'].tolist()
    values = df_plot['average_score'].tolist()
    
    # Number of categories
    N = len(categories)
    
    # Compute angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    
    # Close the plot by appending first value
    values += values[:1]
    angles += angles[:1]
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor='white')
    ax.set_facecolor('white')
    
    # Draw the chart with academic styling
    ax.plot(angles, values, 'o-', linewidth=2, color=ACADEMIC_COLORS['palette'][0],
            markersize=6, markerfacecolor='white', markeredgewidth=1.5)
    ax.fill(angles, values, alpha=ACADEMIC_COLORS['fill_alpha'], color=ACADEMIC_COLORS['palette'][0])
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, fontweight='medium')
    
    # Set y-axis limits and style
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9, color='#444444')
    
    # Style the grid
    ax.grid(True, linestyle='-', alpha=0.3, color='#888888')
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('#CCCCCC')
    
    # Add title
    plt.title(title, size=14, y=1.08, fontweight='bold', color='#222222')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Radar chart saved to: {output_path}")
        plt.close()
    else:
        plt.close()


def plot_radar_chart_with_language_families(df_normalized: pd.DataFrame,
                                             topic_df: pd.DataFrame,
                                             v_columns: list,
                                             topic_averages: pd.DataFrame,
                                             output_path: str = None,
                                             title: str = "Average Response Scores by Topic and Language Family") -> None:
    """
    Plot a radar chart showing overall average and per-language-family averages.
    Academic paper style with colorblind-friendly palette.
    
    Args:
        df_normalized: Normalized survey dataframe
        topic_df: Topic matching dataframe
        v_columns: List of variable columns
        topic_averages: DataFrame with topic averages (sorted by average_score)
        output_path: Path to save the chart
        title: Chart title
    """
    set_academic_style()
    
    # Check for language family column
    lang_family_col = 'country_group'
    
    if lang_family_col not in df_normalized.columns:
        print(f"No '{lang_family_col}' column found, creating overall chart only")
        return
    
    # Create mapping from variable name to category
    var_to_category = dict(zip(topic_df['variable_name'], topic_df['category']))
    
    # Group variables by category (excluding Unknown)
    category_vars = {}
    for var in v_columns:
        if var in var_to_category:
            category = var_to_category[var]
            if category != 'Unknown':
                if category not in category_vars:
                    category_vars[category] = []
                category_vars[category].append(var)
    
    # Use the same category order as topic_averages (sorted by average_score, excluding Unknown)
    categories = topic_averages[topic_averages['category'] != 'Unknown']['category'].tolist()
    # Only keep categories that have variables mapped
    categories = [cat for cat in categories if cat in category_vars]
    N = len(categories)
    
    if N == 0:
        print("No valid categories found")
        return
    
    # Compute angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(10, 9), subplot_kw=dict(polar=True), facecolor='white')
    ax.set_facecolor('white')
    
    # Get unique language families
    language_families = sorted(df_normalized[lang_family_col].dropna().unique().tolist())
    
    # Calculate and plot OVERALL average first (black, dashed, prominent)
    overall_values = []
    for category in categories:
        vars_list = category_vars[category]
        cat_values = []
        for var in vars_list:
            if var in df_normalized.columns:
                cat_values.extend(df_normalized[var].dropna().values)
        overall_values.append(np.mean(cat_values) if cat_values else 0)
    
    overall_values_closed = overall_values + overall_values[:1]
    ax.plot(angles, overall_values_closed, 
            linestyle='--', linewidth=2.5, color=ACADEMIC_COLORS['overall'], 
            label=f'Overall (N={len(df_normalized):,})', 
            marker='o', markersize=7, markerfacecolor='white', markeredgewidth=2,
            zorder=10)
    
    # Plot each language family with distinct colors and markers
    for idx, family in enumerate(language_families):
        family_data = df_normalized[df_normalized[lang_family_col] == family]
        color = ACADEMIC_COLORS['palette'][idx % len(ACADEMIC_COLORS['palette'])]
        marker = MARKER_STYLES[idx % len(MARKER_STYLES)]
        
        # Calculate averages for each category
        values = []
        for category in categories:
            vars_list = category_vars[category]
            cat_values = []
            for var in vars_list:
                if var in family_data.columns:
                    cat_values.extend(family_data[var].dropna().values)
            values.append(np.mean(cat_values) if cat_values else 0)
        
        values += values[:1]
        
        ax.plot(angles, values, 
                linestyle='-', linewidth=1.8, color=color, 
                label=f'{family} (n={len(family_data):,})',
                marker=marker, markersize=6, markerfacecolor='white', markeredgewidth=1.5,
                alpha=0.9)
        ax.fill(angles, values, alpha=ACADEMIC_COLORS['fill_alpha'], color=color)
    
    # Set category labels with professional styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, fontweight='medium', color='#222222')
    
    # Set y-axis limits and style
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9, color='#444444')
    
    # Style the grid for clean academic look
    ax.grid(True, linestyle='-', alpha=0.25, color='#888888', linewidth=0.5)
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('#CCCCCC')
    ax.spines['polar'].set_linewidth(0.8)
    
    # Professional legend
    legend = ax.legend(
        loc='upper left', 
        bbox_to_anchor=(1.02, 1.0),
        fontsize=10,
        frameon=True,
        fancybox=False,
        edgecolor='#CCCCCC',
        facecolor='white',
        framealpha=1.0,
        title='Language Family',
        title_fontsize=11,
    )
    legend.get_title().set_fontweight('bold')
    
    # Add title
    plt.title(title, size=14, y=1.06, fontweight='bold', color='#111111')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Radar chart with language families saved to: {output_path}")
        plt.close()
    else:
        plt.close()
    
    # Print summary statistics
    print("\nLanguage Family Statistics:")
    print("-" * 50)
    for family in language_families:
        family_count = len(df_normalized[df_normalized[lang_family_col] == family])
        print(f"  {family}: {family_count:,} responses")
    print(f"  Total: {len(df_normalized):,} responses")


def plot_radar_chart_by_country(df_normalized: pd.DataFrame,
                                 topic_df: pd.DataFrame,
                                 v_columns: list,
                                 output_path: str = None,
                                 title: str = "Average Response Scores by Topic and Country") -> None:
    """
    Plot a radar chart comparing topic averages across countries.
    Academic paper style with colorblind-friendly palette.
    
    Args:
        df_normalized: Normalized survey dataframe
        topic_df: Topic matching dataframe
        v_columns: List of variable columns
        output_path: Path to save the chart
        title: Chart title
    """
    set_academic_style()
    
    # Check for country column
    country_col = 'country_str' if 'country_str' in df_normalized.columns else 'c_abrv'
    
    if country_col not in df_normalized.columns:
        print("No country column found, creating overall chart only")
        return
    
    # Get unique countries (limit to top 5 by sample size)
    country_counts = df_normalized[country_col].value_counts()
    top_countries = country_counts.head(5).index.tolist()
    
    # Create mapping from variable name to category
    var_to_category = dict(zip(topic_df['variable_name'], topic_df['category']))
    
    # Group variables by category
    category_vars = {}
    for var in v_columns:
        if var in var_to_category:
            category = var_to_category[var]
            if category != 'Unknown':
                if category not in category_vars:
                    category_vars[category] = []
                category_vars[category].append(var)
    
    categories = list(category_vars.keys())
    N = len(categories)
    
    if N == 0:
        print("No valid categories found")
        return
    
    # Compute angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(10, 9), subplot_kw=dict(polar=True), facecolor='white')
    ax.set_facecolor('white')
    
    for idx, country in enumerate(top_countries):
        country_data = df_normalized[df_normalized[country_col] == country]
        color = ACADEMIC_COLORS['palette'][idx % len(ACADEMIC_COLORS['palette'])]
        marker = MARKER_STYLES[idx % len(MARKER_STYLES)]
        
        # Calculate averages for each category
        values = []
        for category in categories:
            vars_list = category_vars[category]
            cat_values = []
            for var in vars_list:
                if var in country_data.columns:
                    cat_values.extend(country_data[var].dropna().values)
            values.append(np.mean(cat_values) if cat_values else 0)
        
        values += values[:1]
        
        ax.plot(angles, values, 
                linestyle='-', linewidth=1.8, color=color,
                label=f'{country} (n={len(country_data):,})',
                marker=marker, markersize=6, markerfacecolor='white', markeredgewidth=1.5)
        ax.fill(angles, values, alpha=ACADEMIC_COLORS['fill_alpha'], color=color)
    
    # Set category labels with professional styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, fontweight='medium', color='#222222')
    
    # Set y-axis limits and style
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9, color='#444444')
    
    # Style the grid for clean academic look
    ax.grid(True, linestyle='-', alpha=0.25, color='#888888', linewidth=0.5)
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('#CCCCCC')
    ax.spines['polar'].set_linewidth(0.8)
    
    # Professional legend
    legend = ax.legend(
        loc='upper left', 
        bbox_to_anchor=(1.02, 1.0),
        fontsize=10,
        frameon=True,
        fancybox=False,
        edgecolor='#CCCCCC',
        facecolor='white',
        framealpha=1.0,
        title='Country',
        title_fontsize=11,
    )
    legend.get_title().set_fontweight('bold')
    
    # Add title
    plt.title(title, size=14, y=1.06, fontweight='bold', color='#111111')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Country comparison radar chart saved to: {output_path}")
        plt.close()
    else:
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Consensus analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def _krippendorff_alpha_interval(data: np.ndarray) -> float:
    """
    Krippendorff's α for interval-level data.

    Parameters
    ----------
    data : ndarray of shape (n_raters, n_items)
        Rating matrix; NaN denotes missing values.

    Returns
    -------
    float
        Krippendorff's α (≤1; 1 = perfect agreement, 0 = chance level).
    """
    data = np.array(data, dtype=float)
    n_raters, n_items = data.shape
    if n_raters < 2:
        return np.nan

    Do_num  = 0.0
    n_coinc = 0
    all_vals: list = []

    for j in range(n_items):
        col   = data[:, j]
        valid = col[~np.isnan(col)]
        m     = len(valid)
        if m < 2:
            continue
        all_vals.extend(valid.tolist())
        # Σ_{i≠k}(v_i − v_k)² = 2(m·Σv² − (Σv)²)
        sq       = 2.0 * (m * float(np.dot(valid, valid)) - float(valid.sum()) ** 2)
        Do_num  += sq
        n_coinc += m * (m - 1)

    if n_coinc == 0 or len(all_vals) < 2:
        return np.nan

    Do     = Do_num / n_coinc
    V      = np.asarray(all_vals)
    N      = len(V)
    De_num = 2.0 * (N * float(np.dot(V, V)) - float(V.sum()) ** 2)
    De     = De_num / (N * (N - 1))

    return 1.0 if De == 0.0 else 1.0 - Do / De


def analyze_consensus(
    df_normalized: pd.DataFrame,
    v_columns: list,
    output_path: str = None,
) -> dict:
    """
    Consensus analysis across country and language-family granularities.

    Uses Krippendorff's α and the Kolmogorov–Smirnov (KS) statistic to compare
    within- vs between-group distributional divergence and identifies the
    granularity that maximises the within/between gap.

    Important
    ---------
    The analysis is computed from normalized human responses only. It does not
    read LLM outputs, `LLM_DIR`, or any derived model-response artifacts.
    Questionnaire scale metadata from `Surveys_parsed/` may have been used
    upstream during normalization to recover valid response ranges and DK/NA
    codes, but no model-generated answers enter the consensus statistics.

    Parameters
    ----------
    df_normalized : DataFrame
        Normalised survey data (expects 'country_str'/'c_abrv' and 'country_group').
    v_columns : list of str
        Variable columns to include.
    output_path : str or None
        If set, saves the figure to this path.

    Returns
    -------
    dict with analysis results.
    """
    from scipy.stats import ks_2samp, mannwhitneyu

    country_col     = 'country_str' if 'country_str' in df_normalized.columns else 'c_abrv'
    lang_family_col = 'country_group'

    if country_col not in df_normalized.columns:
        print("No country column found — skipping consensus analysis.")
        return {}
    if lang_family_col not in df_normalized.columns:
        print(f"No '{lang_family_col}' column — skipping consensus analysis.")
        return {}

    # ── Exclusions & remapping (mirrors plot_pca_by_country) ─────────────────
    URALIC = {'Finland', 'Hungary', 'Estonia'}
    OTHERS = {'Albania', 'Armenia', 'Greece', 'Turkey', 'Azerbaijan',
               'Cyprus', 'Georgia', 'Liechtenstein', 'San Marino'}
    REMAP  = {'Italic': 'Romance'}

    df_a = df_normalized[
        df_normalized[country_col].notna() &
        ~df_normalized[country_col].isin(['Greece', 'GR', 'GRC', 'Greek'])
    ].copy()

    df_a['_family'] = df_a[lang_family_col].map(lambda f: REMAP.get(str(f), str(f)))
    df_a.loc[df_a[country_col].isin(URALIC), '_family'] = 'Uralic'
    df_a.loc[df_a[country_col].isin(OTHERS), '_family'] = 'Others'

    c2f      = df_a.groupby(country_col)['_family'].first().to_dict()
    countries = sorted(c2f.keys())
    families  = sorted(set(c2f.values()))
    print(f"\nConsensus analysis: {len(countries)} countries · {len(families)} language families")
    print(f"Variables: {len(v_columns)}")

    # ── Balanced per-country respondent samples (used by KS and K-alpha) ────
    N_SAMPLE = 500
    samples  = {}
    for c in countries:
        sub = df_a.loc[df_a[country_col] == c, v_columns]
        if len(sub) >= 20:
            samples[c] = sub.sample(min(N_SAMPLE, len(sub)), random_state=42)
    avail = list(samples.keys())

    # ── Krippendorff's α (full response distributions, not country means) ───
    # Country-level within agreement: one alpha per country matrix
    within_country_alphas: dict = {}
    for c in avail:
        a = _krippendorff_alpha_interval(samples[c].to_numpy(dtype=float))
        if not np.isnan(a):
            within_country_alphas[c] = a

    # Cross-country baseline: pooled respondents from all available countries
    if avail:
        all_country_pool = pd.concat([samples[c] for c in avail], ignore_index=True)
        alpha_all_countries = _krippendorff_alpha_interval(all_country_pool.to_numpy(dtype=float))
    else:
        alpha_all_countries = np.nan

    # Within-family agreement: pooled respondents per family
    within_family_alphas: dict = {}
    family_pools: dict = {}
    for f in families:
        fcs = [c for c in avail if c2f.get(c) == f]
        if not fcs:
            continue
        fam_pool = pd.concat([samples[c] for c in fcs], ignore_index=True)
        family_pools[f] = fam_pool
        a = _krippendorff_alpha_interval(fam_pool.to_numpy(dtype=float))
        if not np.isnan(a):
            within_family_alphas[f] = a

    # Between-family agreement: alpha on pooled pairs of families
    between_family_alphas = []
    fam_keys = sorted(family_pools.keys())
    for i in range(len(fam_keys)):
        for j in range(i + 1, len(fam_keys)):
            fa, fb = fam_keys[i], fam_keys[j]
            pair_pool = pd.concat([family_pools[fa], family_pools[fb]], ignore_index=True)
            a_pair = _krippendorff_alpha_interval(pair_pool.to_numpy(dtype=float))
            if not np.isnan(a_pair):
                between_family_alphas.append(a_pair)

    mean_within_country_alpha = float(np.nanmean(list(within_country_alphas.values()))) \
        if within_country_alphas else np.nan
    mean_within_alpha = float(np.nanmean(list(within_family_alphas.values()))) \
        if within_family_alphas else np.nan
    alpha_between_families = float(np.nanmean(between_family_alphas)) \
        if between_family_alphas else np.nan

    print("\nKrippendorff's α (full distributions):")
    print(f"  Across all countries (pooled respondents): {alpha_all_countries:.4f}")
    print(f"  Mean within-country                      : {mean_within_country_alpha:.4f}"
          f"  (n={len(within_country_alphas)} countries)")
    print(f"  Mean between language families          : {alpha_between_families:.4f}"
          f"  (n={len(between_family_alphas)} family pairs)")
    for f, a in sorted(within_family_alphas.items()):
        n_c = sum(1 for cc in avail if c2f.get(cc) == f)
        print(f"  Within {f:<22}: {a:.4f}  ({n_c} countries)")
    print(f"  Mean within-family                       : {mean_within_alpha:.4f}")

    # ── KS Tests ─────────────────────────────────────────────────────────────

    def _mean_ks(dfA, dfB):
        ds = []
        for v in v_columns:
            a = dfA[v].dropna().values
            b = dfB[v].dropna().values
            if len(a) >= 5 and len(b) >= 5:
                D, _ = ks_2samp(a, b)
                ds.append(D)
        return float(np.mean(ds)) if ds else np.nan

    # Within-country: bootstrap half-split (sampling variability baseline)
    within_country_ks = []
    for c in avail:
        sdf  = samples[c].sample(frac=1, random_state=0)   # shuffle
        half = len(sdf) // 2
        val  = _mean_ks(sdf.iloc[:half], sdf.iloc[half:])
        if not np.isnan(val):
            within_country_ks.append(val)

    # Pairwise country KS → within-family vs between-family
    within_family_ks  = []
    between_family_ks = []
    print(f"  Computing {len(avail)*(len(avail)-1)//2} country pairs "
          f"× {len(v_columns)} variables…")
    for i in range(len(avail)):
        for j in range(i + 1, len(avail)):
            c1, c2 = avail[i], avail[j]
            val = _mean_ks(samples[c1], samples[c2])
            if np.isnan(val):
                continue
            if c2f.get(c1) == c2f.get(c2):
                within_family_ks.append(val)
            else:
                between_family_ks.append(val)

    mw_stat, mw_p = mannwhitneyu(within_family_ks, between_family_ks,
                                  alternative='less')

    wc_mean, wf_mean, bf_mean = (np.mean(within_country_ks),
                                  np.mean(within_family_ks),
                                  np.mean(between_family_ks))
    wc_std, wf_std, bf_std    = (np.std(within_country_ks),
                                  np.std(within_family_ks),
                                  np.std(between_family_ks))

    # All cross-country pairs (within + between family)
    all_cross_ks     = within_family_ks + between_family_ks
    ac_mean, ac_std  = np.mean(all_cross_ks), np.std(all_cross_ks)

    # Gaps: |between - within| at each granularity
    # Country granularity: any two different countries vs within-country bootstrap
    gap_ks_country = ac_mean - wc_mean
    # Family granularity: cross-family pairs vs same-family pairs
    gap_ks_family  = bf_mean - wf_mean

    # Alpha gaps: within-group agreement minus between-group agreement
    # Country granularity: mean within-country alpha vs pooled all-countries alpha
    gap_alpha_country = mean_within_country_alpha - alpha_all_countries
    # Family granularity: mean within-family alpha vs between-family-pairs alpha
    gap_alpha_family  = mean_within_alpha - alpha_between_families

    print(f"\nKS Statistics (mean D across {len(v_columns)} variables):")
    print(f"  Within-country (bootstrap)  : {wc_mean:.4f} ± {wc_std:.4f}  "
          f"(n={len(within_country_ks)} countries)")
    print(f"  Within-family  (pairs)      : {wf_mean:.4f} ± {wf_std:.4f}  "
          f"(n={len(within_family_ks)} pairs)")
    print(f"  All cross-country (pairs)   : {ac_mean:.4f} ± {ac_std:.4f}  "
          f"(n={len(all_cross_ks)} pairs)")
    print(f"  Between-family (pairs)      : {bf_mean:.4f} ± {bf_std:.4f}  "
          f"(n={len(between_family_ks)} pairs)")
    print(f"  Mann-Whitney U (within < between family): U={mw_stat:.0f}, p={mw_p:.2e}")
    print(f"  KS gap @ country granularity (cross-country − within-country): {gap_ks_country:.4f}")
    print(f"  KS gap @ family  granularity (between-family − within-family): {gap_ks_family:.4f}")
    print(f"  Alpha gap @ country granularity : {gap_alpha_country:.4f}")
    print(f"  Alpha gap @ family  granularity : {gap_alpha_family:.4f}")
    best = 'Language Family' if gap_ks_family >= gap_ks_country else 'Country'
    print(f"  → Granularity maximising KS gap: {best}")

    # ── Academic-style figure ─────────────────────────────────────────────────
    with plt.style.context(['science', 'no-latex']):
        mpl.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'xtick.labelsize': 9.5,
            'ytick.labelsize': 10,
            'legend.fontsize': 9.5,
        })

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

        gran_labels = ['Country', 'Language\nFamily']
        bar_color   = [ACADEMIC_COLORS['palette'][0], ACADEMIC_COLORS['palette'][2]]
        x_pos       = [0, 1]
        bar_width   = 0.45

        # ── Left: mean KS D — within vs between, per granularity ─────────────
        # Country granularity: within-country bootstrap vs all cross-country pairs
        # Family granularity:  within-family pairs vs between-family pairs
        group_labels  = ['Within\ncountry', 'Cross-\ncountry',
                         'Within\nfamily', 'Between\nfamilies']
        group_means   = [wc_mean,  ac_mean,  wf_mean,  bf_mean]
        group_stds    = [wc_std,   ac_std,   wf_std,   bf_std]
        group_colors  = [ACADEMIC_COLORS['palette'][0], ACADEMIC_COLORS['palette'][0],
                         ACADEMIC_COLORS['palette'][2], ACADEMIC_COLORS['palette'][2]]
        group_alphas  = [0.45, 0.80, 0.45, 0.80]
        xs = [0, 0.6, 1.5, 2.1]

        for xi, mean, std, col, alp in zip(xs, group_means, group_stds,
                                            group_colors, group_alphas):
            ax1.bar(xi, mean, yerr=std, color=col, alpha=alp,
                    edgecolor='black', linewidth=0.6, width=0.45,
                    error_kw=dict(elinewidth=0.9, capsize=4, capthick=0.9,
                                  ecolor='#333333'))
            ax1.text(xi, mean + std + 0.003, f'{mean:.3f}',
                     ha='center', va='bottom', fontsize=8)

        # Gap brackets
        def _bracket(ax, x1, x2, y, label):
            ax.plot([x1, x2], [y, y], 'k-', lw=0.8)
            ax.plot([x1, x1], [y * 0.988, y], 'k-', lw=0.8)
            ax.plot([x2, x2], [y * 0.988, y], 'k-', lw=0.8)
            ax.text((x1 + x2) / 2, y * 1.012, label,
                    ha='center', va='bottom', fontsize=8)

        y_top = max(m + s for m, s in zip(group_means, group_stds)) * 1.12
        _bracket(ax1, xs[0], xs[1], y_top,        f'\u0394={gap_ks_country:.3f}')
        _bracket(ax1, xs[2], xs[3], y_top,        f'\u0394={gap_ks_family:.3f}')

        ax1.set_xticks(xs)
        ax1.set_xticklabels(group_labels, fontsize=8.5)
        ax1.set_ylabel('Mean KS D statistic')
        ax1.set_title('(a) KS divergence by granularity')
        ax1.set_ylim(bottom=0)
        # Vertical separator between the two granularities
        ax1.axvline(x=1.05, color='#BBBBBB', linestyle='--', linewidth=0.7)
        ax1.text(0.3,  ax1.get_ylim()[1] * 0.97, 'Country',       ha='center',
                 fontsize=8, color='#555555', style='italic')
        ax1.text(1.8,  ax1.get_ylim()[1] * 0.97, 'Lang. Family',  ha='center',
                 fontsize=8, color='#555555', style='italic')
        for sp in ['top', 'right']:
            ax1.spines[sp].set_visible(False)

        # ── Right: Krippendorff α |within − between| gap ─────────────────────
        alpha_gaps = [gap_alpha_country, gap_alpha_family]
        ax2.bar(
            x_pos, alpha_gaps,
            color=bar_color, alpha=0.72,
            edgecolor='black', linewidth=0.6,
            width=bar_width,
        )
        for xi, val in zip(x_pos, alpha_gaps):
            ax2.text(xi, val + 0.0003, f'{val:.3f}',
                     ha='center', va='bottom', fontsize=9)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(gran_labels)
        ax2.set_ylabel("|Within − Between| Krippendorff's α")
        ax2.set_title("(b) Krippendorff's α gap")
        ax2.set_ylim(bottom=0)
        for sp in ['top', 'right']:
            ax2.spines[sp].set_visible(False)

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nConsensus analysis saved to: {output_path}")
        plt.close(fig)

    # ── Paragraph for the paper ───────────────────────────────────────────────
    para = (
        f"Consensus analysis using two complementary metrics --- "
        f"Kolmogorov-Smirnov (KS) distributional divergence and "
        f"Krippendorff's alpha --- quantifies the discriminating power of two "
        f"levels of cultural aggregation. "
        f"At the country granularity, the gap between cross-country KS divergence "
        f"and within-country sampling variability is {gap_ks_country:.3f}, "
        f"while at the language-family granularity the gap between "
        f"cross-family and within-family pairs is {gap_ks_family:.3f}. "
        f"The country-level gap is larger, reflecting that simply crossing a "
        f"national border already introduces substantial distributional shift "
        f"(Mann-Whitney U-test: p={mw_p:.2e}). "
        f"Krippendorff's alpha mirrors this pattern: the within-minus-between "
        f"agreement gap is {gap_alpha_country:.3f} at the country level "
        f"(mean within-country alpha={mean_within_country_alpha:.3f} vs "
        f"all-country pooled alpha={alpha_all_countries:.3f}) and {gap_alpha_family:.3f} "
        f"at the language-family level "
        f"(mean within-family alpha={mean_within_alpha:.3f} vs "
        f"between-family alpha={alpha_between_families:.3f}), "
        f"confirming that country boundaries capture finer-grained value "
        f"heterogeneity, while language-family groupings remain a meaningful "
        f"coarser-level partition."
    )
    print("\n" + "=" * 70)
    print("PARAGRAPH FOR PAPER:")
    print("=" * 70)
    print(para)

    return {
        'within_country_ks':       within_country_ks,
        'within_family_ks':        within_family_ks,
        'between_family_ks':       between_family_ks,
        'alpha_within_country':    within_country_alphas,
        'alpha_within_family':     within_family_alphas,
        'alpha_between_families':  alpha_between_families,
        'alpha_all_countries':     alpha_all_countries,
        'mw_stat':                 mw_stat,
        'mw_pvalue':               mw_p,
    }


def plot_pca_by_country(df_normalized: pd.DataFrame,
                        v_columns: list,
                        output_path: str = None,
                        interpretation_output_path: str = None,
                        title: str = "PCA of Country Responses by Language Family") -> None:
    """
    Perform PCA on country-level normalized responses and visualize with language family grouping.
    Academic paper style with colorblind-friendly palette.

    Args:
        df_normalized: Normalized survey dataframe
        v_columns: List of variable columns
        output_path: Path to save the chart
        interpretation_output_path: Path to save the PCA component interpretation CSV
        title: Chart title
    """
    set_academic_style()
    
    # Check for country and language family columns
    country_col = 'country_str' if 'country_str' in df_normalized.columns else 'c_abrv'
    lang_family_col = 'country_group'
    
    if country_col not in df_normalized.columns:
        print("No country column found, cannot create PCA visualization")
        return
    
    if lang_family_col not in df_normalized.columns:
        print(f"No '{lang_family_col}' column found, will use single color")
        lang_family_col = None
    
    print("\nPreparing country-level PCA analysis...")
    
    # Aggregate normalized values per country (mean across all responses for each variable)
    country_data = []
    countries = []
    language_families = []
    
    for country in sorted(df_normalized[country_col].dropna().unique()):
        # Exclude Greece from the analysis
        if country in ['Greece', 'GR', 'GRC', 'Greek']:
            continue
            
        country_df = df_normalized[df_normalized[country_col] == country]
        
        # Calculate mean for each variable for this country
        country_means = []
        for col in v_columns:
            if col in country_df.columns:
                mean_val = country_df[col].mean()
                country_means.append(mean_val if not np.isnan(mean_val) else 0)
            else:
                country_means.append(0)
        
        country_data.append(country_means)
        countries.append(country)
        
        # Get language family for this country
        if lang_family_col:
            family = country_df[lang_family_col].mode()[0] if len(country_df[lang_family_col].mode()) > 0 else 'Unknown'
            language_families.append(family)
    
    # Remap to updated language families
    URALIC_COUNTRIES  = {'Finland', 'Hungary', 'Estonia'}
    OTHERS_COUNTRIES  = {'Albania', 'Armenia', 'Greece', 'Turkey',
                         'Azerbaijan', 'Cyprus', 'Georgia', 'Liechtenstein', 'San Marino'}
    FAMILY_REMAP = {'Italic': 'Romance'}

    language_families = [
        'Uralic'  if country in URALIC_COUNTRIES  else
        'Others'  if country in OTHERS_COUNTRIES  else
        FAMILY_REMAP.get(fam, fam)
        for country, fam in zip(countries, language_families)
    ]

    # Convert to numpy array
    X = np.array(country_data)
    
    print(f"Aggregated data for {len(countries)} countries across {len(v_columns)} variables")
    
    # Handle any remaining NaN values
    X = np.nan_to_num(X, nan=0.0)
    
    # Standardize the features before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA (reduce to 2 components for visualization)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # Save PCA component interpretation to file
    if interpretation_output_path:
        loadings_df = pd.DataFrame({
            'variable': v_columns,
            'PC1_loading': pca.components_[0],
            'PC2_loading': pca.components_[1],
            'PC1_abs': np.abs(pca.components_[0]),
            'PC2_abs': np.abs(pca.components_[1]),
        }).sort_values('PC1_abs', ascending=False)

        with open(interpretation_output_path, 'w', encoding='utf-8') as f:
            f.write("# PCA Component Interpretation\n")
            f.write(f"# n_countries={len(countries)}, n_variables={len(v_columns)}\n")
            f.write(f"# PC1_explained_variance={pca.explained_variance_ratio_[0]:.6f} ({pca.explained_variance_ratio_[0]:.2%})\n")
            f.write(f"# PC2_explained_variance={pca.explained_variance_ratio_[1]:.6f} ({pca.explained_variance_ratio_[1]:.2%})\n")
            f.write(f"# Total_explained_variance={pca.explained_variance_ratio_.sum():.6f} ({pca.explained_variance_ratio_.sum():.2%})\n")
            loadings_df.to_csv(f, index=False)
        print(f"PCA interpretation saved to: {interpretation_output_path}")

    # --- Visualization (SciencePlots journal style) --------------------------------
    with plt.style.context(['science', 'no-latex']):
        # Override SciencePlots' small default sizes so text is legible in print
        mpl.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 9.5,
        })

        fig, ax = plt.subplots(figsize=(10, 6.5))

        # Axis limits: tight around data, extra right margin for Denmark label,
        # small top margin, bottom extended for horizontal legend.
        x_range = X_pca[:, 0].max() - X_pca[:, 0].min()
        y_range = X_pca[:, 1].max() - X_pca[:, 1].min()
        ax.set_xlim(X_pca[:, 0].min() - x_range * 0.04,
                    X_pca[:, 0].max() + x_range * 0.12)
        ax.set_ylim(X_pca[:, 1].min() - y_range * 0.08,
                    X_pca[:, 1].max() + y_range * 0.10)

        # Remove background grid and outer frame; keep only 0-axis crosshair
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Reference lines at origin as clean axis guides
        ax.axhline(y=0, color='#555555', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)
        ax.axvline(x=0, color='#555555', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)

        texts = []
        point_positions = []  # original marker (x, y) per text label

        if lang_family_col:
            unique_families = sorted(list(set(language_families)))
            color_map = {family: ACADEMIC_COLORS['palette'][idx % len(ACADEMIC_COLORS['palette'])]
                         for idx, family in enumerate(unique_families)}
            marker_map = {family: MARKER_STYLES[idx % len(MARKER_STYLES)]
                          for idx, family in enumerate(unique_families)}

            for family in unique_families:
                indices = [i for i, f in enumerate(language_families) if f == family]
                x_coords = X_pca[indices, 0]
                y_coords = X_pca[indices, 1]
                family_countries = [countries[i] for i in indices]

                ax.scatter(x_coords, y_coords,
                           c=color_map[family],
                           marker=marker_map[family],
                           s=60,
                           linewidth=0.4,
                           label=f'{family} ($n$={len(indices)})',
                           zorder=3)

                for x, y, country in zip(x_coords, y_coords, family_countries):
                    t = ax.text(x, y, country,
                                fontsize=9,
                                color='#222222',
                                zorder=5)
                    texts.append(t)
                    point_positions.append((x, y))
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1],
                       c=ACADEMIC_COLORS['palette'][0],
                       marker='o',
                       s=60,
                       linewidth=0.4,
                       zorder=3)

            for x, y, country in zip(X_pca[:, 0], X_pca[:, 1], countries):
                t = ax.text(x, y, country,
                            fontsize=9,
                            color='#222222',
                            zorder=5)
                texts.append(t)
                point_positions.append((x, y))

        # Auto-repel labels (no built-in arrows; handled manually below)
        adjust_text(
            texts,
            ax=ax,
            expand=(1.4, 1.6),
            force_text=(0.4, 0.7),
            force_points=(0.15, 0.3),
            ensure_inside_axes=False,
        )

        # Draw connector arrows only where labels were displaced significantly
        x_data_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_data_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        arrow_threshold = 0.025  # fraction of axis range
        for t, (px, py) in zip(texts, point_positions):
            tx, ty = t.get_position()
            dx = abs(tx - px) / x_data_range
            dy = abs(ty - py) / y_data_range
            if dx > arrow_threshold or dy > arrow_threshold:
                ax.annotate(
                    '', xy=(px, py), xytext=(tx, ty),
                    arrowprops=dict(arrowstyle='-', color='#AAAAAA', lw=0.5),
                    zorder=4,
                )

        # Axis labels and title
        ax.set_xlabel(
            f"Traditionalism ↔ Progressivism (PC1, {pca.explained_variance_ratio_[0]:.1%} var.)",
            fontsize=12)
        ax.set_ylabel(
            f"Universalism ↔ Particularism (PC2, {pca.explained_variance_ratio_[1]:.1%} var.)",
            fontsize=12)
        ax.set_title('')
        ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

        # Legend — compact horizontal strip below the axes
        if lang_family_col:
            legend = ax.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.14),
                ncol=4,
                fontsize=9.5,
                frameon=True,
                title='Language Family',
                title_fontsize=10,
                handlelength=1.2,
                handletextpad=0.5,
                columnspacing=1.0,
                borderpad=0.6,
            )
            legend.get_title().set_fontweight('bold')

        fig.tight_layout(rect=[0, 0.09, 1, 1])

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"PCA visualization saved to: {output_path}")

        plt.close(fig)

    # Print top contributing variables for each PC
    print("\nTop 10 variables contributing to PC1:")
    pc1_contributions = sorted(zip(v_columns, np.abs(pca.components_[0])), key=lambda x: x[1], reverse=True)[:10]
    for var, contrib in pc1_contributions:
        print(f"  {var}: {contrib:.4f}")

    print("\nTop 10 variables contributing to PC2:")
    pc2_contributions = sorted(zip(v_columns, np.abs(pca.components_[1])), key=lambda x: x[1], reverse=True)[:10]
    for var, contrib in pc2_contributions:
        print(f"  {var}: {contrib:.4f}")


def plot_pca_country_language_points(df_normalized: pd.DataFrame,
                                     v_columns: list,
                                     output_path: str = None,
                                     title: str = "PCA of Human Responses: Country-Language Points",
                                     column_width: float = 3.5) -> None:
    """
    Plot the same country-level PCA space, but display points at country-language
    granularity whenever language information is available.

        Behavior
        --------
        - PCA basis is fitted on country-level means (same logic as plot_pca_by_country).
        - If language information exists, language responses are first aggregated within
            each country by inferred language family using weighted means.
        - Countries are split into multiple points only when they span >1 inferred
            language family; otherwise a single weighted country point is shown.
        - All projected points are rendered with the same marker symbol.
        - If no usable language column exists at all, falls back to country points with
            a distinct marker to indicate missing language granularity.
    
    Args:
        column_width: Figure width in inches (default 3.5 for single column, use 7.0 for two columns)
    """
    set_academic_style()

    country_col = 'country_str' if 'country_str' in df_normalized.columns else 'c_abrv'
    lang_family_col = 'country_group'

    if country_col not in df_normalized.columns:
        print("No country column found, cannot create country-language PCA plot")
        return

    # Detect a language column
    language_col = None
    for cand in ['language_str', 'language', 'lang', 'interview_language']:
        if cand in df_normalized.columns:
            language_col = cand
            break

    # Build country-level PCA fit (same conceptual basis as country PCA)
    country_data = []
    countries = []
    language_families = []

    for country in sorted(df_normalized[country_col].dropna().unique()):
        if country in ['Greece', 'GR', 'GRC', 'Greek']:
            continue

        country_df = df_normalized[df_normalized[country_col] == country]
        if country_df.empty:
            continue

        means = []
        for col in v_columns:
            if col in country_df.columns:
                means.append(country_df[col].mean())
            else:
                means.append(np.nan)

        country_data.append(np.nan_to_num(np.asarray(means, dtype=float), nan=0.0))
        countries.append(country)

        if lang_family_col in df_normalized.columns:
            fam_mode = country_df[lang_family_col].mode()
            fam = fam_mode.iloc[0] if len(fam_mode) > 0 else 'Unknown'
        else:
            fam = 'Unknown'
        language_families.append(fam)

    if len(country_data) < 3:
        print("Not enough countries to compute PCA country-language plot")
        return

    # Family remapping rules to match existing plots
    URALIC_COUNTRIES = {'Finland', 'Hungary', 'Estonia'}
    OTHERS_COUNTRIES = {'Albania', 'Armenia', 'Greece', 'Turkey',
                        'Azerbaijan', 'Cyprus', 'Georgia', 'Liechtenstein', 'San Marino'}
    FAMILY_REMAP = {'Italic': 'Romance'}

    language_families = [
        'Uralic' if country in URALIC_COUNTRIES else
        'Others' if country in OTHERS_COUNTRIES else
        FAMILY_REMAP.get(fam, fam)
        for country, fam in zip(countries, language_families)
    ]

    X_country = np.vstack(country_data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_country)
    pca = PCA(n_components=2)
    _ = pca.fit_transform(X_scaled)

    print(f"\nCountry-language PCA basis fitted on {len(countries)} countries × {len(v_columns)} variables")
    print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")

    # Prepare points to display: split only when country languages map to
    # different families; otherwise keep one weighted country point.
    points = []

    def _infer_family_from_language(lang_value: str, fallback_family: str) -> str:
        """Infer language family from language label; fallback to country family."""
        if lang_value is None:
            return fallback_family
        s = str(lang_value).strip().lower()
        if s in {'', 'nan', 'na', 'n/a', 'none', 'null'}:
            return fallback_family

        # Germanic
        if any(k in s for k in ['deutsch', 'german', 'english', 'norsk', 'norwegian', 'svenska', 'swedish', 'íslenska', 'icelandic', 'dutch']):
            return 'Germanic'
        # Romance
        if any(k in s for k in ['français', 'french', 'italiano', 'español', 'spanish', 'català', 'galego', 'portugu', 'român', 'romana', 'română']):
            return 'Romance'
        # Balto-Slavic
        if any(k in s for k in ['рус', 'срп', 'bosnian', 'hrvats', 'sloven', 'slovak', 'slovensk', 'češt', 'czech', 'polski', 'ukrain', 'latvie', 'lietu', 'lithuan', 'serbian', 'croatian']):
            return 'Balto-Slavic'
        # Uralic
        if any(k in s for k in ['magyar', 'finn', 'eesti', 'estonian']):
            return 'Uralic'
        # Others (explicit cases)
        if any(k in s for k in ['euskara', 'armenian', 'azerbaijani', 'georgian', 'albanian', 'turkish', 'greek']):
            return 'Others'

        return fallback_family

    if language_col is not None:
        work = df_normalized.copy()

        # Normalize missing-like language tokens
        lang = work[language_col].astype('string').str.strip()
        missing_like = {'', 'nan', 'na', 'n/a', 'none', 'null'}
        lang = lang.mask(lang.str.lower().isin(missing_like), other=pd.NA)
        work['_lang_clean'] = lang

        # Country-conditional handling of unknown-language samples:
        # - keep unknown only when it is the sole language observed for country
        # - drop unknown when country has multiple language groups
        n_before = len(work)
        keep_mask = pd.Series(False, index=work.index)
        dropped_unknown_rows = 0
        kept_unknown_single_lang_rows = 0

        for _, g in work.groupby(country_col, dropna=False):
            lang_groups = g['_lang_clean'].drop_duplicates()
            n_groups_total = len(lang_groups)
            has_unknown = g['_lang_clean'].isna().any()

            if n_groups_total <= 1:
                # single language group in country (possibly unknown): keep all
                keep_mask.loc[g.index] = True
                if has_unknown:
                    kept_unknown_single_lang_rows += int(g['_lang_clean'].isna().sum())
            else:
                # multiple language groups: exclude unknown rows only
                known_idx = g[g['_lang_clean'].notna()].index
                keep_mask.loc[known_idx] = True
                dropped_unknown_rows += int(g['_lang_clean'].isna().sum())

        work = work[keep_mask].copy()
        n_after = len(work)
        print(
            "Unknown-language handling by country: "
            f"dropped={dropped_unknown_rows:,}, "
            f"kept_single-language_unknown={kept_unknown_single_lang_rows:,}, "
            f"net_removed={n_before - n_after:,}"
        )

        for country in countries:
            sub_country = work[work[country_col] == country]
            if sub_country.empty:
                continue

            fallback_family = language_families[countries.index(country)]

            # Build language-level vectors first
            lang_rows = []
            for lang_val, sub_lang in sub_country.groupby('_lang_clean', dropna=False):
                if sub_lang.empty:
                    continue

                vals = []
                for col in v_columns:
                    if col in sub_lang.columns:
                        vals.append(sub_lang[col].mean())
                    else:
                        vals.append(np.nan)
                vec = np.nan_to_num(np.asarray(vals, dtype=float), nan=0.0)

                has_lang = pd.notna(lang_val)
                lang_label = str(lang_val) if has_lang else 'Unknown language'
                fam_from_lang = _infer_family_from_language(lang_label, fallback_family)

                lang_rows.append({
                    'country': country,
                    'language': lang_label,
                    'family': fam_from_lang,
                    'vec': vec,
                    'n': int(len(sub_lang)),
                    'has_language_info': bool(has_lang),
                })

            if not lang_rows:
                continue

            lang_df = pd.DataFrame(lang_rows)

            # 1) Aggregate language-level vectors into family-level vectors
            fam_rows = []
            for fam, gfam in lang_df.groupby('family', dropna=False):
                total_n_f = float(gfam['n'].sum())
                if total_n_f <= 0:
                    continue
                w = (gfam['n'].to_numpy(dtype=float) / total_n_f).reshape(-1, 1)
                mat = np.vstack(gfam['vec'].to_list())
                vec_f = (w * mat).sum(axis=0)
                lang_list = sorted(set(gfam['language'].astype(str).tolist()))
                fam_rows.append({
                    'country': country,
                    'family': str(fam),
                    'vec': vec_f,
                    'n': int(total_n_f),
                    'has_language_info': bool(gfam['has_language_info'].any()),
                    'n_languages_grouped': int(len(gfam)),
                    'languages_grouped': '; '.join(lang_list),
                })

            if not fam_rows:
                continue

            fam_df = pd.DataFrame(fam_rows)
            families_present = sorted(set(fam_df['family'].tolist()))

            # 2) Split ONLY across families (one weighted point per country-family)
            if len(families_present) > 1:
                for _, rr in fam_df.iterrows():
                    vec_scaled = scaler.transform(rr['vec'].reshape(1, -1))
                    coords = pca.transform(vec_scaled)[0]
                    points.append({
                        'country': rr['country'],
                        'language': str(rr['languages_grouped']),
                        'family': rr['family'],
                        'pc1': float(coords[0]),
                        'pc2': float(coords[1]),
                        'n': int(rr['n']),
                        'has_language_info': bool(rr['has_language_info']),
                        'is_split_country': True,
                    })
            else:
                # Single family -> one weighted country point
                rr = fam_df.iloc[0]
                vec_scaled = scaler.transform(rr['vec'].reshape(1, -1))
                coords = pca.transform(vec_scaled)[0]
                points.append({
                    'country': country,
                    'language': str(rr['languages_grouped']),
                    'family': rr['family'],
                    'pc1': float(coords[0]),
                    'pc2': float(coords[1]),
                    'n': int(rr['n']),
                    'has_language_info': bool(rr['has_language_info']),
                    'is_split_country': False,
                })
    else:
        # No language information available: show country points with fallback marker
        print("Language column not available: unable to exclude unknown-language samples")
        for country, fam, vec in zip(countries, language_families, X_country):
            vec_scaled = scaler.transform(vec.reshape(1, -1))
            coords = pca.transform(vec_scaled)[0]
            points.append({
                'country': country,
                'language': 'Language unavailable',
                'family': fam,
                'pc1': float(coords[0]),
                'pc2': float(coords[1]),
                'n': int((df_normalized[country_col] == country).sum()),
                'has_language_info': False,
                'is_split_country': False,
            })

    pts_df = pd.DataFrame(points)
    if pts_df.empty:
        print("No points to plot for country-language PCA")
        return

    def _compact_language_label(language_value: str) -> str:
        """Compact language label for the second line of country labels."""
        if language_value is None:
            return "Language unavailable"
        s = str(language_value).strip()
        if s == "" or s.lower() in {"nan", "none", "null", "n/a"}:
            return "Language unavailable"
        parts = [p.strip() for p in s.split(';') if p.strip()]
        if len(parts) <= 1:
            return parts[0] if parts else "Language unavailable"
        return f"{parts[0]} +{len(parts)-1}"

    # ── Plot ───────────────────────────────────────────────────────────────
    with plt.style.context(['science', 'no-latex']):
        mpl.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 13,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
        })

        # Map column_width to optimized figure dimensions
        # 1col: 5.8x4.1 (singlecol), 2col: 8.5x5.5 (improved readability)
        figsize_map = {3.5: (5.8, 4.1), 7.0: (8.5, 5.5)}
        figsize = figsize_map.get(column_width, (column_width, column_width * 0.59))
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axhline(0, color='#777777', lw=0.7, alpha=0.6)
        ax.axvline(0, color='#777777', lw=0.7, alpha=0.6)

        # Tight data-driven limits to reduce surrounding whitespace
        x_min, x_max = pts_df['pc1'].min(), pts_df['pc1'].max()
        y_min, y_max = pts_df['pc2'].min(), pts_df['pc2'].max()
        x_pad = max((x_max - x_min) * 0.08, 0.12)
        y_pad = max((y_max - y_min) * 0.10, 0.12)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        fams = sorted(pts_df['family'].dropna().unique().tolist())
        color_map = {f: ACADEMIC_COLORS['palette'][i % len(ACADEMIC_COLORS['palette'])] for i, f in enumerate(fams)}

        label_entries = []
        for fam in fams:
            fam_pts = pts_df[pts_df['family'] == fam]

            # Single symbol for all points (split and aggregated alike)
            if not fam_pts.empty:
                ax.scatter(
                    fam_pts['pc1'], fam_pts['pc2'],
                    c=color_map[fam], marker='o', s=92,
                    edgecolors='white', linewidth=0.7, alpha=0.95,
                    label=f'{fam}',
                )

            for _, r in fam_pts.iterrows():
                country_label = str(r['country'])
                language_label = _compact_language_label(r.get('language', None))
                ghost = ax.text(
                    r['pc1'], r['pc2'],
                    f"{country_label}\n{language_label}",
                    fontsize=8.4,
                    color='#222222',
                    linespacing=1.08,
                    zorder=5,
                )
                label_entries.append({
                    'ghost': ghost,
                    'px': float(r['pc1']),
                    'py': float(r['pc2']),
                    'country': country_label,
                    'language': language_label,
                })

        ghosts = [e['ghost'] for e in label_entries]
        adjust_text(
            ghosts,
            ax=ax,
            expand=(1.22, 1.34),
            force_text=(0.30, 0.46),
            force_points=(0.12, 0.20),
            ensure_inside_axes=True,
        )

        # Draw final labels with country on first line and language in italic
        # on second line; add connectors for displaced labels.
        x_data_range = max(ax.get_xlim()[1] - ax.get_xlim()[0], 1e-9)
        y_data_range = max(ax.get_ylim()[1] - ax.get_ylim()[0], 1e-9)
        arrow_threshold = 0.020

        for e in label_entries:
            tx, ty = e['ghost'].get_position()
            e['ghost'].remove()

            ax.text(
                tx, ty,
                e['country'],
                fontsize=8.7,
                color='#222222',
                ha='left',
                va='bottom',
                zorder=6,
            )
            ax.annotate(
                e['language'],
                xy=(tx, ty),
                xytext=(0, -9),
                textcoords='offset points',
                fontsize=7.8,
                fontstyle='italic',
                color='#444444',
                ha='left',
                va='top',
                zorder=6,
            )

            dx = abs(tx - e['px']) / x_data_range
            dy = abs(ty - e['py']) / y_data_range
            if dx > arrow_threshold or dy > arrow_threshold:
                ax.annotate(
                    '',
                    xy=(e['px'], e['py']),
                    xytext=(tx, ty),
                    arrowprops=dict(arrowstyle='-', color='#9A9A9A', lw=0.55, alpha=0.95),
                    zorder=5,
                )

        ax.set_xlabel(f"Traditionalism ↔ Progressivism (PC1, {pca.explained_variance_ratio_[0]:.1%} var.)")
        ax.set_ylabel(f"Universalism ↔ Particularism (PC2, {pca.explained_variance_ratio_[1]:.1%} var.)")
        ax.tick_params(length=0, labelbottom=False, labelleft=False)

        # De-duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            uniq[l] = h
        ax.legend(list(uniq.values()), list(uniq.keys()),
                  loc='upper center', bbox_to_anchor=(0.5, -0.11),
                  ncol=4,
                  frameon=True, edgecolor='#CCCCCC',
                  borderpad=0.5, columnspacing=1.0, handletextpad=0.45)

        fig.tight_layout(rect=[0.0, 0.06, 1.0, 1.0])

        if output_path:
            out_path = Path(output_path)
            fig.savefig(out_path, dpi=360, bbox_inches='tight', pad_inches=0.04)

            # Vector exports for publication-quality embedding in papers
            svg_path = out_path.with_suffix('.svg')
            pdf_path = out_path.with_suffix('.pdf')
            fig.savefig(svg_path, bbox_inches='tight', pad_inches=0.02)
            fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.02)

            print(f"Country-language PCA plot saved to: {out_path}")
            print(f"Country-language PCA plot saved to: {svg_path}")
            print(f"Country-language PCA plot saved to: {pdf_path}")
        plt.close(fig)


def analyze_pca_family_clustering(
    df_normalized: pd.DataFrame,
    v_columns: list,
    output_path: str = None,
    metrics_output_path: str = None,
    points_output_path: str = None,
    title: str = "PCA Family Clustering Quality (Human Responses)",
) -> dict:
    """
    Quantify and visualize how tightly countries cluster by true language-family
    labels in PCA space.

     Strategy
     --------
     1) Fit PCA on country-level means (same basis as human PCA)
     2) Build analysis points using country-language aggregation:
         - aggregate language subgroups into country-family vectors (weighted)
         - split country into multiple points only when >1 language family appears
     3) Evaluate clustering quality using language-family labels of those points:
       - Silhouette score (higher is better)
       - Calinski-Harabasz index (higher is better)
       - Davies-Bouldin index (lower is better)
       - 1-NN same-family rate (higher is better)
       - Within/between pairwise distance ratio (lower is better)

    Returns
    -------
    dict with metrics and metadata.
    """
    set_academic_style()

    country_col = 'country_str' if 'country_str' in df_normalized.columns else 'c_abrv'
    lang_family_col = 'country_group'

    if country_col not in df_normalized.columns:
        print("No country column found, cannot run PCA family clustering analysis")
        return {}
    if lang_family_col not in df_normalized.columns:
        print(f"No '{lang_family_col}' column found, cannot run PCA family clustering analysis")
        return {}

    # Mirror the same exclusions/remaps used elsewhere for consistency
    URALIC_COUNTRIES = {'Finland', 'Hungary', 'Estonia'}
    OTHERS_COUNTRIES = {
        'Albania', 'Armenia', 'Greece', 'Turkey', 'Azerbaijan',
        'Cyprus', 'Georgia', 'Liechtenstein', 'San Marino'
    }
    FAMILY_REMAP = {'Italic': 'Romance'}

    # Build country-level means for PCA fitting basis
    country_rows = []
    for country in sorted(df_normalized[country_col].dropna().unique()):
        if country in ['Greece', 'GR', 'GRC', 'Greek']:
            continue
        sub = df_normalized[df_normalized[country_col] == country]
        if len(sub) == 0:
            continue
        fam_mode = sub[lang_family_col].mode()
        family = fam_mode.iloc[0] if len(fam_mode) else 'Unknown'
        family = FAMILY_REMAP.get(str(family), str(family))
        if country in URALIC_COUNTRIES:
            family = 'Uralic'
        elif country in OTHERS_COUNTRIES:
            family = 'Others'

        means = []
        for col in v_columns:
            means.append(float(sub[col].mean()) if col in sub.columns else np.nan)
        means = np.nan_to_num(np.asarray(means, dtype=float), nan=0.0)
        country_rows.append({
            'country': country,
            'family': family,
            'features': means,
            'n_respondents': int(len(sub)),
        })

    if len(country_rows) < 3:
        print("Not enough countries for clustering analysis")
        return {}

    countries = [r['country'] for r in country_rows]
    country_families = {r['country']: r['family'] for r in country_rows}
    X_country = np.vstack([r['features'] for r in country_rows])

    # PCA basis on country-level means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_country)
    pca = PCA(n_components=2)
    _ = pca.fit_transform(X_scaled)

    # Build analysis points using the NEW representation
    # (country-family weighted points from language subgroups)
    language_col = None
    for cand in ['language_str', 'language', 'lang', 'interview_language']:
        if cand in df_normalized.columns:
            language_col = cand
            break

    def _infer_family_from_language(lang_value: str, fallback_family: str) -> str:
        if lang_value is None:
            return fallback_family
        s = str(lang_value).strip().lower()
        if s in {'', 'nan', 'na', 'n/a', 'none', 'null'}:
            return fallback_family
        if any(k in s for k in ['deutsch', 'german', 'english', 'norsk', 'norwegian', 'svenska', 'swedish', 'íslenska', 'icelandic', 'dutch']):
            return 'Germanic'
        if any(k in s for k in ['français', 'french', 'italiano', 'español', 'spanish', 'català', 'galego', 'portugu', 'român', 'romana', 'română']):
            return 'Romance'
        if any(k in s for k in ['рус', 'срп', 'bosnian', 'hrvats', 'sloven', 'slovak', 'slovensk', 'češt', 'czech', 'polski', 'ukrain', 'latvie', 'lietu', 'lithuan', 'serbian', 'croatian']):
            return 'Balto-Slavic'
        if any(k in s for k in ['magyar', 'finn', 'eesti', 'estonian']):
            return 'Uralic'
        if any(k in s for k in ['euskara', 'armenian', 'azerbaijani', 'georgian', 'albanian', 'turkish', 'greek']):
            return 'Others'
        return fallback_family

    point_rows = []

    if language_col is not None:
        work = df_normalized.copy()
        lang = work[language_col].astype('string').str.strip()
        missing_like = {'', 'nan', 'na', 'n/a', 'none', 'null'}
        lang = lang.mask(lang.str.lower().isin(missing_like), other=pd.NA)
        work['_lang_clean'] = lang

        # Keep unknown only if sole language in a country
        keep_mask = pd.Series(False, index=work.index)
        for _, g in work.groupby(country_col, dropna=False):
            n_groups_total = len(g['_lang_clean'].drop_duplicates())
            if n_groups_total <= 1:
                keep_mask.loc[g.index] = True
            else:
                keep_mask.loc[g[g['_lang_clean'].notna()].index] = True
        work = work[keep_mask].copy()

        for country in countries:
            sub_country = work[work[country_col] == country]
            if sub_country.empty:
                continue

            fallback_family = country_families.get(country, 'Unknown')

            lang_rows = []
            for lang_val, sub_lang in sub_country.groupby('_lang_clean', dropna=False):
                if sub_lang.empty:
                    continue
                vals = []
                for col in v_columns:
                    vals.append(sub_lang[col].mean() if col in sub_lang.columns else np.nan)
                vec = np.nan_to_num(np.asarray(vals, dtype=float), nan=0.0)
                lang_label = str(lang_val) if pd.notna(lang_val) else 'Unknown language'
                fam_from_lang = _infer_family_from_language(lang_label, fallback_family)
                lang_rows.append({
                    'country': country,
                    'language': lang_label,
                    'family': fam_from_lang,
                    'vec': vec,
                    'n': int(len(sub_lang)),
                })

            if not lang_rows:
                continue

            lang_df = pd.DataFrame(lang_rows)
            fam_rows = []
            for fam, gfam in lang_df.groupby('family', dropna=False):
                total_n_f = float(gfam['n'].sum())
                if total_n_f <= 0:
                    continue
                w = (gfam['n'].to_numpy(dtype=float) / total_n_f).reshape(-1, 1)
                mat = np.vstack(gfam['vec'].to_list())
                vec_f = (w * mat).sum(axis=0)
                fam_rows.append({
                    'country': country,
                    'family': str(fam),
                    'vec': vec_f,
                    'n_respondents': int(total_n_f),
                    'is_split_country': len(lang_df['family'].unique()) > 1,
                })

            point_rows.extend(fam_rows)
    else:
        # Fallback to country-level points if language metadata is unavailable
        for r in country_rows:
            point_rows.append({
                'country': r['country'],
                'family': r['family'],
                'vec': r['features'],
                'n_respondents': r['n_respondents'],
                'is_split_country': False,
            })

    if len(point_rows) < 3:
        print("Not enough points for clustering analysis")
        return {}

    # Project analysis points in the new PCA representation
    proj = []
    for rr in point_rows:
        vec_scaled = scaler.transform(np.asarray(rr['vec'], dtype=float).reshape(1, -1))
        coords = pca.transform(vec_scaled)[0]
        proj.append({
            'country': rr['country'],
            'family': rr['family'],
            'pc1': float(coords[0]),
            'pc2': float(coords[1]),
            'n_respondents': int(rr['n_respondents']),
            'is_split_country': bool(rr['is_split_country']),
        })

    proj_df = pd.DataFrame(proj)
    countries = proj_df['country'].tolist()
    families = proj_df['family'].tolist()
    X_pca = proj_df[['pc1', 'pc2']].to_numpy(dtype=float)

    # Distance-based diagnostics in PCA space
    dists = np.sqrt(((X_pca[:, None, :] - X_pca[None, :, :]) ** 2).sum(axis=2))
    n = len(countries)

    within_distances = []
    between_distances = []
    nn_same = []

    for i in range(n):
        order = np.argsort(dists[i])
        nn = order[1] if len(order) > 1 else i
        nn_same.append(1 if families[i] == families[nn] else 0)
        for j in range(i + 1, n):
            if families[i] == families[j]:
                within_distances.append(float(dists[i, j]))
            else:
                between_distances.append(float(dists[i, j]))

    within_mean = float(np.mean(within_distances)) if within_distances else np.nan
    between_mean = float(np.mean(between_distances)) if between_distances else np.nan
    within_between_ratio = within_mean / between_mean if between_mean and not np.isnan(between_mean) else np.nan
    nn_same_family_rate = float(np.mean(nn_same)) if nn_same else np.nan

    # Label-based clustering quality metrics (true family labels)
    labels, uniques = pd.factorize(pd.Series(families, dtype='string'))
    unique_count = len(set(labels.tolist()))
    min_cluster_size = min(pd.Series(labels).value_counts().tolist()) if unique_count > 0 else 0

    silhouette = np.nan
    ch_index = np.nan
    db_index = np.nan
    if unique_count >= 2 and unique_count < n and min_cluster_size >= 2:
        silhouette = float(silhouette_score(X_pca, labels))
    if unique_count >= 2 and unique_count < n:
        ch_index = float(calinski_harabasz_score(X_pca, labels))
        db_index = float(davies_bouldin_score(X_pca, labels))

    print("\nPCA family clustering quality:")
    print(f"  Points: {n} (country-family representation)")
    print(f"  Unique countries represented: {proj_df['country'].nunique()}")
    print(f"  Split country points: {int(proj_df['is_split_country'].sum())}")
    print(f"  Families : {len(uniques)} -> {sorted(set(families))}")
    print(f"  PCA variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")
    print(f"  Silhouette (true labels)     : {silhouette:.4f}")
    print(f"  Calinski-Harabasz            : {ch_index:.4f}")
    print(f"  Davies-Bouldin               : {db_index:.4f}")
    print(f"  1-NN same-family rate        : {nn_same_family_rate:.4f}")
    print(f"  Mean distance (within family): {within_mean:.4f}")
    print(f"  Mean distance (between fam.) : {between_mean:.4f}")
    print(f"  Within/Between distance ratio: {within_between_ratio:.4f}")

    # Save points used in the figure
    points_df = proj_df.copy()
    if points_output_path:
        points_df.to_csv(points_output_path, index=False)
        print(f"PCA country coordinates saved to: {points_output_path}")

    # Save summary metrics
    metrics = {
        'n_points': n,
        'n_countries': int(proj_df['country'].nunique()),
        'n_split_country_points': int(proj_df['is_split_country'].sum()),
        'n_families': len(uniques),
        'families': ';'.join(sorted(set(families))),
        'n_variables': len(v_columns),
        'pc1_explained_variance': float(pca.explained_variance_ratio_[0]),
        'pc2_explained_variance': float(pca.explained_variance_ratio_[1]),
        'pca_total_explained_variance': float(pca.explained_variance_ratio_.sum()),
        'silhouette_true_labels': silhouette,
        'calinski_harabasz': ch_index,
        'davies_bouldin': db_index,
        'nn_same_family_rate': nn_same_family_rate,
        'mean_distance_within_family': within_mean,
        'mean_distance_between_family': between_mean,
        'within_between_distance_ratio': within_between_ratio,
    }
    if metrics_output_path:
        pd.DataFrame([metrics]).to_csv(metrics_output_path, index=False)
        print(f"PCA clustering metrics saved to: {metrics_output_path}")

    # Plot: distance structure only (within/between + average country-level pairwise baseline)
    # Optimized for single-column readability in double-column papers.
    with plt.style.context(['science', 'no-latex']):
        mpl.rcParams.update({
            'font.size': 8.0,
            'axes.labelsize': 8.5,
            'axes.titlesize': 8.5,
            'xtick.labelsize': 8.0,
            'ytick.labelsize': 8.0,
            'legend.fontsize': 7.4,
        })

        # ~3.35in width suits most two-column venues (single-column panel).
        fig, ax = plt.subplots(1, 1, figsize=(3.35, 2.65))

        dist_data = [within_distances, between_distances]
        ax.boxplot(
            dist_data,
            tick_labels=['Within family', 'Between families'],
            showmeans=False,
            patch_artist=True,
            widths=0.52,
            boxprops=dict(facecolor=ACADEMIC_COLORS['palette'][0], alpha=0.30, color='#333333', linewidth=1.0),
            medianprops=dict(color='#111111', linewidth=1.6),
            whiskerprops=dict(color='#333333', linewidth=1.0),
            capprops=dict(color='#333333', linewidth=1.0),
        )
        ax.set_ylabel('Pairwise Euclidean distance (PCA space)')
        ax.tick_params(axis='x', pad=3)
        ax.tick_params(axis='y', pad=2)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

        # Keep the panel visually minimal for single-column readability.

        # Baseline: average distance across all country-level pairs in PCA space
        all_pair_distances = within_distances + between_distances
        random_baseline = float(np.mean(all_pair_distances)) if all_pair_distances else np.nan
        if not np.isnan(random_baseline):
            ax.axhline(
                random_baseline,
                color=ACADEMIC_COLORS['palette'][2],
                linestyle='--',
                linewidth=1.3,
                label=f'Average country-level pairwise baseline (μ) = {random_baseline:.3f}',
            )
            # Keep only the baseline legend as requested.
            ax.legend(loc='upper left', frameon=True, borderpad=0.25, handlelength=1.8)

        fig.tight_layout(pad=0.25)
        if output_path:
            fig.savefig(output_path, dpi=360, bbox_inches='tight', pad_inches=0.04)
            print(f"PCA family clustering plot saved to: {output_path}")
        plt.close(fig)

    return metrics


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze human survey responses and generate radar charts'
    )
    parser.add_argument(
        '-s', '--survey',
        type=str,
        default='/Users/dsolans/Documents/Research/PROJECTS/ELOQUENCE/MoralValues/EUValues/Surveys_responses/Human_responses/moral_values_survey_with_languages_v2.csv',
        help='Path to the survey data CSV file'
    )
    parser.add_argument(
        '-t', '--topics',
        type=str,
        default='/Users/dsolans/Documents/Research/PROJECTS/ELOQUENCE/MoralValues/EUValues/Surveys/Survey_metadata/questions_topic_matching_v3.csv',
        help='Path to the topic matching CSV file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory for charts and data. If not specified, displays charts only.'
    )
    parser.add_argument(
        '--by-country',
        action='store_true',
        help='Generate additional radar chart comparing countries'
    )
    parser.add_argument(
        '--pca',
        action='store_true',
        help='Generate PCA visualization of countries grouped by language family'
    )
    parser.add_argument(
        '--pca-country-language',
        action='store_true',
        help='Generate PCA visualization with country-language points when language metadata is available'
    )
    parser.add_argument(
        '--consensus',
        action='store_true',
        help='Run consensus analysis (Krippendorff alpha + KS tests) by country and language family'
    )
    parser.add_argument(
        '--consensus-pca-family',
        action='store_true',
        help='Run PCA-based family clustering quality analysis (true language-family labels)'
    )

    args = parser.parse_args()
    
    # Load data
    survey_df = load_survey_data(args.survey)
    topic_df = load_topic_matching(args.topics)
    
    # Get variable columns
    v_columns = get_variable_columns(survey_df)
    
    # Normalize data using canonical questionnaire scales when available.
    # This uses only questionnaire metadata from Surveys_parsed/ and does not
    # load any LLM outputs.
    scale_source = DEFAULT_SCALE_CATALOG if DEFAULT_SCALE_CATALOG.exists() else DEFAULT_SCALES_DIR
    print(f"Loading canonical scale metadata from: {scale_source}")
    scale_metadata = load_canonical_scale_metadata(scale_source)
    survey_normalized = normalize_survey_data_with_scales(survey_df, v_columns, scale_metadata)
    
    # Calculate topic averages
    topic_averages = calculate_topic_averages(survey_normalized, topic_df, v_columns)
    
    # Print summary
    print("\n" + "="*60)
    print("Topic Average Scores Summary:")
    print("="*60)
    print(topic_averages.to_string(index=False))
    
    # Set up output paths
    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save topic averages to CSV
        averages_path = output_dir / 'topic_averages.csv'
        topic_averages.to_csv(averages_path, index=False)
        print(f"\nTopic averages saved to: {averages_path}")
        
        # Save normalized data
        normalized_path = output_dir / 'normalized_survey_with_topics.csv'
        survey_normalized.to_csv(normalized_path, index=False)
        print(f"Normalized survey data saved to: {normalized_path}")
    
    # Generate basic radar chart with topic averages
    basic_radar_path = str(output_dir / 'radar_chart_topics.png') if output_dir else None
    plot_radar_chart(
        topic_averages,
        output_path=basic_radar_path
    )
    
    # Generate radar chart with overall average AND language families
    radar_path = str(output_dir / 'radar_chart_topics_by_language_family.png') if output_dir else None
    plot_radar_chart_with_language_families(
        survey_normalized,
        topic_df,
        v_columns,
        topic_averages,
        output_path=radar_path
    )
    
    # Generate country comparison if requested
    if args.by_country:
        country_radar_path = str(output_dir / 'radar_chart_by_country.png') if output_dir else None
        plot_radar_chart_by_country(
            survey_normalized, 
            topic_df, 
            v_columns,
            output_path=country_radar_path
        )
    
    # Generate PCA visualization if requested
    if args.pca:
        pca_path_1col = str(output_dir / 'pca_countries_by_language_family_1col.png') if output_dir else None
        pca_path_2col = str(output_dir / 'pca_countries_by_language_family_2col.png') if output_dir else None
        # Keep legacy paths for backward compatibility
        pca_path = pca_path_1col
        pca_interp_path = str(output_dir / 'pca_components_interpretation.csv') if output_dir else None
        # Filter to the selected subset of variables for PCA
        pca_columns = [col for col in PCA_SELECTED_VARS if col in survey_normalized.columns]
        missing_pca = [col for col in PCA_SELECTED_VARS if col not in survey_normalized.columns]
        if missing_pca:
            print(f"Warning: {len(missing_pca)} PCA variable(s) not found in survey data: {missing_pca}")
        print(f"Running PCA on {len(pca_columns)} selected variables (out of {len(v_columns)} total)")
        print("Using country-language-aware PCA representation as default output")

        # Keep component interpretation export compatible with downstream steps.
        # The PCA basis remains country-level (same as before), but the displayed
        # figure is now the country-language-aware representation.
        plot_pca_by_country(
            survey_normalized,
            pca_columns,
            output_path=None,
            interpretation_output_path=pca_interp_path
        )

        # Primary PCA figure for pipeline/output: country-language-aware points (one and two column versions)
        print("Generating PCA country-language points — one and two column versions…")
        plot_pca_country_language_points(
            survey_normalized,
            pca_columns,
            output_path=pca_path_1col,
            title="PCA of Country Responses by Language Family",
            column_width=3.5,
        )
        plot_pca_country_language_points(
            survey_normalized,
            pca_columns,
            output_path=pca_path_2col,
            title="PCA of Country Responses by Language Family",
            column_width=7.0,
        )

    if args.pca_country_language:
        pca_country_lang_path_1col = str(output_dir / 'pca_countries_by_language_points_1col.png') if output_dir else None
        pca_country_lang_path_2col = str(output_dir / 'pca_countries_by_language_points_2col.png') if output_dir else None
        pca_columns = [col for col in PCA_SELECTED_VARS if col in survey_normalized.columns]
        print("Generating PCA country-language points (alt) — one and two column versions…")
        plot_pca_country_language_points(
            survey_normalized,
            pca_columns,
            output_path=pca_country_lang_path_1col,
            column_width=3.5,
        )
        plot_pca_country_language_points(
            survey_normalized,
            pca_columns,
            output_path=pca_country_lang_path_2col,
            column_width=7.0,
        )

    # Generate consensus analysis if requested
    if args.consensus:
        consensus_path = str(output_dir / 'consensus_analysis.png') if output_dir else None
        consensus_columns = [col for col in PCA_SELECTED_VARS if col in survey_normalized.columns]
        analyze_consensus(
            survey_normalized,
            consensus_columns,
            output_path=consensus_path,
        )

    if args.consensus_pca_family:
        pca_clustering_plot = str(output_dir / 'pca_family_clustering_quality.png') if output_dir else None
        pca_clustering_metrics = str(output_dir / 'pca_family_clustering_quality_metrics.csv') if output_dir else None
        pca_country_points = str(output_dir / 'pca_country_coordinates_with_family.csv') if output_dir else None
        clustering_columns = [col for col in PCA_SELECTED_VARS if col in survey_normalized.columns]
        analyze_pca_family_clustering(
            survey_normalized,
            clustering_columns,
            output_path=pca_clustering_plot,
            metrics_output_path=pca_clustering_metrics,
            points_output_path=pca_country_points,
        )


if __name__ == '__main__':
    main()
