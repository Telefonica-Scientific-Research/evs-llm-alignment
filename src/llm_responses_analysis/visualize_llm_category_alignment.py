#!/usr/bin/env python3
"""
Visualize LLM alignment with semantic categories based on questions_topic_matching_v3.csv.

Reads responses from multiple LLM models, maps them to semantic categories 
using questions_topic_matching_v3.csv, computes average scores per category per model,
and produces visualizations (radar chart and bar charts).

Saves output to the same folder as responses.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

BASE_DIR = Path('/Users/dsolans/Documents/Research/PROJECTS/ELOQUENCE/MoralValues')
RESPONSES_DIR = BASE_DIR / 'Responses' / 'Parsed'
QUESTIONS_FILE = BASE_DIR / 'questions_topic_matching_v3.csv'
OUT_DIR = RESPONSES_DIR.parent / 'Category_Analysis'

# Create output directory if it doesn't exist
OUT_DIR.mkdir(exist_ok=True)

def load_questions_mapping():
    """
    Load the questions_topic_matching_v3.csv file and create a mapping from
    variable ID (e.g., 'v1') to semantic category.
    
    File format: semicolon-delimited with columns:
    - EVS 2017 Variable Name (e.g., 'v1', 'v2', ...)
    - EVS 2017 Variable Label (description)
    - Category (semantic category, excludes 'Unknown')
    
    Returns:
        dict: mapping from variable_id to category
    """
    df = pd.read_csv(QUESTIONS_FILE, sep=';')
    
    print(f"Loaded {len(df)} rows from {QUESTIONS_FILE}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create mapping from variable ID to category
    # Filter out rows with 'Unknown' category
    var_mapping = {}
    
    for idx, row in df.iterrows():
        var_id = str(row['EVS 2017 Variable Name']).strip()
        category = str(row['Category']).strip()
        
        # Skip Unknown categories
        if category.lower() != 'unknown':
            var_mapping[var_id] = category
    
    print(f"Created mapping for {len(var_mapping)} variables")
    
    # Show category distribution
    categories = list(var_mapping.values())
    unique_cats = set(categories)
    print(f"Unique categories: {sorted(unique_cats)}")
    for cat in sorted(unique_cats):
        count = categories.count(cat)
        print(f"  {cat}: {count} variables")
    
    return var_mapping

def read_model_responses(csv_path):
    """
    Read a single model's response CSV file.
    
    Returns:
        DataFrame: with respondent_id as index and variables as columns
    """
    df = pd.read_csv(csv_path, index_col=0)
    return df


def compute_category_averages(response_df, var_mapping, detailed=False):
    """
    Compute average response score for each category.
    
    Args:
        response_df: DataFrame with variables as columns
        var_mapping: dict mapping variable_id to category
        detailed: if True, return detailed breakdown per category
        
    Returns:
        dict or tuple: mapping from Category to average score
                      (and detailed breakdown if detailed=True)
    """
    category_scores = {}
    category_details = {}
    
    # Get all numeric columns (variables starting with 'v')
    numeric_cols = [col for col in response_df.columns if col.startswith('v')]
    
    # For each variable, calculate average score
    for col in numeric_cols:
        # Look up the category for this variable
        if col not in var_mapping:
            # Variable not in mapping (probably metadata or unknown)
            continue
        
        category = var_mapping[col]
        
        # Initialize category if not present
        if category not in category_scores:
            category_scores[category] = []
            category_details[category] = []
        
        # Get average score for this variable (ignore NaN and special values)
        values = pd.to_numeric(response_df[col], errors='coerce')
        # Filter out special codes (88, 89, 99) and NaN, keep only valid responses (usually 1-10)
        clean_values = values[(values >= 0) & (values < 88) & (~values.isna())]
        
        if len(clean_values) > 0:
            var_avg = clean_values.mean()
            category_scores[category].append(var_avg)
            category_details[category].append({
                'variable': col,
                'score': var_avg,
                'count': len(clean_values)
            })
    
    # Average scores within each category
    category_averages = {}
    for cat, scores in category_scores.items():
        if scores:
            category_averages[cat] = np.mean(scores)
    
    if detailed:
        return category_averages, category_details
    return category_averages

def build_summary(responses_dir, var_mapping):
    """
    Build a summary of average scores per category for all models.
    
    Args:
        responses_dir: directory containing parsed response CSVs
        var_mapping: dict from variable_id to category
    
    Returns:
        tuple: (summary_df, detailed_dict)
    """
    summary = {}
    detailed = {}
    
    for csv_file in sorted(responses_dir.glob('*_responses_all_languages.csv')):
        model_name = csv_file.stem.replace('_responses_all_languages', '').strip()
        
        print(f"\nProcessing {model_name}...")
        
        try:
            response_df = read_model_responses(csv_file)
            print(f"  Loaded {len(response_df)} respondents with {len(response_df.columns)} columns")
            
            category_avgs, category_detail = compute_category_averages(
                response_df, var_mapping, detailed=True
            )
            
            if category_avgs:
                summary[model_name] = category_avgs
                detailed[model_name] = category_detail
                print(f"  Computed averages for {len(category_avgs)} categories")
                for cat, avg in sorted(category_avgs.items()):
                    print(f"    {cat}: {avg:.2f}")
            else:
                print(f"  Warning: No category averages computed for {model_name}")
                
        except Exception as e:
            print(f"  Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    if not summary:
        raise SystemExit("No models processed successfully.")
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(summary).T
    print(f"\nFinal summary shape: {summary_df.shape}")
    print(f"Categories: {summary_df.columns.tolist()}")
    
    return summary_df, detailed

def plot_radar(summary_df, out_file):
    """
    Create a publication-ready radar chart showing model performance across categories.
    """
    import matplotlib as mpl
    
    out_path = Path(out_file)
    
    # Paper-style rcParams
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Georgia', 'serif'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.linewidth': 0.8
    })
    
    labels = list(summary_df.columns)
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    
    # Handle inf/nan values
    arr = summary_df.replace([np.inf, -np.inf], np.nan).values
    finite_mask = ~np.isnan(arr)
    
    if finite_mask.any():
        finite_max = float(np.nanmax(arr[finite_mask]))
        finite_min = float(np.nanmin(arr[finite_mask]))
    else:
        finite_max = 10.0
        finite_min = 0.0
    
    fallback = finite_max * 1.1
    plot_df = summary_df.fillna(fallback)
    
    # Create polar plot
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    cmap = plt.get_cmap('tab10')
    
    # Rotate so first axis is at top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Plot each model
    for idx, (model_name, row) in enumerate(plot_df.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=1.5, label=model_name, 
                color=cmap(idx), markersize=4)
        ax.fill(angles, values, alpha=0.15, color=cmap(idx))
    
    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0, finite_max * 1.15)
    ax.set_yticks(np.linspace(0, finite_max, 5))
    ax.grid(True, linewidth=0.5, alpha=0.6)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9, framealpha=0.95)
    ax.set_title('LLM Alignment with Semantic Categories\n(Average Response Scores)', 
                fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✓ Radar chart saved to {out_path}")
    plt.close()

def plot_grouped_bars(summary_df, out_file):
    """
    Create grouped bar chart showing category scores for each model.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(summary_df.columns))
    width = 0.15
    colors = plt.cm.tab10(np.linspace(0, 1, len(summary_df)))
    
    for idx, (model_name, row) in enumerate(summary_df.iterrows()):
        offset = width * (idx - len(summary_df) / 2 + 0.5)
        ax.bar(x + offset, row.values, width, label=model_name, color=colors[idx], 
               alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Semantic Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Alignment Across Semantic Categories', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df.columns, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"✓ Grouped bar chart saved to {out_file}")
    plt.close()

def plot_heatmap(summary_df, out_file):
    """
    Create a heatmap showing category scores for all models.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(summary_df.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=10)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(summary_df.columns)))
    ax.set_yticks(np.arange(len(summary_df)))
    ax.set_xticklabels(summary_df.columns, fontsize=10)
    ax.set_yticklabels(summary_df.index, fontsize=10)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(summary_df)):
        for j in range(len(summary_df.columns)):
            text = ax.text(j, i, f'{summary_df.values[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Model Alignment Heatmap\n(Average Scores by Category)', 
                fontweight='bold', fontsize=12, pad=15)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Score', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved to {out_file}")
    plt.close()

def save_summary_csv(summary_df, out_file):
    """
    Save summary DataFrame to CSV.
    """
    summary_df.to_csv(out_file)
    print(f"✓ Summary CSV saved to {out_file}")

def save_detailed_report(detailed_dict, out_file):
    """
    Save detailed category analysis to JSON.
    """
    # Convert numpy types to native Python types for JSON serialization
    json_safe = {}
    for model, categories in detailed_dict.items():
        json_safe[model] = {}
        for cat, vars_list in categories.items():
            json_safe[model][cat] = []
            for var_info in vars_list:
                json_safe[model][cat].append({
                    'variable': str(var_info['variable']),
                    'score': float(var_info['score']),
                    'count': int(var_info['count'])
                })
    
    with open(out_file, 'w') as f:
        json.dump(json_safe, f, indent=2)
    print(f"✓ Detailed report saved to {out_file}")

def print_summary_statistics(summary_df):
    """
    Print summary statistics for each category across models.
    """
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS BY CATEGORY")
    print("=" * 80)
    
    for col in summary_df.columns:
        values = summary_df[col]
        print(f"\n{col}:")
        print(f"  Mean:   {values.mean():.3f}")
        print(f"  Median: {values.median():.3f}")
        print(f"  Std:    {values.std():.3f}")
        print(f"  Min:    {values.min():.3f} ({values.idxmin()})")
        print(f"  Max:    {values.max():.3f} ({values.idxmax()})")


def main():
    print("=" * 80)
    print("LLM Semantic Category Alignment Analysis")
    print("=" * 80)
    
    # Load questions mapping
    print("\n1. Loading questions mapping...")
    var_mapping = load_questions_mapping()
    
    # Build summary of category averages
    print("\n2. Building category averages across models...")
    summary_df, detailed_dict = build_summary(RESPONSES_DIR, var_mapping)
    
    # Print summary statistics
    print_summary_statistics(summary_df)
    
    # Save summary
    print("\n3. Saving results...")
    save_summary_csv(summary_df, OUT_DIR / 'category_alignment_summary.csv')
    save_detailed_report(detailed_dict, OUT_DIR / 'category_alignment_detailed.json')
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    plot_radar(summary_df, OUT_DIR / 'category_alignment_radar.png')
    plot_grouped_bars(summary_df, OUT_DIR / 'category_alignment_bars.png')
    plot_heatmap(summary_df, OUT_DIR / 'category_alignment_heatmap.png')
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {OUT_DIR}")
    print("=" * 80)
    
    return summary_df

if __name__ == '__main__':
    summary = main()
