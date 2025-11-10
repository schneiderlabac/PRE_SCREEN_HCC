import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
from matplotlib.cm import viridis
import seaborn as sns
import os
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import scipy
import sklearn
from itertools import combinations
from scipy.stats import sem, t
from scipy import stats
from itertools import combinations
import os
import yaml
import warnings
from typing import Dict, List, Optional, Tuple
import matplotlib.figure
import matplotlib.axes
warnings.filterwarnings("ignore", message="indexing past lexsort depth may impact performance")





# Scraping default colors from the default_colors.yaml. In the notebooks, these can be overwritten by userspecific color schemes
# try:
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     default_config_path = os.path.join(script_dir, 'default_colors.yaml')

#     with open(default_config_path, 'r') as file:
#         _config = yaml.safe_load(file)

#     all_scenarios = _config.get("scenarios_colors", {})
#     scenario_lists = _config.get("scenario_lists", {})
# except (FileNotFoundError, yaml.YAMLError):
#     print(f"Error: Could not load default colors from YAML.")

# Parameters for the plotting:
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["figure.titlesize"] = 14

#################################################################################################################################
########################################    Helpers    ###############################################################################
###################################################################################################################################

def plot_colorbar(scenario_colors, alpha=1.0):
    """
    Plots a colorbar based on the given dictionary of scenario colors.

    Parameters:
    - scenario_colors (dict): A dictionary where keys are scenario labels and values are hex color codes.

    Example:
    plot_colorbar({
        'A': '#36617B',  # Color for scenario A
        'B': '#4993AA',  # Color for scenario B
        'C': '#e3c983',  # Color for scenario C
        'D': '#ECB354',  # Color for scenario D
        'E': '#B95224',  # Color for scenario E
    })
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(5.5, 1.1))  # Size corresponding to 275x55 pixels
    # Iterate over the items in the dictionary to create colored rectangles and labels
    for i, (label, color) in enumerate(scenario_colors.items()):
        # Draw the colored rectangles
        rect = patches.Rectangle((i * 55, 0), 55, 55, linewidth=2, edgecolor='white', facecolor=color)
        ax.add_patch(rect)
        # Add labels below each rectangle
        ax.text(i * 55 + 27.5, -10, label, ha='center', va='top', fontsize=10, color='black')

    # Set the limits and turn off the axes
    ax.set_xlim(0, 275)
    ax.set_ylim(-20, 55)
    ax.axis('off')
    plt.show()

# set the seaborn parameters
sns.set_theme(context='poster',style='white')
sns.set_palette('bright')



#######################################################################################################################################

def adjust_alpha(hex_color, alpha=1.0):
    """ Adjust the alpha value of a hex color. """
    import matplotlib.colors as mcolors
    color_rgba = mcolors.to_rgba(hex_color, alpha)
    return color_rgba

#######################################################################################################################################
def adjust_background():
    paths = plt.gca().collections
    for path in paths:
        verts = path.get_offsets()
        x = [vert[0] for vert in verts]
        y = [vert[1] for vert in verts]

        # Create a patch filled with a pattern
        poly = patches.Polygon(np.column_stack((x, y)), closed=True, edgecolor='none', facecolor='lightgray', hatch='\\')

        # Add the patch to the plot
        plt.gca().add_patch(poly)

#######################################################################################################################################
def summarize_dataframes(dataframes, report_columns=True):
    """
    Summarize attributes such as number of cases/controls, prediction values
    for the different dataframes in the dictionary

    Parameters:
    - dataframes (dict): Dictionary of dataframes to summarize
    - report_columns (bool): Whether to print a report about column availability

    Returns:
    - pd.DataFrame: Summary statistics for all dataframes
    """
    summary = []

    # Track column availability
    has_status_cancerreg = []
    missing_status_cancerreg = []

    for key, df in dataframes.items():
        num_rows, num_cols = df.shape
        num_status_1 = df['status'].sum()
        num_status_0 = num_rows - num_status_1
        avg_prediction_all = df['y_pred'].mean()
        avg_prediction_status_1 = df[df['status'] == 1]['y_pred'].mean()
        avg_prediction_status_0 = df[df['status'] == 0]['y_pred'].mean()

        # Check for status_cancerreg column
        has_cancerreg = 'status_cancerreg' in df.columns

        if has_cancerreg:
            has_status_cancerreg.append(key)
            num_status_1_cancerreg = df['status_cancerreg'].sum()
            avg_prediction_status_1_cancerreg = df[df['status_cancerreg'] == 1]['y_pred'].mean()
        else:
            missing_status_cancerreg.append(key)
            # Use status values as fallback
            num_status_1_cancerreg = num_status_1
            avg_prediction_status_1_cancerreg = avg_prediction_status_1

        summary.append({
            'Dataframe': key,
            'Rows': num_rows,
            'Columns': num_cols,
            'Status 1': num_status_1,
            'Status 1 CancerReg': num_status_1_cancerreg,
            'Status 0': num_status_0,
            'Has CancerReg Column': has_cancerreg,
            'Avg Prediction (All)': round(avg_prediction_all, 4),
            'Avg Prediction (Status 1)': round(avg_prediction_status_1, 4),
            'Avg Prediction (Status 1 CancerReg)': round(avg_prediction_status_1_cancerreg, 4),
            'Avg Prediction (Status 0)': round(avg_prediction_status_0, 4)
        })

    # Print column availability report
    if report_columns:
        print("=" * 70)
        print("DATAFRAME COLUMN AVAILABILITY REPORT")
        print("=" * 70)

        total_dataframes = len(dataframes)
        has_cancerreg_count = len(has_status_cancerreg)
        missing_cancerreg_count = len(missing_status_cancerreg)

        print(f"Total dataframes analyzed: {total_dataframes}")
        print(f"Dataframes with 'status_cancerreg' column: {has_cancerreg_count}")
        print(f"Dataframes missing 'status_cancerreg' column: {missing_cancerreg_count}")

        if has_status_cancerreg:
            print(f"\n✓ Dataframes WITH 'status_cancerreg':")
            for key in has_status_cancerreg:
                print(f"  - {key}")

        if missing_status_cancerreg:
            print(f"\n⚠ Dataframes MISSING 'status_cancerreg' (using 'status' as fallback):")
            for key in missing_status_cancerreg:
                print(f"  - {key}")

        if has_cancerreg_count == total_dataframes:
            print(f"\n🎉 All dataframes have the 'status_cancerreg' column!")
        elif missing_cancerreg_count == total_dataframes:
            print(f"\n⚠ No dataframes have the 'status_cancerreg' column - using 'status' for all calculations.")
        else:
            print(f"\n⚠ Mixed availability: {has_cancerreg_count}/{total_dataframes} dataframes have 'status_cancerreg'")

        print("=" * 70)
        print()

    return pd.DataFrame(summary)


def check_column_consistency(dataframes, required_columns=None, optional_columns=None):
    """
    Check consistency of columns across all dataframes.

    Parameters:
    - dataframes (dict): Dictionary of dataframes to check
    - required_columns (list): Columns that should be in all dataframes
    - optional_columns (list): Columns to check availability for

    Returns:
    - dict: Summary of column availability across dataframes
    """
    if required_columns is None:
        required_columns = ['status', 'y_pred']

    if optional_columns is None:
        optional_columns = ['status_cancerreg']

    all_columns = required_columns + optional_columns
    column_summary = {}

    for col in all_columns:
        column_summary[col] = {
            'present_in': [],
            'missing_from': [],
            'coverage': 0
        }

    # Check each dataframe
    for key, df in dataframes.items():
        for col in all_columns:
            if col in df.columns:
                column_summary[col]['present_in'].append(key)
            else:
                column_summary[col]['missing_from'].append(key)

    # Calculate coverage
    total_dfs = len(dataframes)
    for col in all_columns:
        present_count = len(column_summary[col]['present_in'])
        column_summary[col]['coverage'] = present_count / total_dfs

    # Print detailed report
    print("=" * 70)
    print("DETAILED COLUMN CONSISTENCY REPORT")
    print("=" * 70)

    for col in all_columns:
        coverage = column_summary[col]['coverage']
        present_count = len(column_summary[col]['present_in'])
        col_type = "REQUIRED" if col in required_columns else "OPTIONAL"

        print(f"\nColumn '{col}' ({col_type}):")
        print(f"  Coverage: {present_count}/{total_dfs} ({coverage:.1%})")

        if coverage == 1.0:
            print(f"  ✓ Present in ALL dataframes")
        elif coverage == 0.0:
            print(f"  ✗ Missing from ALL dataframes")
        else:
            print(f"  ⚠ Partial coverage")
            if column_summary[col]['present_in']:
                print(f"    Present in: {', '.join(column_summary[col]['present_in'])}")
            if column_summary[col]['missing_from']:
                print(f"    Missing from: {', '.join(column_summary[col]['missing_from'])}")

    print("=" * 70)

    return column_summary



#######################################################################################################################################

def assign_colors(keys_ordered, color_dict):
    """
    Assign colors to keys with a hierarchical fallback system.

    For each key in keys_ordered (e.g., 'all_Model_A'), it tries to find a color in this order:
    1. Exact match in color_dict (e.g., 'all_Model_A')
    2. Match for substring after first underscore (e.g., 'Model_A')
    3. Match for last substring after underscore (e.g., 'A')
    4. Default color if no match is found

    Parameters:
    -----------
    keys_ordered : list
        List of keys to assign colors to
    color_dict : dict
        Dictionary mapping keys to color values

    Returns:
    --------
    dict
        Dictionary mapping each key in keys_ordered to a color
    """
    assigned_colors = {}
    default_color = "#808080"  # Default gray color

    for key in keys_ordered:
        # Try exact match
        if key in color_dict:
            assigned_colors[key] = color_dict[key]
            continue

        # Try match for part after first underscore
        if '_' in key:
            partial_key = key.split('_', 1)[1]  # Get everything after first underscore
            if partial_key in color_dict:
                assigned_colors[key] = color_dict[partial_key]
                continue

        # Try match for last part after underscore
        if '_' in key:
            last_part = key.split('_')[-1]  # Get last part
            if last_part in color_dict:
                assigned_colors[key] = color_dict[last_part]
                continue

        # If we're here, no match was found - try matching part of the key
        matched = False
        for color_key in color_dict:
            if color_key in key:
                assigned_colors[key] = color_dict[color_key]
                matched = True
                break

        # If still no match, use default
        if not matched:
            assigned_colors[key] = default_color
            print(f"Warning: No color found for '{key}'. Using default color.")

    return assigned_colors




def clean_keys_for_titles(keys, prefixes_to_remove=None, suffixes_to_remove=None,
                          separator='_', title_case=False, replace_separator_with=' '):
    """
    Clean keys to create readable titles by removing prefixes/suffixes and formatting.

    Parameters:
    - keys (list or str): Single key or list of keys to clean
    - prefixes_to_remove (list): List of prefixes to remove (case-insensitive).
                                Default: ['all', 'par', 'model']
    - suffixes_to_remove (list): List of suffixes to remove (case-insensitive).
                                Default: None
    - separator (str): Character used to separate parts in the key. Default: '_'
    - title_case (bool): Whether to convert to title case. Default: False (preserves original case)
    - replace_separator_with (str): What to replace the separator with. Default: ' '

    Returns:
    - str or dict: If input is a string, returns cleaned string.
                   If input is a list, returns dict mapping original keys to cleaned titles.

    Examples:
    - clean_keys_for_titles("all_Model_Demographics") → "Model Demographics"
    - clean_keys_for_titles("all_aMAP") → "aMAP"
    - clean_keys_for_titles("all_FIB4") → "FIB4"
    """

    # Set default prefixes if none provided
    if prefixes_to_remove is None:
        prefixes_to_remove = ['all', 'par', 'model']

    # Convert to lowercase for comparison
    prefixes_lower = [p.lower() for p in prefixes_to_remove] if prefixes_to_remove else []
    suffixes_lower = [s.lower() for s in suffixes_to_remove] if suffixes_to_remove else []

    def clean_single_key(key):
        """Clean a single key."""
        if not isinstance(key, str):
            return str(key)

        # Split by separator to handle each part
        parts = key.split(separator)
        cleaned_parts = []

        # Process each part
        for i, part in enumerate(parts):
            part_lower = part.lower()

            # Skip if it's a prefix we want to remove (only check first few parts)
            if i < 3 and part_lower in prefixes_lower:  # Only check first 3 parts for prefixes
                continue

            # Skip if it's a suffix we want to remove (only check last few parts)
            if i >= len(parts) - 3 and part_lower in suffixes_lower:  # Only check last 3 parts for suffixes
                continue

            # Keep this part (preserve original capitalization)
            cleaned_parts.append(part)

        # Join remaining parts
        if cleaned_parts:
            cleaned_key = separator.join(cleaned_parts)
        else:
            cleaned_key = key  # Fallback if everything was removed

        # Replace separator with specified replacement
        if replace_separator_with is not None:
            cleaned_key = cleaned_key.replace(separator, replace_separator_with)

        # Apply title case if requested (now defaults to False)
        if title_case:
            cleaned_key = cleaned_key.title()

        return cleaned_key

    # Handle single key vs list of keys
    if isinstance(keys, str):
        return clean_single_key(keys)
    else:
        # Return dictionary mapping original keys to cleaned titles
        return {key: clean_single_key(key) for key in keys}


def get_title_with_fallback(key, title_dict, report=False, **kwargs):
    """
    Get title from title_dict or generate one from the key, with optional reporting.

    Parameters:
    - key (str): The key to look up or clean
    - title_dict (dict): Dictionary of custom titles
    - report (bool): Whether to print where the title came from
    - **kwargs: Additional arguments passed to clean_keys_for_titles()

    Returns:
    - str: Title from dict or cleaned key
    """
    if key in title_dict:
        title = title_dict[key]
        if report:
            print(f"✓ '{key}' → '{title}' (from title_dict)")
        return title
    else:
        title = clean_keys_for_titles(key, **kwargs)
        if report:
            print(f"⚡ '{key}' → '{title}' (auto-generated)")
        return title


def get_titles_with_report(keys_ordered, title_dict, **kwargs):
    """
    Get titles for all keys and print a summary report.

    Parameters:
    - keys_ordered (list): List of keys to process
    - title_dict (dict): Dictionary of custom titles
    - **kwargs: Additional arguments passed to clean_keys_for_titles()

    Returns:
    - dict: Dictionary mapping keys to their final titles
    """
    print("=" * 60)
    print("TITLE GENERATION REPORT")
    print("=" * 60)

    titles = {}
    custom_count = 0
    auto_count = 0

    for key in keys_ordered:
        if key in title_dict:
            title = title_dict[key]
            titles[key] = title
            print(f"✓ '{key}' → '{title}' (from title_dict)")
            custom_count += 1
        else:
            title = clean_keys_for_titles(key, **kwargs)
            titles[key] = title
            print(f"⚡ '{key}' → '{title}' (auto-generated)")
            auto_count += 1

    print("-" * 60)
    print(f"Summary: {custom_count} custom titles, {auto_count} auto-generated titles")
    print("=" * 60)

    return titles


# Example usage functions for your specific cases:

def get_violin_title(key, title_dict):
    """Get title for violin plots - removes common model prefixes."""
    return get_title_with_fallback(
        key, title_dict,
        prefixes_to_remove=['all', 'par', 'model']
    )

def get_pr_curve_label(key, title_dict=None):
    """
    Get label for precision-recall curves - extracts last two parts after underscore.

    Examples:
    - "RandomForest_train_fold1" → "Train Fold1"
    - "SVM_validation_final" → "Validation Final"
    """
    if title_dict and key in title_dict:
        return title_dict[key]

    # Split by underscore and take last two parts
    parts = key.split('_')
    if len(parts) >= 2:
        label = f"{parts[-2]}_{parts[-1]}"
        return clean_keys_for_titles(label)
    else:
        return clean_keys_for_titles(key)







#################################################################################################################################
########################################    Violin Plots    ###############################################################################
###################################################################################################################################
def create_violin_plots(dataframes, keys_ordered, color_dict, title_dict,
                        display, title_display, gap, inner_detail="quart",
                        highlight_column=None,change_display=None, highlight_color="grey", highlight_marker="o",
                        highlight_size=0.5, alpha=0.7,
                        n_rows=1, n_cols=5, split_by_sex=False, truth="status_cancerreg",
                        font_size=24, fig_path=None, report_titles=True):
                          # initialize display-change flag before use
    """
    Create violin plots for the given dataframes with custom settings.

    Parameters:
    - dataframes (dict): Dictionary of DataFrames keyed by identifiers.
    - keys_ordered (list): List of keys to order the plots.
    - color_dict (dict): Dictionary mapping keys to colors.
    - title_dict (dict): Dictionary mapping keys to titles for each subplot.
    - display (str): Label to describe the content or type of the plot for saving.
    - n_rows (int): Number of rows in the subplot grid.
    - split_by_sex (bool): Whether to split the data by sex and create separate plots for each.
    - highlight_column (str, optional): Column name to use for highlighting (e.g. a certain diagnosis).
    - highlight_color (str): Color of the highlight points.
    - highlight_marker (str): Marker style for highlighted points.
    - highlight_size (int): Size of the highlighted markers.
    - alpha (float): Transparency of the highlight markers.
    - report_titles (bool): Whether to print a report of title generation. Default: True

    Returns:
    - fig (Figure): The figure object containing the plots.
    - axes (array): Array of Axes objects containing the subplots.
    """

    # Generate titles with optional reporting
    if report_titles:
        title_mapping = get_titles_with_report(keys_ordered, title_dict)
    else:
        title_mapping = {key: get_title_with_fallback(key, title_dict) for key in keys_ordered}

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharey=True, figsize=(n_cols * 2, n_rows * 10))
    plt.subplots_adjust(wspace=0, hspace=0.1)

    hue_column=truth
    change_display=False

    if n_rows == 1:
        axes = np.array([axes])

    ax_rav = axes.ravel()

    for i, (key, ax) in enumerate(zip(keys_ordered, ax_rav)):
        df = dataframes[key].copy()
        df['ones'] = key
        color = color_dict[key]

        palette = {0: adjust_alpha(color, 0.5), 1: adjust_alpha(color, 1)}

        if inner_detail == "quart":
            inner = "quart"
        elif inner_detail == "ci":
            inner = None  # No inner lines; we will calculate CIs manually

        # Set consistent x-axis for all subplots
        ax.set_xlim(-0.5, 0.5)

        if split_by_sex:
            for sex in df['SEX'].unique():
                palette = {0: adjust_alpha(color, 0.7), 1: adjust_alpha(color, 0.7)}
                df_sex = df[df['SEX'] == sex]
                sns.violinplot(data=df_sex, y="y_pred", x="ones", hue=hue_column, split=True, dodge="auto", gap=gap, inner=None, ax=ax, #hue_column indicates the column to choose for split of half-violins
                               linecolor='white', linewidth=2, palette=palette, saturation=1)
                # Adjusting the style for the specific sex
                if sex == 'Female':
                    for artist in ax.collections:
                        artist.set_hatch('////')

            # Use the pre-generated title mapping
            plot_title = title_mapping[key]
            ax.set_title(plot_title, fontsize=24, pad=20)
        else:
            sns.violinplot(data=df, y="y_pred", x="ones", hue=hue_column, split=True, dodge="auto", gap=gap, inner=inner, ax=ax,
                           linecolor='white', linewidth=4, palette=palette, saturation=1)

            # Use the pre-generated title mapping
            plot_title = title_mapping[key]
            ax.set_title(plot_title, fontsize=font_size, pad=15, rotation=45, horizontalalignment='left')

        # Overlay highlight markers and corresponding legend if the feature exists
        if highlight_column and highlight_column in df.columns:
            highlight_cases = df[df[highlight_column] == 1]  # Select rows where feature is present

            change_display= False

            # Use fixed x positions to avoid affecting violin sizes
            x_positions = np.zeros(len(highlight_cases))

            # Adjust x-coordinates based on 'status' (truth column) for split violin alignment
            # Use smaller fixed offsets instead of proportional positioning
            x_positions[highlight_cases[truth] == 0] = -0.12 + np.random.normal(0, 0.02, sum(highlight_cases[truth] == 0))
            x_positions[highlight_cases[truth] == 1] = 0.12 + np.random.normal(0, 0.02, sum(highlight_cases[truth] == 1))

            ax.scatter(
                x_positions, highlight_cases["y_pred"],
                color=highlight_color, marker=highlight_marker,
                s=highlight_size, alpha=alpha, edgecolors="black",
                zorder=10  # Ensure dots are drawn on top
            )
            # Add a legend for the highlighted points
            scatter_legend = mlines.Line2D(
                [], [],
                color=highlight_color,
                marker=highlight_marker,
                linestyle='None',
                markersize=np.sqrt(highlight_size)*16,  # Scale to reasonable size
                markeredgecolor='black',
                alpha=alpha,
                label=highlight_column.replace('_', ' ').title()
            )

            # Position it near the existing legend
            fig.legend(
                handles=[scatter_legend],
                loc='lower right',
                bbox_to_anchor=(0.9, 0.1),
                fontsize=font_size-3,
                frameon=False,
                handletextpad=0.5,
                prop={'family': 'Arial', 'weight': 'normal'}
            )

        # Reset all axes to consistent sizes
        ax.set_xlim(-0.5, 0.5)
        ax.legend().set_visible(False)
        ax.set_xlabel('')
        ax.set_xticks([])
        ax.set_ylabel('', fontsize=font_size+4)
        ax.set_ylim((0, 1))
        ax.tick_params(axis='y', labelsize=font_size+4)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(True)

    for ax in ax_rav:
        for c in ax.collections:
            c.set_edgecolor('face')
            c.set_linewidth(4)

    for i in range(n_rows):
        axes[i, 0].set_ylabel('Predicted Probability', fontsize=font_size+4)
        axes[i, 0].spines['left'].set_visible(True)
        axes[i, -1].spines['right'].set_visible(True)

    if split_by_sex:
        custom_lines = [
            patches.Patch(facecolor=adjust_alpha('#000000', 1.0), edgecolor='white', linewidth=2, label='Male'),
            patches.Patch(facecolor=adjust_alpha('#000000', 0.7), edgecolor='white', linewidth=2, hatch='////', label='Female')
        ]
        fig.legend(handles=custom_lines, loc='upper left', bbox_to_anchor=(0.13, 0.88),  fontsize=font_size, title=None, frameon=False)


    n_cases = dataframes[keys_ordered[0]][truth].sum()
    n_controls = len(dataframes[keys_ordered[0]]) - n_cases

    if not split_by_sex:
        tn_patch = patches.Patch(facecolor=adjust_alpha('#808080', 0.3), edgecolor=adjust_alpha('#808080', 1), linewidth=3, label=f'Controls (n={n_controls})')  # Grey color
        tp_patch = patches.Patch(facecolor=adjust_alpha('#808080', 1), edgecolor=adjust_alpha('#808080', 1), linewidth=3, label=f'Cases (n={n_cases})')
        fig.legend(handles=[tn_patch, tp_patch], bbox_to_anchor=(0.13, 0.1), loc='lower left', fontsize=font_size-3, title=None, frameon=False)

    fig.text(0.5, 0.05, title_display, ha='center', fontsize=font_size+6)

    if change_display: #if highlight_column is activated, this needs to be passed to the figure name
        display=display+highlight_column

    if fig_path:
        svg_path = os.path.join(fig_path, f"Violins_{display}.svg")
        fig.savefig(svg_path, format='svg', bbox_inches='tight', transparent=True)

    return fig, axes

#################################################################################################################################
########################################    Precision Recall Curves    ###############################################################################
###################################################################################################################################



# def plot_precision_recall_curves(
#     dataframes: Dict[str, pd.DataFrame],
#     keys_ordered: List[str],
#     colors: Dict[str, str],
#     fig: matplotlib.figure.Figure,
#     ax: matplotlib.axes.Axes,
#     y_label: Optional[str] = None,
#     x_label: Optional[str] = None,
#     display: str = 'Precision-Recall-Curve',
#     xlim: Tuple[float, float] = (0, 1),
#     ylim: Tuple[float, float] = (0, 1),
#     fill_bet: bool = False,
#     title: str = '',
#     fig_path: Optional[str] = None,
#     line_style: str = '-',
#     dotted_keys: Optional[List[str]] = None,
#     plot_legend: bool = True,
#     lw: int = 2,
#     font_size: int = 16,
#     truth: str = "status_cancerreg",
#     # NEW PARAMETERS FOR EXTERNAL PRC DATAFRAME
#     external_prc_df: Optional[pd.DataFrame] = None,
#     external_keys: Optional[List[str]] = None,
#     external_colors: Optional[Dict[str, str]] = None,
#     external_line_styles: Optional[Dict[str, str]] = None,
#     external_curve_type: str = 'precision_at_recall',  # 'precision_at_recall' or 'recall_at_precision'
#     external_recall_points: Optional[np.ndarray] = None  # Custom recall points for external data
# ) -> None:
#     """
#     Plots overlaying precision-recall curves for multiple datasets, including external PRC DataFrame.

#     Parameters:
#     - dataframes: Dictionary of DataFrames, each containing columns 'y_pred' and the ground truth.
#     - keys_ordered: List of keys in the plotting order (must match keys in `dataframes` and `colors`).
#     - colors: Dictionary mapping each key to a color.
#     - fig: Matplotlib Figure object.
#     - ax: Matplotlib Axes object.
#     - y_label: Optional custom label for the y-axis.
#     - x_label: Optional custom label for the x-axis.
#     - display: Title label to display in the legend and title.
#     - xlim: X-axis limits.
#     - ylim: Y-axis limits.
#     - fill_bet: Whether to fill between standard deviation bounds of interpolated PR curves.
#     - title: Title of the figure (currently unused but could be used for saving).
#     - fig_path: Directory path to save the SVG figure.
#     - line_style: Default line style for unknown estimators.
#     - dotted_keys: Keys to be plotted using dotted lines (overrides estimator-based style).
#     - plot_legend: Whether to display the legend.
#     - lw: Line width for the curves.
#     - font_size: Font size for text.
#     - truth: Column name for true labels in each DataFrame.
#     - external_prc_df: DataFrame with PRC values (rows=models, columns=PRC values at different points)
#     - external_keys: List of row names from external_prc_df to plot
#     - external_colors: Colors for external curves
#     - external_line_styles: Line styles for external curves
#     - external_curve_type: 'precision_at_recall' or 'recall_at_precision'
#     - external_recall_points: Custom recall points (0-1) for external data. If None, uses evenly spaced points.
#     """

#     def create_clean_label(key):
#         """Create a clean label from different key formats."""
#         label_parts = key.split('_')

#         if 'Model' in label_parts:
#             model_idx = label_parts.index('Model')
#             after_model = label_parts[model_idx+1:]  # Everything after 'Model'
#             before_model = label_parts[:model_idx]   # Everything before 'Model'

#             # Handle different formats
#             if len(after_model) >= 2:
#                 # Format: "AOU_all_cca_Model_TOP5_CatBoost_model_mean"
#                 # TOP5 should be the first part after 'Model'
#                 top_part = after_model[0] if after_model else ""
#                 # AOU should be the first part before 'Model'
#                 aou_part = before_model[0] if before_model else ""
#                 # CatBoost should be the second part after 'Model'
#                 model_type = after_model[1] if len(after_model) > 1 else ""

#             elif len(after_model) == 1:
#                 # Format: "CatBoost_all_cca_Model_TOP5"
#                 # Estimator is before Model, TOP5 is after Model
#                 estimator = before_model[0] if before_model else ""
#                 cohort = before_model[1] if len(before_model) > 1 else ""  # e.g., "all"
#                 top_part = after_model[0] if after_model else ""  # e.g., "TOP5"

#                 # Use estimator as model_type, skip "all" cohort
#                 model_type = estimator
#                 aou_part = cohort if cohort != "all" else ""  # Skip "all"

#             else:
#                 # Fallback
#                 top_part = ""
#                 aou_part = before_model[0] if before_model else ""
#                 model_type = ""

#             # Construct the clean label in desired format: TOP5 - AOU - CatBoost
#             clean_label = f"{top_part} - {aou_part} - {model_type}"

#             # Clean up any empty parts
#             parts = [part.strip() for part in clean_label.split(' - ') if part.strip()]
#             clean_label = ' - '.join(parts)

#         else:
#             clean_label = key

#         return clean_label

#     if dotted_keys is None:
#         dotted_keys = []
#     if external_prc_df is None:
#         external_prc_df = pd.DataFrame()
#     if external_keys is None:
#         external_keys = []
#     if external_colors is None:
#         external_colors = {}
#     if external_line_styles is None:
#         external_line_styles = {}

#     mean_precisions = []
#     base_recall = np.linspace(0, 1, 100)

#     # Plot regular dataframe-based curves
#     for key in keys_ordered:
#         df = dataframes[key]
#         precision, recall, _ = precision_recall_curve(df[truth], df["y_pred"])
#         pr_auc = auc(recall, precision)

#         # Ensure non-increasing precision for interpolation
#         precision_inv = np.fliplr([precision])[0]
#         recall_inv = np.fliplr([recall])[0]
#         for j in range(len(precision_inv) - 2, -1, -1):
#             if precision_inv[j + 1] > precision_inv[j]:
#                 precision_inv[j] = precision_inv[j + 1]

#         decreasing_max_precision = np.maximum.accumulate(precision_inv[::-1])[::-1]
#         mean_precision = np.interp(base_recall, recall[::-1], decreasing_max_precision)
#         mean_precisions.append(mean_precision)

#         # Determine line style
#         estimator = key.split('_')[0]
#         linestyle = line_style
#         if key in dotted_keys:
#             linestyle = ':'
#         elif estimator == "CatBoost":
#             linestyle = '-'
#         elif estimator == "RFC":
#             linestyle = ':'

#         # Construct label using the clean label function
#         clean_label = create_clean_label(key)
#         ax.plot(recall, precision, lw=lw, linestyle=linestyle,
#                 color=colors[key], label=f'{clean_label} ({pr_auc:.2f})')

#     # Plot external PRC curves from DataFrame
#     if not external_prc_df.empty and external_keys:
#         num_points = external_prc_df.shape[1]

#         # Set up recall points (x-axis values)
#         if external_recall_points is not None:
#             if len(external_recall_points) != num_points:
#                 raise ValueError(f"Length of external_recall_points ({len(external_recall_points)}) "
#                                f"must match number of columns in external_prc_df ({num_points})")
#             recall_points = external_recall_points
#         else:
#             # Default: evenly spaced recall points from 0 to 1
#             recall_points = np.linspace(0, 1, num_points)

#         for ext_key in external_keys:
#             if ext_key in external_prc_df.index:
#                 # Get the row data (PRC values)
#                 prc_values = external_prc_df.loc[ext_key].values

#                 # Handle the curve type
#                 if external_curve_type == 'precision_at_recall':
#                     # Values are precision at fixed recall points
#                     recall_ext = recall_points.copy()
#                     precision_ext = prc_values.copy()

#                     # Convert to 0-1 range if values are in 0-100 range
#                     if np.max(precision_ext) > 1.0:
#                         precision_ext = precision_ext / 100.0

#                 elif external_curve_type == 'recall_at_precision':
#                     # Values are recall at fixed precision points
#                     precision_ext = recall_points.copy()  # Using points as precision values
#                     recall_ext = prc_values.copy()

#                     # Convert to 0-1 range if values are in 0-100 range
#                     if np.max(recall_ext) > 1.0:
#                         recall_ext = recall_ext / 100.0

#                     # For plotting, we typically want precision on y-axis, recall on x-axis
#                     # So we might need to swap and interpolate
#                     # This is more complex as we need to handle the inversion properly

#                 else:
#                     raise ValueError("external_curve_type must be 'precision_at_recall' or 'recall_at_precision'")

#                 # Remove any NaN or infinite values
#                 valid_mask = np.isfinite(precision_ext) & np.isfinite(recall_ext)
#                 precision_ext = precision_ext[valid_mask]
#                 recall_ext = recall_ext[valid_mask]

#                 # Ensure values are within [0, 1] range
#                 precision_ext = np.clip(precision_ext, 0, 1)
#                 recall_ext = np.clip(recall_ext, 0, 1)

#                 # Calculate AUC if we have enough points
#                 if len(precision_ext) > 1 and len(recall_ext) > 1:
#                     # Sort by recall for proper AUC calculation
#                     sort_idx = np.argsort(recall_ext)
#                     recall_sorted = recall_ext[sort_idx]
#                     precision_sorted = precision_ext[sort_idx]

#                     # Remove duplicates in recall values for AUC calculation
#                     unique_recall, unique_indices = np.unique(recall_sorted, return_index=True)
#                     unique_precision = precision_sorted[unique_indices]

#                     if len(unique_recall) > 1:
#                         pr_auc_ext = auc(unique_recall, unique_precision)
#                     else:
#                         pr_auc_ext = 0.0
#                 else:
#                     pr_auc_ext = 0.0

#                 # Get color and line style
#                 color_ext = external_colors.get(ext_key, 'black')
#                 linestyle_ext = external_line_styles.get(ext_key, line_style)

#                 # Use the same clean label function for external keys
#                 clean_label = create_clean_label(ext_key)

#                 # Plot external curve
#                 ax.plot(recall_ext, precision_ext, lw=lw, linestyle=linestyle_ext,
#                         color=color_ext, label=f'{clean_label} ({pr_auc_ext:.2f})', alpha=0.8)

#                 # Add to mean precision calculation if desired
#                 if fill_bet and len(precision_ext) > 1 and len(recall_ext) > 1:
#                     try:
#                         # Sort by recall for interpolation
#                         sort_idx = np.argsort(recall_ext)
#                         recall_sorted = recall_ext[sort_idx]
#                         precision_sorted = precision_ext[sort_idx]

#                         # Interpolate to base recall points
#                         mean_precision_ext = np.interp(base_recall, recall_sorted, precision_sorted)
#                         mean_precisions.append(mean_precision_ext)
#                     except Exception as e:
#                         print(f"Warning: Could not interpolate external curve {ext_key}: {e}")

#     # Plot average curve with confidence bands (only if fill_bet is True)
#     if fill_bet and mean_precisions:
#         fig.set_size_inches(5, 4.1)
#         mean_precisions = np.array(mean_precisions)
#         mean_precision = mean_precisions.mean(axis=0)
#         std_precision = mean_precisions.std(axis=0)

#         upper = np.minimum(mean_precision + std_precision, 1)
#         lower = np.maximum(mean_precision - std_precision, 0)
#         ax.fill_between(base_recall, lower, upper, color='grey', alpha=0.2)

#     # Axes formatting
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
#     ax.tick_params(axis='both', which='major', pad=1, labelsize=font_size)
#     ax.tick_params(axis='both', which='minor', pad=1)
#     ax.set_xlabel(x_label or 'Recall (TP / (TP + FN))', fontsize=font_size)
#     ax.set_ylabel(y_label or 'Precision (TP / (TP + FP))', fontsize=font_size)
#     ax.xaxis.set_tick_params(pad=-5)
#     ax.yaxis.set_tick_params(pad=-5)
#     ax.set_title(display, fontsize=font_size + 2, pad=5)

#     # Legend
#     if plot_legend:
#         font_prop = fm.FontProperties(family='Arial', size=font_size -2, stretch='condensed')
#         ax.legend(loc="upper right", bbox_to_anchor=(1.01, 1), frameon=True, prop=font_prop)

#     # Border thickness
#     for spine in ax.spines.values():
#         spine.set_linewidth(0.8)

#     # Save figure if path provided
#     if fig_path:
#         os.makedirs(fig_path, exist_ok=True)
#         svg_path = os.path.join(fig_path, f"Prec_Recall_{display}_{str(ylim)}.svg")
#         fig.canvas.draw()  # Ensure it's fully rendered before saving
#         fig.savefig(svg_path, format='svg', bbox_inches='tight', transparent=True)
#         print(f"Saved PR curve SVG to: {svg_path}")

def plot_precision_recall_curves_old(
    dataframes: Dict[str, pd.DataFrame],
    keys_ordered: List[str],
    colors: Dict[str, str],
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    y_label: Optional[str] = None,
    x_label: Optional[str] = None,
    display: str = 'Precision-Recall-Curve',
    xlim: Tuple[float, float] = (0, 1),
    ylim: Tuple[float, float] = (0, 1),
    fill_bet: bool = False,
    title: str = '',
    fig_path: Optional[str] = None,
    line_style: str = '-',
    dotted_keys: Optional[List[str]] = None,
    plot_legend: bool = True,
    lw: int = 2.5,  # Standardized line width (matches first function)
    font_size: int = 20,  # Standardized font size (matches first function)
    truth: str = "status_cancerreg",
    # NEW PARAMETERS FOR EXTERNAL PRC DATAFRAME
    external_prc_df: Optional[pd.DataFrame] = None,
    external_keys: Optional[List[str]] = None,
    external_colors: Optional[Dict[str, str]] = None,
    external_line_styles: Optional[Dict[str, str]] = None,
    external_curve_type: str = 'precision_at_recall',  # 'precision_at_recall' or 'recall_at_precision'
    external_recall_points: Optional[np.ndarray] = None,  # Custom recall points for external data
    format: str = 'svg',  # Added format parameter to match first function
    add_color_legend: bool = False,  # Added color legend option to match first function
) -> None:
    """
    Plots overlaying precision-recall curves for multiple datasets, including external PRC DataFrame.
    Styled to match plot_rocs_multi_estimator function.
    """

    def create_clean_label(key):
        """Create a clean label from different key formats."""
        label_parts = key.split('_')

        if 'Model' in label_parts:
            model_idx = label_parts.index('Model')
            after_model = label_parts[model_idx+1:]  # Everything after 'Model'
            before_model = label_parts[:model_idx]   # Everything before 'Model'

            # Handle different formats
            if len(after_model) >= 2:
                # Format: "AOU_all_cca_Model_TOP5_CatBoost_model_mean"
                top_part = after_model[0] if after_model else ""
                aou_part = before_model[0] if before_model else ""
                model_type = after_model[1] if len(after_model) > 1 else ""

            # NEW Case 2: "..._Model_'TOP5 - PMBB'" (single token with a suffix after ' - ')
            elif len(after_model) == 1 and ' - ' in after_model[0]:
                # Example: "CatBoost_all_cca_Model_TOP5 - PMBB"
                top_raw = after_model[0]
                top_part, suffix_cohort = [s.strip() for s in top_raw.split(' - ', 1)]
                model_type = before_model[0] if before_model else ""  # "CatBoost"
                aou_part = suffix_cohort

            elif len(after_model) == 1:
                # Format: "CatBoost_all_cca_Model_TOP5"
                estimator = before_model[0] if before_model else ""
                cohort = before_model[1] if len(before_model) > 1 else ""
                top_part = after_model[0] if after_model else ""
                model_type = estimator
                aou_part = cohort if cohort != "all" else "UKB"

            else:
                # Fallback
                top_part = ""
                aou_part = before_model[0] if before_model else ""
                model_type = ""

            # Assemble: "TOP5 - PMBB - CatBoost"
            parts = [p.strip() for p in (aou_part, model_type) if p and p.strip()]
            clean_label = ' - '.join(parts)

        else:
            clean_label = key

        return clean_label

    if dotted_keys is None:
        dotted_keys = []
    if external_prc_df is None:
        external_prc_df = pd.DataFrame()
    if external_keys is None:
        external_keys = []
    if external_colors is None:
        external_colors = {}
    if external_line_styles is None:
        external_line_styles = {}

    # Set figure size to match first function
    fig.set_size_inches(8, 6.5)

    # Set axis labels and tick formatting to match first function
    ax.set_xlabel(x_label or 'Recall (TP / (TP + FN))', fontsize=font_size)
    ax.set_ylabel(y_label or 'Precision (TP / (TP + FP))', fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size, pad=0)  # Match first function's tick styling

    mean_precisions = []
    base_recall = np.linspace(0, 1, 100)
    plot_count = 0  # Track successful plots like first function

    # Plot regular dataframe-based curves
    for key in keys_ordered:
        try:
            df = dataframes[key]
            precision, recall, _ = precision_recall_curve(df[truth], df["y_pred"])
            pr_auc = auc(recall, precision)

            # Ensure non-increasing precision for interpolation
            precision_inv = np.fliplr([precision])[0]
            recall_inv = np.fliplr([recall])[0]
            for j in range(len(precision_inv) - 2, -1, -1):
                if precision_inv[j + 1] > precision_inv[j]:
                    precision_inv[j] = precision_inv[j + 1]

            decreasing_max_precision = np.maximum.accumulate(precision_inv[::-1])[::-1]
            mean_precision = np.interp(base_recall, recall[::-1], decreasing_max_precision)
            mean_precisions.append(mean_precision)

            # Determine line style
            estimator = key.split('_')[0]
            linestyle = line_style
            if key in dotted_keys:
                linestyle = ':'
            elif estimator == "CatBoost":
                linestyle = '-'
            elif estimator == "RFC":
                linestyle = ':'

            # Construct label using the clean label function
            clean_label = create_clean_label(key)
            ax.plot(recall, precision, lw=lw, linestyle=linestyle,
                    color=colors[key], label=f'{clean_label} ({pr_auc:.2f})', alpha=1.0)

            plot_count += 1

        except Exception as e:
            print(f"Warning: Error processing key '{key}': {e}")
            continue

    # Plot external PRC curves from DataFrame
    if not external_prc_df.empty and external_keys:
        num_points = external_prc_df.shape[1]

        # Set up recall points (x-axis values)
        if external_recall_points is not None:
            if len(external_recall_points) != num_points:
                raise ValueError(f"Length of external_recall_points ({len(external_recall_points)}) "
                               f"must match number of columns in external_prc_df ({num_points})")
            recall_points = external_recall_points
        else:
            # Default: evenly spaced recall points from 0 to 1
            recall_points = np.linspace(0, 1, num_points)

        for ext_key in external_keys:
            try:
                if ext_key in external_prc_df.index:
                    # Get the row data (PRC values)
                    prc_values = external_prc_df.loc[ext_key].values

                    # Handle the curve type
                    if external_curve_type == 'precision_at_recall':
                        # Values are precision at fixed recall points
                        recall_ext = recall_points.copy()
                        precision_ext = prc_values.copy()

                        # Convert to 0-1 range if values are in 0-100 range
                        if np.max(precision_ext) > 1.0:
                            precision_ext = precision_ext / 100.0

                    elif external_curve_type == 'recall_at_precision':
                        # Values are recall at fixed precision points
                        precision_ext = recall_points.copy()  # Using points as precision values
                        recall_ext = prc_values.copy()

                        # Convert to 0-1 range if values are in 0-100 range
                        if np.max(recall_ext) > 1.0:
                            recall_ext = recall_ext / 100.0

                    else:
                        raise ValueError("external_curve_type must be 'precision_at_recall' or 'recall_at_precision'")

                    # Remove any NaN or infinite values
                    valid_mask = np.isfinite(precision_ext) & np.isfinite(recall_ext)
                    precision_ext = precision_ext[valid_mask]
                    recall_ext = recall_ext[valid_mask]

                    # Ensure values are within [0, 1] range
                    precision_ext = np.clip(precision_ext, 0, 1)
                    recall_ext = np.clip(recall_ext, 0, 1)

                    # Calculate AUC if we have enough points
                    if len(precision_ext) > 1 and len(recall_ext) > 1:
                        # Sort by recall for proper AUC calculation
                        sort_idx = np.argsort(recall_ext)
                        recall_sorted = recall_ext[sort_idx]
                        precision_sorted = precision_ext[sort_idx]

                        # Remove duplicates in recall values for AUC calculation
                        unique_recall, unique_indices = np.unique(recall_sorted, return_index=True)
                        unique_precision = precision_sorted[unique_indices]

                        if len(unique_recall) > 1:
                            pr_auc_ext = auc(unique_recall, unique_precision)
                        else:
                            pr_auc_ext = 0.0
                    else:
                        pr_auc_ext = 0.0

                    # Get color and line style
                    color_ext = external_colors.get(ext_key, 'black')
                    linestyle_ext = external_line_styles.get(ext_key, line_style)

                    # Use the same clean label function for external keys
                    clean_label = create_clean_label(ext_key)

                    # Plot external curve
                    ax.plot(recall_ext, precision_ext, lw=lw, linestyle=linestyle_ext,
                            color=color_ext, label=f'{clean_label} ({pr_auc_ext:.2f})', alpha=1.0)

                    # Add to mean precision calculation if desired
                    if fill_bet and len(precision_ext) > 1 and len(recall_ext) > 1:
                        try:
                            # Sort by recall for interpolation
                            sort_idx = np.argsort(recall_ext)
                            recall_sorted = recall_ext[sort_idx]
                            precision_sorted = precision_ext[sort_idx]

                            # Interpolate to base recall points
                            mean_precision_ext = np.interp(base_recall, recall_sorted, precision_sorted)
                            mean_precisions.append(mean_precision_ext)
                        except Exception as e:
                            print(f"Warning: Could not interpolate external curve {ext_key}: {e}")

                    plot_count += 1

            except Exception as e:
                print(f"Warning: Error processing external key '{ext_key}': {e}")
                continue

    # Plot average curve with confidence bands (only if fill_bet is True)
    if fill_bet and mean_precisions:
        mean_precisions = np.array(mean_precisions)
        mean_precision = mean_precisions.mean(axis=0)
        std_precision = mean_precisions.std(axis=0)

        upper = np.minimum(mean_precision + std_precision, 1)
        lower = np.maximum(mean_precision - std_precision, 0)
        ax.fill_between(base_recall, lower, upper, color='grey', alpha=0.3)

    # Only set title and legends if we actually plotted something (matching first function)
    if plot_count > 0:
        # Set title to match first function styling
        if title:
            ax.set_title(title, fontsize=font_size, pad=10)
        elif display:
            ax.set_title(display, fontsize=font_size, pad=10)

        # Set axis limits to match first function
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Main legend (styled to match first function)
        if plot_legend:
            handles, labels = ax.get_legend_handles_labels()
            if handles:  # Only add legend if we have items to display
                legend = ax.legend(handles=handles, loc='upper right', frameon=False,
                                 fontsize=font_size)  # Match first function's font size reduction
                ax.add_artist(legend)

        # Optional second legend for color scheme (matching first function)
        if add_color_legend:
            from matplotlib.lines import Line2D
            color_legend_elements = [
                Line2D([0], [0], color='#ff6666', lw=4, label='UKB'),
                Line2D([0], [0], color='#6666ff', lw=4, label='PMBB'),
                Line2D([0], [0], color='#30cc62', lw=4, label='AOU'),
            ]
            second_legend = ax.legend(
                handles=color_legend_elements,
                title='Dataset',
                loc='center right',
                bbox_to_anchor=(1, 0.65),
                frameon=False,
                fontsize=font_size,
                title_fontsize=font_size
            )
            ax.add_artist(second_legend)

        # Save figure if path provided (matching first function's save logic)
        if fig_path:
            try:
                # Create directory if it doesn't exist
                os.makedirs(fig_path, exist_ok=True)

                # Generate filename similar to first function

                svg_path = os.path.join(fig_path, f"Prec_Recall_{display}_{str(ylim)}.svg")
                fig.canvas.draw()  # Ensure it's fully rendered before saving
                fig.savefig(svg_path, format=format, bbox_inches='tight', transparent=True)
                print(f"Saved precision-recall curve to: {svg_path}")

            except Exception as e:
                print(f"Error saving figure: {e}")

        plt.show()  # Show the plot like first function
    else:
        print(f"Warning: No valid data to plot")


import re
from typing import Dict, List, Optional, Set, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

class LegendDimensionManager:
    """
    Manages multiple legend dimensions for precision-recall curves.

    This class extracts semantic information from keys and allows users to
    control which dimensions appear in legends.
    """

    def __init__(self):
        # Default dimension patterns - users can extend these
        self.dimension_patterns = {
            'estimator': {
                'patterns': ['RFC', 'XGB', 'CatBoost', 'SVM', 'LogReg', 'AMAP-RFC'],
                'priority': 1,
                'legend_title': 'Estimator'
            },
            'model_type': {
                'patterns': ['Model_A', 'Model_B', 'Model_C', 'Model_D', 'Model_E',
                           'Model_Demographics', 'Model_Blood', 'Model_SNP', 'Model_Metabolomics',
                           'Model_TOP15', 'Model_TOP30', 'Model_TOP75', 'Model_AMAP-RFC'],
                'priority': 2,
                'legend_title': 'Model'
            },
            'cohort': {
                'patterns': ['all', 'par', 'unscreened', 'UKB', 'AOU', 'PMBB'],
                'priority': 3,
                'legend_title': 'Cohort'
            },
            'biomarker': {
                'patterns': ['aMAP', 'APRI', 'FIB4', 'NFS', 'AFP', 'LiverRisk', 'Liver cirrhosis', 'Cirrhosis'],
                'priority': 4,
                'legend_title': 'Biomarker'
            },
            'sex': {
                'patterns': ['Male', 'Female', 'male', 'female'],
                'priority': 5,
                'legend_title': 'Sex'
            }
        }

    def add_dimension(self, dimension_name: str, patterns: List[str],
                     priority: int = 10, legend_title: str = None):
        """Add a custom dimension with patterns to match"""
        self.dimension_patterns[dimension_name] = {
            'patterns': patterns,
            'priority': priority,
            'legend_title': legend_title or dimension_name.title()
        }

    def extract_dimensions(self, key: str) -> Dict[str, str]:
        """Extract all dimension values from a key"""
        dimensions = {}

        for dim_name, dim_config in self.dimension_patterns.items():
            for pattern in dim_config['patterns']:
                # For patterns with underscores, match exactly
                if '_' in pattern:
                    if pattern in key:
                        dimensions[dim_name] = pattern
                        break
                else:
                    # Use word boundaries for single words but be more flexible
                    if re.search(rf'(^|_){re.escape(pattern)}($|_)', key, re.IGNORECASE):
                        dimensions[dim_name] = pattern
                        break

        return dimensions

    def group_keys_by_dimensions(self, keys: List[str],
                               group_by_dims: List[str]) -> Dict[Tuple, List[str]]:
        """Group keys by specified dimensions"""
        groups = {}

        for key in keys:
            dimensions = self.extract_dimensions(key)
            group_key = tuple(dimensions.get(dim, 'Unknown') for dim in group_by_dims)

            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(key)

        return groups

    def create_legend_elements(self, keys: List[str], colors: Dict[str, str],
                             legend_dims: List[str],
                             line_styles: Dict[str, str] = None,
                             line_width: float = 2,
                             show_scores: bool = False,
                             score_dict: Dict[str, float] = None) -> List[mlines.Line2D]:
        """Create legend elements grouped by dimensions"""

        if line_styles is None:
            line_styles = {}

        if score_dict is None:
            score_dict = {}

        legend_elements = []
        seen_combinations = set()

        # Group keys by the legend dimensions
        groups = self.group_keys_by_dimensions(keys, legend_dims)

        # Sort groups by dimension priority and values
        sorted_groups = sorted(groups.items(),
                             key=lambda x: [self.dimension_patterns.get(dim, {}).get('priority', 999)
                                          for dim in legend_dims] + list(x[0]))

        for group_key, group_keys in sorted_groups:
            if group_key not in seen_combinations:
                # Use the first key in the group as representative
                repr_key = group_keys[0]

                # Create label from dimension values
                label_parts = []
                for i, dim in enumerate(legend_dims):
                    if group_key[i] != 'Unknown':
                        label_parts.append(group_key[i])

                if label_parts:  # Only add if we have meaningful dimension values
                    label = ' - '.join(label_parts)

                    # Add score if requested and available
                    if show_scores and repr_key in score_dict:
                        label = f'{label} ({score_dict[repr_key]:.2f})'

                    color = colors.get(repr_key, 'black')
                    linestyle = line_styles.get(repr_key, '-')

                    legend_elements.append(
                        mlines.Line2D([], [], color=color, linestyle=linestyle,
                                    linewidth=line_width, label=label)
                    )
                    seen_combinations.add(group_key)

        return legend_elements


def plot_precision_recall_curves(
    dataframes: Dict[str, 'pd.DataFrame'],
    keys_ordered: List[str],
    colors: Dict[str, str],
    fig: 'matplotlib.figure.Figure',
    ax: 'matplotlib.axes.Axes',
    # Legend control parameters
    legend_manager: LegendDimensionManager = None,
    main_legend_dims: List[str] = None,
    secondary_legend_dims: List[str] = None,
    show_main_legend: bool = True,
    show_secondary_legend: bool = False,
    show_scores_in_legend: bool = True,  # New parameter to control score display
    main_legend_pos: str = 'upper right',
    secondary_legend_pos: str = 'center right',
    # Existing parameters
    y_label: Optional[str] = None,
    x_label: Optional[str] = None,
    display: str = 'Precision-Recall-Curve',
    xlim: Tuple[float, float] = (0, 1),
    ylim: Tuple[float, float] = (0, 1),
    fill_bet: bool = False,
    title: str = '',
    fig_path: Optional[str] = None,
    line_style: str = '-',
    dotted_keys: Optional[List[str]] = None,
    lw: int = 2,
    font_size: int = 20,
    truth: str = "status_cancerreg",
    format: str = 'svg',
    **kwargs
) -> None:
    """
    Enhanced precision-recall curve plotting with flexible legend management.

    Parameters:
    -----------
    show_scores_in_legend : bool
        Whether to show AUPRC scores in legend labels
    """

    # Initialize legend manager if not provided
    if legend_manager is None:
        legend_manager = LegendDimensionManager()

    # Set default legend dimensions if not specified
    if main_legend_dims is None:
        main_legend_dims = ['model_type', 'biomarker']

    if dotted_keys is None:
        dotted_keys = []

    # Import required modules for PR curve calculation
    from sklearn.metrics import precision_recall_curve, auc
    import numpy as np
    import os

    # Set figure size and basic formatting
    fig.set_size_inches(8, 6.5)
    ax.set_xlabel(x_label or 'Recall (TP / (TP + FN))', fontsize=font_size)
    ax.set_ylabel(y_label or 'Precision (TP / (TP + FP))', fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size, pad=0)

    mean_precisions = []
    base_recall = np.linspace(0, 1, 100)
    plot_count = 0
    score_dict = {}  # Store AUPRC scores for legend

    def create_clean_label(key):
        """Create a clean label from different key formats."""
        label_parts = key.split('_')

        if 'Model' in label_parts:
            model_idx = label_parts.index('Model')
            after_model = label_parts[model_idx+1:]
            before_model = label_parts[:model_idx]

            if len(after_model) >= 2:
                top_part = after_model[0] if after_model else ""
                aou_part = before_model[0] if before_model else ""
                model_type = after_model[1] if len(after_model) > 1 else ""
            elif len(after_model) == 1 and ' - ' in after_model[0]:
                top_raw = after_model[0]
                top_part, suffix_cohort = [s.strip() for s in top_raw.split(' - ', 1)]
                model_type = before_model[0] if before_model else ""
                aou_part = suffix_cohort
            elif len(after_model) == 1:
                estimator = before_model[0] if before_model else ""
                cohort = before_model[1] if len(before_model) > 1 else ""
                top_part = after_model[0] if after_model else ""
                model_type = estimator
                aou_part = cohort if cohort != "all" else "UKB"
            else:
                top_part = ""
                aou_part = before_model[0] if before_model else ""
                model_type = ""

            parts = [p.strip() for p in (aou_part, model_type) if p and p.strip()]
            clean_label = ' - '.join(parts)
        else:
            clean_label = key

        return clean_label

    # Plot precision-recall curves for each key
    for key in keys_ordered:
        try:
            df = dataframes[key]
            precision, recall, _ = precision_recall_curve(df[truth], df["y_pred"])
            pr_auc = auc(recall, precision)

            # Store the score for legend use
            score_dict[key] = pr_auc

            # Ensure non-increasing precision for interpolation
            precision_inv = np.fliplr([precision])[0]
            recall_inv = np.fliplr([recall])[0]
            for j in range(len(precision_inv) - 2, -1, -1):
                if precision_inv[j + 1] > precision_inv[j]:
                    precision_inv[j] = precision_inv[j + 1]

            decreasing_max_precision = np.maximum.accumulate(precision_inv[::-1])[::-1]
            mean_precision = np.interp(base_recall, recall[::-1], decreasing_max_precision)
            mean_precisions.append(mean_precision)

            # Determine line style
            estimator = key.split('_')[0]
            linestyle = line_style
            if key in dotted_keys:
                linestyle = '--'
            elif estimator == "CatBoost":
                linestyle = '-'
            elif estimator == "RFC":
                linestyle = ':'

            # Plot the curve with clean label
            clean_label = create_clean_label(key)
            ax.plot(recall, precision, lw=lw, linestyle=linestyle,
                    color=colors[key], label=f'{clean_label} ({pr_auc:.2f})', alpha=1.0)

            plot_count += 1

        except Exception as e:
            print(f"Warning: Error processing key '{key}': {e}")
            continue

    # Plot average curve with confidence bands (only if fill_bet is True)
    if fill_bet and mean_precisions:
        mean_precisions = np.array(mean_precisions)
        mean_precision = mean_precisions.mean(axis=0)
        std_precision = mean_precisions.std(axis=0)

        upper = np.minimum(mean_precision + std_precision, 1)
        lower = np.maximum(mean_precision - std_precision, 0)
        ax.fill_between(base_recall, lower, upper, color='grey', alpha=0.3)

    # Only set title and legends if we actually plotted something
    if plot_count > 0:
        if title:
            ax.set_title(title, fontsize=font_size, pad=10)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Enhanced legend handling
        if show_main_legend and main_legend_dims:
            # Create line styles dict for dotted keys
            line_styles = {}
            for key in keys_ordered:
                line_styles[key] = '--' if key in dotted_keys else line_style

            # Create main legend elements
            main_legend_elements = legend_manager.create_legend_elements(
                keys_ordered, colors, main_legend_dims, line_styles, lw,
                show_scores=show_scores_in_legend, score_dict=score_dict
            )

            if main_legend_elements:
                main_legend = ax.legend(
                    handles=main_legend_elements,
                    loc=main_legend_pos,
                    frameon=False,
                    fontsize=font_size-2
                )
                ax.add_artist(main_legend)

        # Secondary legend (e.g., for cohorts)
        if show_secondary_legend and secondary_legend_dims:
            line_styles = {}
            for key in keys_ordered:
                line_styles[key] = ':' if key in dotted_keys else line_style

            secondary_legend_elements = legend_manager.create_legend_elements(
                keys_ordered, colors, secondary_legend_dims, line_styles, lw,
                show_scores=show_scores_in_legend, score_dict=score_dict
            )

            if secondary_legend_elements:
                secondary_legend = ax.legend(
                    handles=secondary_legend_elements,
                    title=legend_manager.dimension_patterns[secondary_legend_dims[0]]['legend_title'],
                    loc=secondary_legend_pos,
                    bbox_to_anchor=(1, 0.65),
                    frameon=False,
                    fontsize=font_size-2,
                    title_fontsize=font_size-2
                )
                ax.add_artist(secondary_legend)

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # Save figure if path provided
        if fig_path:
            try:
                os.makedirs(fig_path, exist_ok=True)
                svg_path = os.path.join(fig_path, f"Prec_Recall_{display}_{str(ylim)}.svg")
                fig.canvas.draw()
                fig.savefig(svg_path, format=format, bbox_inches='tight', transparent=True)
                print(f"Saved precision-recall curve to: {svg_path}")
            except Exception as e:
                print(f"Error saving figure: {e}")



        plt.show()
    else:
        print(f"Warning: No valid data to plot")


# Example usage patterns:
def example_usage():
    """Example of how to use the enhanced legend system"""

    # 1. Basic usage with default dimensions
    legend_mgr = LegendDimensionManager()

    # 2. Add custom dimensions
    legend_mgr.add_dimension('validation_type', ['train', 'val', 'test'], priority=6)

    # 3. Different legend configurations for different plots

    # Configuration 1: Show only model types and biomarkers
    plot_precision_recall_curves_enhanced(
        dataframes, keys_ordered, colors, fig, ax,
        legend_manager=legend_mgr,
        main_legend_dims=['model_type', 'biomarker'],
        show_main_legend=True,
        show_secondary_legend=False
    )

    # Configuration 2: Main legend for models, secondary for cohorts
    plot_precision_recall_curves_enhanced(
        dataframes, keys_ordered, colors, fig, ax,
        legend_manager=legend_mgr,
        main_legend_dims=['model_type'],
        secondary_legend_dims=['cohort'],
        show_main_legend=True,
        show_secondary_legend=True
    )

    # Configuration 3: No legends (for publication figures)
    plot_precision_recall_curves_enhanced(
        dataframes, keys_ordered, colors, fig, ax,
        legend_manager=legend_mgr,
        show_main_legend=False,
        show_secondary_legend=False
    )


# Utility functions for common use cases
def create_model_biomarker_legend(keys: List[str], colors: Dict[str, str], **kwargs):
    """Convenience function for model + biomarker legends"""
    legend_mgr = LegendDimensionManager()
    return legend_mgr.create_legend_elements(
        keys, colors, ['model_type', 'biomarker'], **kwargs
    )

def create_cohort_legend(keys: List[str], colors: Dict[str, str], **kwargs):
    """Convenience function for cohort-only legends"""
    legend_mgr = LegendDimensionManager()
    return legend_mgr.create_legend_elements(
        keys, colors, ['cohort'], **kwargs
    )