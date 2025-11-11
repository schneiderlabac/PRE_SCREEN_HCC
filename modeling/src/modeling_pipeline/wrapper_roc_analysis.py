import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
import seaborn as sns
import os
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score, average_precision_score
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from scipy import stats
from itertools import combinations
import os
import re
import yaml
import warnings
warnings.filterwarnings("ignore", message="indexing past lexsort depth may impact performance")
import joblib






#TODO if else statement für cusotm colors einbauen





#Scraping default colors from the default_colors.yaml. In the notebooks, these can be overwritten by userspecific color schemes
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(script_dir, 'custom_colors.yaml')

    with open(default_config_path, 'r') as file:
        _config = yaml.safe_load(file)

    all_scenarios = _config.get("scenarios_colors", {})
    scenario_lists = _config.get("scenario_lists", {})
except (FileNotFoundError, yaml.YAMLError):
    print(f"Error: Could not load default colors from YAML.")

################################################################################################################################
################################################################################################################################
###########################################      Data Load         ###########################################################



def load_tprs(path, model_types="RFC", drop_special_models=None):
    """
    Load TPRs with option to exclude special model variants.

    Parameters:
    ----------
    path : str
        Base path to the models
    model_types : str or list
        Model types to load
    drop_special_models : str or list, optional
        String(s) to exclude from column names. Columns containing any of these
        strings will be dropped. Example: "Sensitivity_No_NA"

    Returns:
    -------
    DataFrame with MultiIndex: (cohort, scenario, model, estimator)
    """
    all_tprs = pd.DataFrame()

    # Ensure model_types is a list
    if isinstance(model_types, str):
        model_types = [model_types]

    # Ensure drop_special_models is a list if provided
    if drop_special_models is not None:
        if isinstance(drop_special_models, str):
            drop_special_models = [drop_special_models]

    for model_type in model_types:
        tprs_joblib_path = os.path.join(path + "/Models/Pipelines/", model_type, "combined_output/val/TPRS_combined.joblib")
        tprs_excel_path = os.path.join(path+  "/Models/Pipelines/", model_type, "combined_output/val/TPRS_combined.xlsx")

        # Load data
        if os.path.exists(tprs_joblib_path):
            tprs = joblib.load(tprs_joblib_path)
            print(f"Loaded joblib TPRs for {model_type}")
        elif os.path.exists(tprs_excel_path):
            tprs = pd.read_excel(tprs_excel_path)
            print(f"Loaded Excel TPRs for {model_type}")
        else:
            print(f"No TPR file found for {model_type}")
            continue

        # Filter out special models if requested
        original_columns = list(tprs.columns)
        if drop_special_models is not None:
            filtered_columns = []
            dropped_columns = []

            for col in original_columns:
                should_drop = any(special_string in col for special_string in drop_special_models)
                if should_drop:
                    dropped_columns.append(col)
                else:
                    filtered_columns.append(col)

            tprs = tprs[filtered_columns]
            print(f"Dropped {len(dropped_columns)} columns containing {drop_special_models}")
            if dropped_columns:
                print(f"Dropped columns: {dropped_columns[:3]}{'...' if len(dropped_columns) > 3 else ''}")

        # Clean column names
        columns = [col.replace('_met', '') for col in tprs.columns]
        tprs.columns = columns

        # Build metadata
        mapper = pd.DataFrame({'col_names': columns})
        mapper["estimator"] = model_type
        mapper['cohort'] = [i.split('_Model')[0] for i in mapper.col_names]
        mapper['scenario'] = [i.split('Model_')[1].split('_')[0] for i in mapper.col_names]
        mapper['model'] = [i.split('_model')[1] for i in mapper.col_names]
        mapper.set_index('col_names', inplace=True)

        # Combine with TPR values
        mapped_tprs = pd.concat([mapper, tprs.transpose()], axis=1)
        mapped_tprs.set_index(['cohort', 'scenario', 'model', 'estimator'], inplace=True)

        all_tprs = pd.concat([all_tprs, mapped_tprs])

    return all_tprs


def load_prediction_values(path, model_types=["RFC"], prefix_keys=True):
    """
    Load prediction values from joblib or Excel files.

    Args:
        model_types: List of model types to load
        path: Base path to the models
        prefix_keys: If True, prefix keys with model_type (default: True)
                    If False, use original keys without model_type prefix

    Returns:
        Dictionary with prediction DataFrames
    """
    all_predictions = {}
    for model_type in model_types:
        pred_excel_path = os.path.join(path+ "/Models/Pipelines/", model_type, "combined_output/val/Prediction_values_combined.xlsx")
        pred_joblib_path = os.path.join(path+ "/Models/Pipelines/", model_type, "combined_output/val/Prediction_values_combined.joblib")

        if os.path.exists(pred_joblib_path):
            try:
                preds = joblib.load(pred_joblib_path)
                if isinstance(preds, dict):
                    for key, df in preds.items():
                        final_key = f"{model_type}_{key}" if prefix_keys else key
                        all_predictions[final_key] = df
                        print(f"Loaded predictions for {final_key} from joblib")
                else:
                    print(f"Unexpected format in joblib file for {model_type}, expected dict but got {type(preds)}")
            except Exception as e:
                print(f"Failed to load joblib predictions for {model_type}: {e}")

        elif os.path.exists(pred_excel_path):
            try:
                excel_file = pd.ExcelFile(pred_excel_path)
                for sheet_name in excel_file.sheet_names:
                    if sheet_name != 'Sheet':  # Skip placeholder sheet
                        final_key = f"{model_type}_{sheet_name}" if prefix_keys else sheet_name
                        all_predictions[final_key] = pd.read_excel(excel_file, sheet_name=sheet_name)
                        print(f"Loaded predictions for {final_key} from Excel")
            except Exception as e:
                print(f"Failed to load Excel predictions for {model_type}: {e}")
        else:
            print(f"No prediction file found for {model_type} (neither joblib nor Excel).")

    return all_predictions


def calculate_tprs_from_pred_val_dict(pred_value_dict, base_fpr=np.linspace(0, 1, 100), single_key=None):
    """
    Calculates interpolated TPRs from a prediction value dict with multi-level keys
    and organizes them in a Multi-Index DataFrame.

    Args:
        pred_value_dict (dict): Keys like 'RFC_all_cca_Model_TOP5', values are DataFrames
                               with 'eid', 'status_cancerreg', 'y_pred' (mean), and
                               'y_pred_val_model_0', 'y_pred_val_model_1', etc.
        base_fpr (np.array): Common FPR values to interpolate on (default: 100 points from 0 to 1).
        single_key (str, optional): If provided, only process this specific key from the dict.
                                   Example: 'RFC_all_cca_Model_TOP5'

    Returns:
        dict: {
            'tprs': pd.DataFrame with Multi-Index ('cohort', 'scenario', 'model', 'estimator')
            'aucs': dict with same structure as keys, values are AUC scores
            'auprcs': dict with same structure as keys, values are AUPRC scores
            'raw_curves': dict with raw ROC curve data
        }
    """

    def parse_key(key):
        """
        Parse keys like 'RFC_all_cca_Model_TOP5' into components.
        Expected format: {estimator}_{cohort}_Model_{scenario}
        """
        parts = key.split('_')
        if len(parts) < 4 or 'Model' not in parts:
            print(f"Warning: Could not parse key '{key}', skipping...")
            return None, None, None

        model_idx = parts.index('Model')

        estimator = parts[0]  # First part is estimator
        cohort = '_'.join(parts[1:model_idx])  # Parts between estimator and 'Model'
        scenario = '_'.join(parts[model_idx+1:])  # Everything after 'Model'

        return estimator, cohort, scenario

    # Storage for results
    all_tprs = []
    aucs = {}
    auprcs = {}
    raw_curves = {}

    # Filter dict if single_key is specified
    if single_key:
        if single_key not in pred_value_dict:
            print(f"Error: Key '{single_key}' not found in prediction dictionary!")
            return {
                'tprs': pd.DataFrame(),
                'aucs': {},
                'auprcs': {},
                'raw_curves': {}
            }
        filtered_dict = {single_key: pred_value_dict[single_key]}
        print(f"Processing single key: '{single_key}'")
    else:
        filtered_dict = pred_value_dict
        print(f"Processing {len(pred_value_dict)} prediction dictionaries...")

    for key, df in filtered_dict.items():
        if df is None:
            print(f"Skipping key '{key}' - DataFrame is None")
            continue

        # Parse the key
        estimator, cohort, scenario = parse_key(key)
        if estimator is None:
            continue

        # Check required columns
        if 'status_cancerreg' not in df.columns:
            print(f"Skipping key '{key}' - missing 'status_cancerreg' column")
            continue

        y_true = df['status_cancerreg']

        # Find all model prediction columns (y_pred_val_model_0, y_pred_val_model_1, etc.)
        pred_cols = [col for col in df.columns if col.startswith('y_pred_val_model_')]

        if not pred_cols:
            print(f"No model prediction columns found for key '{key}'")
            continue

        # Process each model iteration
        for pred_col in pred_cols:
            # Extract model number (e.g., '0' from 'y_pred_val_model_0')
            model_match = re.search(r'y_pred_val_model_(\d+)', pred_col)
            if not model_match:
                continue

            model_num = model_match.group(1)
            model_id = f"_{model_num}"  # Format as '_0', '_1', etc.

            try:
                y_pred = df[pred_col]

                # Calculate ROC curve
                fpr, tpr, thresholds = roc_curve(y_true, y_pred)

                # Interpolate TPR at base FPR points
                tpr_interp = np.interp(base_fpr, fpr, tpr)

                # Calculate metrics
                auc_value = auc(base_fpr, tpr_interp)
                auprc_value = average_precision_score(y_true, y_pred)

                # Create multi-index tuple
                multi_idx = (cohort, scenario, model_id, estimator)

                # Store TPR data
                tpr_data = {
                    'multi_index': multi_idx,
                    'tprs': tpr_interp
                }
                all_tprs.append(tpr_data)

                # Store metrics
                metric_key = f"{estimator}_{cohort}_{scenario}{model_id}"
                aucs[metric_key] = auc_value
                auprcs[metric_key] = auprc_value
                raw_curves[metric_key] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': thresholds
                }

            except Exception as e:
                print(f"Error processing {key} - {pred_col}: {e}")
                continue

    # Create Multi-Index DataFrame for TPRs
    if all_tprs:
        # Extract multi-indices and TPR values
        multi_indices = [item['multi_index'] for item in all_tprs]
        tpr_values = [item['tprs'] for item in all_tprs]

        # Create MultiIndex
        multi_idx = pd.MultiIndex.from_tuples(
            multi_indices,
            names=['cohort', 'scenario', 'model', 'estimator']
        )

        # Create DataFrame with TPRs as rows
        tprs_df = pd.DataFrame(
            tpr_values,
            index=multi_idx,
            columns=[f'fpr_{i:03d}' for i in range(len(base_fpr))]
        )

        print(f"Successfully created TPRs DataFrame with shape: {tprs_df.shape}")
        print(f"Multi-Index levels: {tprs_df.index.names}")

    else:
        print("No valid TPR data found!")
        tprs_df = pd.DataFrame()

    return {
        'tprs': tprs_df,
        'aucs': aucs,
        'auprcs': auprcs,
        'raw_curves': raw_curves
    }





def create_benchmark_dict_from_master(benchmarks_df, cohort_dict, required_non_na_cols=None, require_all=True, row_mask=None):
    """
    Create a dictionary of benchmark dataframes, one for each cohort.

    Parameters
    ----------
    benchmarks_df : pd.DataFrame
        Master benchmark dataframe with 'eid' column
    cohort_dict : dict
        Dictionary mapping cohort names to lists of eids
    required_non_na_cols : str or list[str], optional
        If provided, only rows where these column(s) are non-NA are kept before subsetting.
        Examples:
            required_non_na_cols='AFP'             -> keep rows where AFP not NA
            required_non_na_cols=['AFP','X']      -> if require_all True keep rows where BOTH not NA
    require_all : bool, default True
        If True, require all columns in required_non_na_cols to be non-NA. If False, keep rows where any is non-NA.
    row_mask : pd.Series (bool), optional
        Alternative: pass a boolean mask to filter rows directly. Takes precedence over required_non_na_cols.

    Returns
    -------
    dict
        Dictionary mapping cohort names to benchmark dataframes
    """
    # Apply row-level filtering first (if requested)
    if row_mask is not None:
        master_df = benchmarks_df.loc[row_mask].copy()
    else:
        master_df = benchmarks_df.copy()
        if required_non_na_cols is not None:
            if isinstance(required_non_na_cols, (list, tuple)):
                if require_all:
                    mask = master_df[required_non_na_cols].notna().all(axis=1)
                else:
                    mask = master_df[required_non_na_cols].notna().any(axis=1)
            else:
                mask = master_df[required_non_na_cols].notna()
            master_df = master_df.loc[mask].copy()

    benchmark_dict = {}
    for cohort_name, eids in cohort_dict.items():
        cohort_mask = master_df['eid'].isin(eids)
        cohort_benchmarks = master_df[cohort_mask].copy()
        benchmark_dict[cohort_name] = cohort_benchmarks

        print(f"Cohort '{cohort_name}': {len(cohort_benchmarks)} patients with benchmark data")

        benchmark_cols = [col for col in cohort_benchmarks.columns
                          if col not in ['eid', 'status', 'status_cancerreg']]

        for bcol in benchmark_cols:
            valid_count = cohort_benchmarks[bcol].notna().sum()
            print(f"  {bcol}: {valid_count} valid scores")

    return benchmark_dict

def add_benchmarks_to_mapped_tprs(mapped_tprs, benchmark_dict, benchmark_names=None, n_folds=5, status_col='status'):
    """
    Add benchmark scores to mapped_tprs by calculating TPR values from test scores and true labels.

    Parameters:
    ----------
    mapped_tprs : pd.DataFrame
        Existing multi-index DataFrame with TPR values
        Index: (cohort, scenario, model, estimator)
        Columns: 0-99 (representing TPR values at standardized FPR points)
    benchmark_dict : dict
        Dictionary mapping cohort names to benchmark dataframes
        Each dataframe should have: 'eid', benchmark score columns, 'status'/'status_cancerreg'
    benchmark_names : list, optional
        List of benchmark column names to include. If None, auto-detects.
    n_folds : int
        Number of folds to replicate (to match CV structure). Default: 5

    Returns:
    -------
    pd.DataFrame
        Updated mapped_tprs with benchmark scores included
    """

    # Create a copy to avoid modifying the original
    updated_tprs = mapped_tprs.copy()

    # Get the number of FPR points from existing data (should be 100)
    n_fpr_points = len(mapped_tprs.columns)
    base_fpr = np.linspace(0, 1, n_fpr_points)

    print(f"Processing {len(benchmark_dict)} cohorts...")
    print(f"Using {n_fpr_points} FPR points for interpolation")

    for cohort_name, cohort_df in benchmark_dict.items():
        print(f"\nProcessing cohort: {cohort_name}")

        # Auto-detect benchmark columns if not provided
        if benchmark_names is None:
            detected_benchmarks = [col for col in cohort_df.columns
                                 if col not in ['eid', 'status', 'status_cancerreg']]
        else:
            detected_benchmarks = benchmark_names

        print(f"Found benchmark columns: {detected_benchmarks}")

        # Determine which status column to use
        print(f"Using status column: {status_col}")

        for benchmark_name in detected_benchmarks:
            print(f"  Processing benchmark: {benchmark_name}")

            # Get valid data (non-null predictions and labels)
            valid_mask = (cohort_df[benchmark_name].notna() &
                         cohort_df[status_col].notna())

            if valid_mask.sum() == 0:
                print(f"    Warning: No valid data for {benchmark_name}")
                continue

            n_valid = valid_mask.sum()
            n_cases = cohort_df.loc[valid_mask, status_col].sum()
            print(f"    Valid samples: {n_valid}, Cases: {n_cases}")

            try:
                # Step 1: Calculate ROC curve
                fpr, tpr, thresholds = roc_curve(
                    cohort_df.loc[valid_mask, status_col],  # True labels
                    cohort_df.loc[valid_mask, benchmark_name]  # Test scores
                )


                # Step 2: Calculate AUC for validation (BEFORE interpolation)
                auc_score = auc(fpr, tpr)

                # Step 3: Interpolate TPR values at standardized FPR points
                tpr_interpolated = np.interp(base_fpr, fpr, tpr)
                print(f"    AUC: {auc_score:.3f}")

                # Step 4: Add to mapped_tprs with proper multi-index structure
                # Since benchmarks don't have CV folds, replicate the same TPR curve
                for fold in range(n_folds):
                    # Create multi-index tuple: (cohort, scenario, model, estimator)
                    index_tuple = (
                        cohort_name,      # cohort (e.g., 'all', 'par')
                        benchmark_name,   # scenario (e.g., 'aMAP', 'APRI')
                        f"_{fold}",            # model (0-4 for CV folds)
                        'linear'         # estimator (all benchmarks are linear)
                    )

                    # Add the interpolated TPR values
                    updated_tprs.loc[index_tuple] = tpr_interpolated

                print(f"    Successfully added {benchmark_name} for cohort {cohort_name}")

            except Exception as e:
                print(f"    Error processing {benchmark_name}: {e}")

    print(f"\nFinal mapped_tprs shape: {updated_tprs.shape}")
    return updated_tprs

################################################################################################################################
################################################################################################################################
###########################################      Visuals         ###########################################################
def get_colors(scenario_list, scenarios_colors=None):
    """
    Retrieves color codes for scenarios, with optional override.

    Parameters:
    ----------
    scenario_list : list
        List of scenario identifiers.
    scenarios_colors : dict, optional
        Dictionary mapping scenario names to colors.
        If None, uses the global default colors.

    Returns:
    -------
    dict
        Mapping of scenario identifiers to their color codes.
    """
    if scenarios_colors is None:
        # Use the global dictionary if no custom one is provided
        scenarios_colors = all_scenarios
    return {scenario: scenarios_colors.get(scenario, '#000000') for scenario in scenario_list}


def plot_colorbar(scenarios_colors=None):
    """
    Plots a colorbar based on the given scenarios.
    Parameters:
    ----------
    scenarios : list
        A list of scenario labels.
    scenarios_colors : dict, optional
        Dictionary mapping scenario names to colors.
        If None, uses the global default colors.
    """
    # Use scenarios_colors directly, with fallback to black for missing keys
    if scenarios_colors is None:
        scenarios_colors = {}  # or your global default dict

    fig, ax = plt.subplots(figsize=(5.5, 1.1))
    for i, scenario in enumerate(scenarios_colors):
        color = scenarios_colors.get(scenario, '#000000')  # Default to black if not found
        rect = plt.Rectangle((i * 55, 0), 55, 55, linewidth=2, edgecolor='white', facecolor=color)
        ax.add_patch(rect)
        ax.text(i * 55 + 27.5, -10, scenario, ha='center', va='top', fontsize=10, color='black')

    ax.set_xlim(0, len(scenarios_colors) * 55)
    ax.set_ylim(-20, 55)
    ax.axis('off')
    plt.show()


def plot_roc_curve(test_scores, true_labels, ax=False, label=None, color='#c9c9c9', lw=2.5, linestyle="--", fig_path=None, font_size=16):
    """
    Plots a ROC curve given test scores and true labels.

    Parameters:
    ----------
    test_scores : array-like
        Test scores for the positive class.
        true_labels : array-like
            True labels for the positive class.
            ax : matplotlib.axes, optional
            Axes object to plot on. If None, creates a new figure.
            label : str, optional
            Label for the ROC curve. If None, uses the AUC value.
            color : str, optional"""

    # Calculate ROC curve and AUC

    fpr, tpr, thresholds = roc_curve(true_labels, test_scores)
    roc_auc = auc(fpr, tpr)
    roc_auc = round(roc_auc, 2)
    base_fpr = np.linspace(0, 1, 100)
    tpr = np.interp(base_fpr, fpr, tpr)

    if label is None:
        plot_label = 'aMAP ({:.2f})'.format(roc_auc)
    else:
        plot_label = '{} ({:.2f})'.format(label, roc_auc)

    # Create the ROC curve plot
    if ax == False:
        plt.plot(base_fpr, tpr, color=color, lw=lw, label=plot_label, alpha=1, linestyle=linestyle)
    else:
        ax.plot(base_fpr, tpr, color=color, lw=lw, label=plot_label, alpha=1, linestyle=linestyle)
    return thresholds, fpr, tpr




def plot_rocs(tprs, fig, ax, plot_all=True, y_amap=None, col_line=None, scenario='',
              fill_bet=True, title='', fig_type='', n_splits=5, line_style='-', save_fig=True,
              # Add new parameters for customization
              individual_alpha=0.15,       # Lower alpha for individual lines
              individual_lw=1,             # Thinner individual lines
              individual_color=None,       # Optional different color for individual lines
              mean_lw=2.5,                 # Keep mean line width the same
              mean_alpha=1.0,               # Full opacity for mean line
              fig_path=None,
              font_size=None,
              linewidth=1.5):
    """
    Creates multiple ROC curves on the same plot. Gets called by plot_rocs_flexible and plot_rocs_multi_estimator.

    Parameters:
    ----------
    mapped_tprs : DataFrame
        Multi-indexed DataFrame containing TPR values.
    fig : matplotlib.figure
        Figure object to plot on.
    ax : matplotlib.axes
        Axes object to plot on.
    scenarios : list
        List of scenario identifiers to plot.
    cohort : str
        Cohort identifier.
    scenarios_colors : dict, optional
        Dictionary mapping scenario names to colors.
        If None, uses the global default colors.
    plot_all : bool, optional
        Whether to plot individual fold curves.
    fill_bet : bool, optional
        Whether to fill the area between standard deviation bounds.
    title : str, optional
        Title for the plot.
    fig_type : str, optional
        Figure type identifier for saving.
    n_splits : int, optional
        Number of cross-validation splits.
    """
    if font_size == None:
        font_size = 20

    # Compute mean ROC curve and AUC

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    base_fpr = np.linspace(0, 1, 100)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    # Plot ROC curves for each fold and mean ROC curve
    if plot_all:
        individual_col = individual_color if individual_color is not None else col_line
        for i in range(n_splits):
            ax.plot(base_fpr, tprs[i], color=individual_col, alpha=individual_alpha, lw=individual_lw)

    # Plot mean line with original parameters
    ax.plot(base_fpr, mean_tprs, col_line, linestyle=line_style,
            label=f'{scenario} ({round(auc(base_fpr,mean_tprs),ndigits=2)})',
            lw=mean_lw, alpha=mean_alpha)

    if fill_bet:
        ax.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=2.5)
    if y_amap is not None:
        plot_roc_curve(test_scores=y_amap.amap, true_labels=y_amap.status, ax=ax)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate', fontsize=font_size)
    ax.set_ylabel('True Positive Rate', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size, pad=0)
    ax.set_title(title, fontsize=font_size+2, pad=10)
    condensed_font = fm.FontProperties(family='Arial', style='normal', weight='normal', stretch='condensed')
    ax.legend(loc="lower right", bbox_to_anchor=(1.01, -0.02), fontsize=font_size, frameon=False, prop=condensed_font)

    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)
    plt.rcParams.update({'font.size': font_size})

    # Export
    if save_fig & (fig_path is not None):
        save_figure(fig, title, fig_type, fig_path)


def plot_rocs_wrapper(all_tprs, fig, ax, scenarios, cohort,
                       scenarios_colors=None, plot_all=False, fill_bet=False,
                       title='', fig_type='', n_splits=5, fig_path=None, font_size=16, linewidth=1.5):
    """
    Creates multiple ROC curves on the same plot.

    Parameters:
    ----------
    mapped_tprs : DataFrame
        Multi-indexed DataFrame containing TPR values.
    fig : matplotlib.figure
        Figure object to plot on.
    ax : matplotlib.axes
        Axes object to plot on.
    scenarios : list
        List of scenario identifiers to plot.
    cohort : str
        Cohort identifier.
    scenarios_colors : dict, optional
        Dictionary mapping scenario names to colors.
        If None, uses the global default colors.
    plot_all : bool, optional
        Whether to plot individual fold curves.
    fill_bet : bool, optional
        Whether to fill the area between standard deviation bounds.
    title : str, optional
        Title for the plot.
    fig_type : str, optional
        Figure type identifier for saving.
    n_splits : int, optional
        Number of cross-validation splits.
    fig_path : str, optional
        Base path to save figures to.
    """

    colors = get_colors(scenarios, scenarios_colors)
    print(colors)
    for scenario in scenarios:
        color = colors[scenario]
        scenario_tprs = all_tprs.loc[(cohort, scenario), :]

        plot_rocs(tprs=scenario_tprs.values, fig=fig, ax=ax, plot_all=plot_all,
                  fill_bet=fill_bet, col_line=color, scenario=scenario,
                  title=title, fig_type=fig_type, n_splits=n_splits, fig_path=fig_path, font_size=font_size, save_fig=False, linewidth=linewidth)

    save_figure(fig, f"{title}_{cohort}", fig_type, fig_path)



def plot_rocs_multi_estimator(all_tprs, model_types, scenario_list, cohorts, scenarios_colors=None,scenario_lists=None,
                             n_splits=5, fig_path=None, title='', line_styles=None,
                             font_size=20, format='svg', add_color_legend=False):
    """
    Creates ROC curve visualizations comparing different estimator types.

    Parameters:
    ----------
    all_tprs : DataFrame
        Multi-indexed DataFrame containing TPR values for multiple estimators.
    model_types : list
        List of estimator types to compare.
    scenario_list : str or list
        Key for the scenario_lists dictionary or direct list of scenarios.
    cohorts : list
        List of cohorts to generate plots for.
    scenarios_colors : dict, optional
        Dictionary mapping scenario names to colors.
        If None, uses the global default colors.
    n_splits : int
        Number of cross-validation splits.
    fig_path : str
        Base path to save figures to.
    title : str
        Base title for the plots.
    fig_type : str
        Figure type identifier for saving.
    line_styles : dict, optional
        Dictionary mapping estimator names to line styles (e.g. {'XGB': '--'}).
        If None, a default dictionary is used.
    font_size : int, optional
        Font size for plot text elements. Default: 20.
    format : str, optional
        File format for saving figures. Default: 'svg'.
    add_color_legend : bool, optional
        Whether to add a separate color legend. Default: False.
    """
    # Default line styles if none are passed
    if line_styles is None:
        line_styles = {
            'XGB': '--',
            'RFC': ':',
            'CatBoost': '-',
            'NeuronMLP': (0, (3, 5, 1, 5)) # Dashdotdotted
        }

    fig_type = "AUROC_Multi_Estimator_"

    # Get scenario list if a key was provided
    if isinstance(scenario_list, str) and scenario_list in scenario_lists:
        chosen_scenarios = scenario_lists[scenario_list]
    else:
        chosen_scenarios = scenario_list

    # Get colors using the potentially provided custom dictionary
    if scenarios_colors is None:
        scenarios_colors = all_scenarios
    chosen_scenario_colors = {scenario: scenarios_colors.get(scenario) for scenario in chosen_scenarios
                             if scenario in scenarios_colors}

    # Extract available combinations from the index
    available_combos = set()
    for idx in all_tprs.index:
        if len(idx) >= 4:  # Make sure it has all levels
            combo = (idx[0], idx[1], idx[3])  # (cohort, scenario, estimator)
            available_combos.add(combo)

    for cohort in cohorts:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        ax.set_xlabel('False Positive Rate', fontsize=font_size)
        ax.set_ylabel('True Positive Rate', fontsize=font_size)
        ax.tick_params(axis='both', labelsize=font_size, pad=0)

        # Plot the diagonal line first
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=2.5)

        plot_count = 0  # Track if we've successfully plotted anything

        for scenario in chosen_scenarios:
            color = chosen_scenario_colors.get(scenario)
            if color is None:
                print(f"Warning: No color found for scenario {scenario}. Using default.")
                color = "#808080"  # Default to gray if no color found

            for estimator in model_types:
                # Check if this combination exists in our set of available combinations
                if (cohort, scenario, estimator) not in available_combos:
                    print(f"Info: No data found for cohort '{cohort}', scenario '{scenario}', estimator '{estimator}'. Skipping.")
                    continue

                try:
                    # Safely select the data using loc
                    scenario_tprs = all_tprs.loc[(cohort, scenario, slice(None), estimator), :]

                    # Skip if empty dataframe is returned
                    if scenario_tprs.empty:
                        print(f"Info: Empty data for cohort '{cohort}', scenario '{scenario}', estimator '{estimator}'. Skipping.")
                        continue

                    line_style = line_styles.get(estimator, '-')

                    # Compute metrics directly
                    tprs = scenario_tprs.values
                    mean_tprs = tprs.mean(axis=0)
                    base_fpr = np.linspace(0, 1, 100)
                    auroc_value = round(auc(base_fpr, mean_tprs), 2)

                    # Plot the line directly
                    ax.plot(base_fpr, mean_tprs, color=color, linestyle=line_style,
                            label=f'{scenario} - {estimator} ({auroc_value})',
                            lw=2.5, alpha=1.0)

                    plot_count += 1

                except Exception as e:
                    print(f"Warning: Error processing cohort '{cohort}', scenario '{scenario}', estimator '{estimator}': {e}")
                    continue

        # Only set title and legends if we actually plotted something
        if plot_count > 0:
            ax.set_title(title, fontsize=font_size+2, pad=10)

            # Set axis limits
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])

            # Main legend
            handles, labels = ax.get_legend_handles_labels()
            if handles:  # Only add legend if we have items to display
                legend = ax.legend(handles=handles, loc='lower right', frameon=False, fontsize=font_size - 6)
                ax.add_artist(legend)

            # Optional second legend for color scheme
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
                    fontsize=font_size - 6,
                    title_fontsize=font_size - 6
                )
                ax.add_artist(second_legend)

            # Save the figure
            if fig_path:
                try:
                    save_figure(fig, title=title, fig_type=fig_type, fig_path=fig_path, format=format)
                except Exception as e:
                    print(f"Error saving figure: {e}")

            plt.show()
        else:
            print(f"Warning: No valid data to plot for cohort '{cohort}'")

        plt.close(fig)  # Close the figure to free up memory


def save_figure(fig, title, fig_type, fig_path, format='svg'):
    """
    Saves a matplotlib figure as an SVG or PNG file in the specified directory with a standardized filename.

    This function ensures the target directory exists, sanitizes the file name by replacing
    spaces and slashes, and then saves the figure in (default) SVG format with transparent background.

    Args:
        fig (matplotlib.figure.Figure): The figure object to save.
        title (str): Title string used to generate the filename.
        fig_type (str): Identifier string for the type of figure (used in filename).
        fig_path (str): Directory path where the figure should be saved.
        format (str): File format to save the figure in ('svg' or 'png'). Defaults to 'svg'.

    Returns:
        None
    """
    # Create necessary directories
    os.makedirs(fig_path, exist_ok=True)

    # Replace spaces and special characters in title for filename
    file_name = title.replace(' ', '_').replace('/', '_')

    # Construct file paths for PNG and SVG
    png_path = os.path.join(fig_path, f"{fig_type}_{file_name}.png")
    svg_path = os.path.join(fig_path, f"{fig_type}_{file_name}.svg")

    if format == 'svg':
        fig.savefig(svg_path, format='svg', bbox_inches='tight', transparent=True)
        print(f"Saved figure as SVG to: {svg_path}")

    if format == 'png':
        fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
        print(f"Saved figure as PNG to: {svg_path}")






################################################################################################################################
################################################################################################################################
###########################################      Statistics         ###########################################################


def check_data_structure(all_tprs, cohorts, scenarios, estimators):
    """
    Validates the structure of the `all_tprs` DataFrame and ensures that specified cohorts,
    scenarios, and estimators are present.

    Parameters:
    ----------
    all_tprs : pd.DataFrame
        Multi-index DataFrame containing TPR values.
    cohorts : list
        List of cohorts to check in the DataFrame.
    scenarios : list
        List of scenarios to check in the DataFrame.
    estimators : list
        List of estimators to check in the DataFrame.

    Returns:
    -------
    None
        Prints data structure details and reports missing items if any.
    """

    print("Checking data structure...")
    print(f"all_tprs shape: {all_tprs.shape}")
    print(f"all_tprs index levels: {all_tprs.index.names}")
    print(f"all_tprs columns: {all_tprs.columns}")

    # Check if cohorts, scenarios, and estimators exist in all_tprs
    missing_cohorts = [cohort for cohort in cohorts if cohort not in all_tprs.index.get_level_values('cohort')]
    missing_scenarios = [scenario for scenario in scenarios if scenario not in all_tprs.index.get_level_values('scenario')]
    missing_estimators = [estimator for estimator in estimators if estimator not in all_tprs.index.get_level_values('estimator')]

    if missing_cohorts:
        print(f"Error: Missing cohorts in all_tprs: {missing_cohorts}")
    if missing_scenarios:
        print(f"Error: Missing scenarios in all_tprs: {missing_scenarios}")
    if missing_estimators:
        print(f"Error: Missing estimators in all_tprs: {missing_estimators}")

def delong_roc_variance(tpr1, tpr2):
    """
    Computes the variance required for DeLong's test using TPRs.

    Parameters:
    ----------
    tpr1 : array-like
        TPR values for the first model.
    tpr2 : array-like
        TPR values for the second model.

    Returns:
    -------
    float
        The variance for DeLong's test.
    """

    n = len(tpr1)
    v10 = np.var(tpr1)
    v11 = np.var(tpr2)

    # Estimate covariance
    cov = np.cov(tpr1, tpr2)[0, 1]

    return (v10 + v11 - 2 * cov) / n

def delong_roc_test(tpr1, tpr2):
    """
    Performs DeLong's test for comparing the AUCs of two models using TPR values.

    Parameters:
    ----------
    tpr1 : array-like
        TPR values for the first model, with rows representing folds.
    tpr2 : array-like
        TPR values for the second model, with rows representing folds.

    Returns:
    -------
    z : float
        Z-statistic of the test.
    p : float
        P-value of the test.
    mean_diff : float
        Mean AUC difference between the two models.
    se_diff : float
        Standard error of the AUC differences.
    """

    # Assuming tpr1 and tpr2 are 2D arrays where each row is a fold
    auc1 = np.mean(tpr1, axis=1)  # AUC for each fold
    auc2 = np.mean(tpr2, axis=1)  # AUC for each fold

    # Compute the differences in AUC for each fold
    auc_diffs = auc1 - auc2

    # Compute mean and standard error of the differences
    mean_diff = np.mean(auc_diffs)
    se_diff = np.std(auc_diffs, ddof=1) / np.sqrt(len(auc_diffs))

    z = mean_diff / se_diff
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p, mean_diff, se_diff

def perform_delong_test(all_tprs, cohorts, scenarios, estimators, compare_all=False, reference_scenario=None, reference_estimator=None):
    """
    Performs DeLong's test for pairwise comparison of AUCs across cohorts, scenarios, and estimators.

    Parameters:
    ----------
    all_tprs : pd.DataFrame
        Multi-index DataFrame containing TPR values.
    cohorts : list
        List of cohorts to include in the analysis.
    scenarios : list
        List of scenarios to compare.
    estimators : list
        List of estimators to analyze.
    compare_all : bool, optional
        If True, compares all scenarios pairwise. Defaults to False.
    reference_scenario : str, optional
        Reference scenario for comparisons. Required if compare_all is False.
    reference_estimator : str, optional
        Reference estimator for comparisons. Required if compare_all is False.

    Returns:
    -------
    dict
        A dictionary where each key is a cohort and the value is a DataFrame
        containing the DeLong test results with Bonferroni-adjusted p-values.

    Results DataFrame Columns:
    --------------------------
    - 'Estimator': Name of the estimator.
    - 'Model1': Name of the first scenario compared.
    - 'Model2': Name of the second scenario compared.
    - 'Z-statistic': Z-statistic from DeLong's test.
    - 'p-value': P-value from DeLong's test.
    - 'Mean AUC Difference': Mean difference in AUC between the two models.
    - 'SE of Difference': Standard error of the AUC differences.
    - 'Bonferroni-adjusted p-value': Adjusted p-value after applying Bonferroni correction.
    - 'Significant (α=0.05)': Boolean indicating statistical significance after adjustment.
    """

    check_data_structure(all_tprs, cohorts, scenarios, estimators)

    results = {}

    for cohort in cohorts:
        cohort_results = []

        if compare_all:
            # Compare all scenarios with each other
            scenario_pairs = list(combinations(scenarios, 2))
        else:
            # Compare only with the reference scenario
            if reference_scenario is None or reference_estimator is None:
                raise ValueError("reference_scenario and reference_estimator must be provided when compare_all is False")
            scenario_pairs = [(reference_scenario, scenario) for scenario in scenarios if scenario != reference_scenario]

        for scenario1, scenario2 in scenario_pairs:
            for estimator in estimators:
                try:
                    if compare_all:
                        # Both scenarios use the same estimator
                        tpr1 = all_tprs.loc[(cohort, scenario1, slice(None), estimator), :].values
                        tpr2 = all_tprs.loc[(cohort, scenario2, slice(None), estimator), :].values
                        estimator_label = estimator
                    else:
                        # Reference scenario uses reference_estimator, comparison uses current estimator
                        tpr1 = all_tprs.loc[(cohort, scenario1, slice(None), reference_estimator), :].values
                        tpr2 = all_tprs.loc[(cohort, scenario2, slice(None), estimator), :].values
                        estimator_label = f"{reference_estimator} vs {estimator}"

                    # Perform DeLong's test
                    z, p_value, mean_diff, se_diff = delong_roc_test(tpr1, tpr2)

                    cohort_results.append({
                        'Estimator': estimator_label,
                        'Model1': f"{scenario1}",
                        'Model2': f"{scenario2}",
                        'Z-statistic': np.round(z, 4),
                        'p-value': p_value,
                        'Mean AUC Difference': round(mean_diff, 4),
                        'SE of Difference': round(se_diff, 4)
                    })

                except KeyError as e:
                    print(f"Skipping comparison for cohort {cohort}, {scenario1} vs {scenario2}, estimator {estimator}: data not found")
                    continue

        # Create DataFrame for the cohort
        results[cohort] = pd.DataFrame(cohort_results)

        # Apply Bonferroni correction
        n_tests = len(results[cohort])
        if n_tests > 0:
            results[cohort]['Bonferroni-adjusted p-value'] = np.minimum(results[cohort]['p-value'] * n_tests, 1.0)

            # Determine significance after Bonferroni correction
            results[cohort]['Significant (α=0.05)'] = results[cohort]['Bonferroni-adjusted p-value'].apply(
                lambda x: "True" if x < 0.05 else "False"
            )

    return results


def perform_delong_test_custom(all_tprs, cohorts, ref_combo, comparison_combos):
    """
    Compare a reference (scenario, estimator) combination against specific other combinations.

    Parameters:
    -----------
    all_tprs : DataFrame
        Multi-indexed DataFrame with TPR values
    cohorts : list
        List of cohorts to analyze
    ref_combo : tuple
        Reference (scenario, estimator) combination
    comparison_combos : list of tuples
        List of (scenario, estimator) combinations to compare against reference

    Returns:
    --------
    dict : Results dictionary similar to perform_delong_test
    """
    results = {}

    for cohort in cohorts:
        cohort_results = []
        ref_scenario, ref_estimator = ref_combo

        for comp_scenario, comp_estimator in comparison_combos:
            try:
                tpr1 = all_tprs.loc[(cohort, ref_scenario, slice(None), ref_estimator), :].values
                tpr2 = all_tprs.loc[(cohort, comp_scenario, slice(None), comp_estimator), :].values

                z, p_value, mean_diff, se_diff = delong_roc_test(tpr1, tpr2)

                cohort_results.append({
                    'Reference': f"{ref_scenario}-{ref_estimator}",
                    'Comparison': f"{comp_scenario}-{comp_estimator}",
                    'Z-statistic': np.round(z, 4),
                    'p-value': p_value,
                    'Mean AUC Difference': round(mean_diff, 4),
                    'SE of Difference': round(se_diff, 4)
                })
            except KeyError as e:
                print(f"Skipping {comp_scenario}-{comp_estimator} for {cohort}: {e}")
                continue

        # Create DataFrame and apply Bonferroni correction
        results[cohort] = pd.DataFrame(cohort_results)
        if len(results[cohort]) > 0:
            n_tests = len(results[cohort])
            results[cohort]['Bonferroni-adjusted p-value'] = np.minimum(results[cohort]['p-value'] * n_tests, 1.0)
            results[cohort]['Significant (α=0.05)'] = results[cohort]['Bonferroni-adjusted p-value'].apply(
                lambda x: "True" if x < 0.05 else "False"
            )
    print("✅ DeLong tests completed successfully!")
    return results


# Helper function to save results to Excel for a dedicated dataframe
def save_results_to_excel(results, file_name):
    with pd.ExcelWriter(file_name) as writer:
        for cohort, df in results.items():
            df.to_excel(writer, sheet_name=cohort, index=False)