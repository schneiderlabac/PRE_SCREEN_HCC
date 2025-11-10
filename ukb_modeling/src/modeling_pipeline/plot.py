################# Content #############
# 1. `export_visual`: Helper function to streamline the export of svgs, pngs for defined subsets to defined paths.
# 2. `adjust_alpha`: Function to adjust the alpha value of a given RGB color.
# 3. `feature_imp_barplot`: Function to provide an overview visualization of the individual feature importances.
# 4. `plot_roc_curve`: Function to plot the ROC curve for given test scores and true labels.
# 5. `plot_rocs`: Function to plot composed ROC curves for multiple true positive rates and AUCs.
# 6. `save_colorbar`: Helper function to export a colorbar as a separate svg.
# 7. `wrapper_eval_prediction_mono`: Wrapper function for 2-class conf. matrices (1 threshold -> High/Low) to evaluate and plot predictions for a single model.
# 8. `wrapper_eval_prediction_multi`: Wrapper function for 3 (or more)-class conf. matrices (2+ thresholds -> e.g. low/middle/high)to evaluate and plot predictions for multiple models.
# 9. `create_violin_plot`: Function to create a violin plot for predicted probabilities.
# 10. `plot_auc_time`: Function to plot time-dependent AUROC and AUPRC.
# 11. `plot_ablation_dual_visual`: Function to create a dual-axis plot for AUROC and AUPRC against the number of included features.
# 12. `plot_precision_recall_curves`: Function to plot overlaying precision-recall curves for multiple datasets.
# 13. `shap_analysis`: Function to generate a SHAP summary plot for the best-performing estimator.
# 14. plot_Kaplan_Meier: Function to plot Kaplan-Meier curves for risk of target event occuring over time for respective risk groups
# 15. Export interpolated precision-recall curves for external validation using the test_train_pred DataFrame



# Plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from scipy.stats import sem, t
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, PrecisionRecallDisplay
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
import matplotlib.font_manager as fm
import shap
from lifelines import KaplanMeierFitter
import math
from typing import Literal,List, Dict, Optional, Tuple, Union
import joblib

####################################################################################################
# Parameters for the plotting:
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["figure.titlesize"] = 14


####################################################################################################


def export_visual(fig, type, row_subset, col_subset, modeltype, fig_path, threshold=None, format="svg", figsize=None):
    """
    Export the given figure to the specified path.  -> framework for plt.fig.savefig()
    construct a framework for the fig_save() function from matplotlib

    Parameters:
    - fig: The figure object to be saved.
    - type: The type of plot to be stored
    - row_subset: The subdirectory under the base path.
    - col_subset: Used to name the file.
    - modeltype: Used to name the file.
    - format: Desired file format, either "svg" or "png". Defaults to "svg".
    - figsize: Tuple (width, height) to specify figure size in inches. Used only if format is "png".
    - base_path: The main directory where the visuals are saved. Defaults to the given path.

    Example:
    export_visual(fig, row_subset, col_subset, modeltype, format="png", figsize=(10, 10))
    """

    # Create necessary directories
    sub_directory = os.path.join(fig_path, row_subset)
    os.makedirs(sub_directory, exist_ok=True)

    # Construct the file name
    if threshold:
        file_name = f"{type}_{col_subset}_{modeltype}_{threshold}.{format}"
    else:
        file_name = f"{type}_{col_subset}_{modeltype}.{format}"

    # Construct the full path for the file
    full_path = os.path.join(sub_directory, file_name)

    # Set figure size if format is png and figsize is provided
    if format.startswith("."):
        format = format.lstrip(".")

    if format == "png" and figsize:
        fig.set_size_inches(*figsize)

    # Save the figure
    fig.savefig(full_path, format=format, dpi=600 if format == "png" else None)  # Set dpi for better resolution for png


def adjust_alpha(rgb, new_sat=0.5):
    return mcolors.hsv_to_rgb(mcolors.rgb_to_hsv(rgb) * [1, new_sat, 1])

from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np

def get_catboost_feature_importances(cat_model, X, method='FeatureImportance', prettified=True):
    """
    Wrapper to get CatBoost feature importances in a format compatible with feature_imp_barplot.

    Args:
        cat_model (CatBoostClassifier): trained CatBoost model
        X (pd.DataFrame): features
        method (str): feature importance type, e.g., 'FeatureImportance', 'ShapValues'
        prettified (bool): return with feature names and sorted

    Returns:
        pd.DataFrame: DataFrame with columns ['feature', 'mean_feature_imp']
    """
    pool = Pool(X)
    importances = cat_model.get_feature_importance(data=pool, type=method, prettified=False)

    df_feature_imp = pd.DataFrame({
        "feature": X.columns,
        "mean_feature_imp": importances
    }).sort_values(by="mean_feature_imp", ascending=False)

    return df_feature_imp

def feature_imp_barplot(
    model,
    X_ohe_map,
    self,
    n_features=60,
    xlabel="Feature importance",
    ylabel=None,
    small_plot=True,
    fontsize=16,
    fontsize_smallplot=3,
    pos_smallplot="lower right",
    size_smallplot="30%",
    bar_height_factor=17,
    fig_width=7,
    func_for_aggregation=np.mean,
    features_to_exclude=[],
    short_names=True,
    export=True,
    color_dict=None,
    borderpad=2,
    linewidth=1.5,
    df_source_map_rename={
        "df_covariates": "Demography\n& Lifestyle",
        "df_diagnosis": "EHR",
        "df_blood": "Blood count\n& Serum",
        "df_snp": "Genomics",
        "df_metabolomics": "Metabolomics",
        "df_metadata": "Metadata",
    },
):
    """
    Args:
        model (model): Model to evaluate and draw the feature importance from.
        X_ohe_map (df): df for the column mapping with the feature source and name.
        n_features (int or 'all'): Number of features to plot in the main plot.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        small_plot (bool): Include small aggregated plot.
        fontsize_smallplot (int): Font size for small plot.
        pos_smallplot (str): Location of inset plot.
        size_smallplot (str): Size of inset plot as %.
        func_for_aggregation (func): Aggregation function (e.g., np.sum or np.mean).
        features_to_exclude (list): List of features to exclude from plotting.
        export (bool): Save plot as SVG.
        color_dict (dict): Optional color dictionary.
        df_source_map_rename (dict): Mapping of source names to plot-friendly labels.

    Returns:
        fig, ax, plot_X: Matplotlib figure and axis, and processed DataFrame.
    """
    try:
        feature_imp_all = model.feature_importances_()
        feature_imp = feature_imp_all["mean_feature_imp"]
        is_master_model = True

        # Calculate std_feature_imp directly from the individual model columns
        model_cols = [col for col in feature_imp_all.columns if col.startswith('model_')]
        model_values = feature_imp_all[model_cols]
        std_feature_imp = model_values.std(axis=1)

    except Exception as e:
        print(f"Got this Exception {e}")
        print("For the feature_imp. plot only a single model was used.".center(100, "-"))
        is_master_model = False
        std_feature_imp = None
        feature_imp = model.feature_importances_

        # Normalize for fallback single model
        norm_factor = feature_imp.sum()
        if norm_factor > 0:
            feature_imp = feature_imp / norm_factor

        is_master_model = False

    ## Consistency check
    col_subset = self.user_input.col_subset
    row_subset = self.user_input.row_subset
    fig_path = self.user_input.fig_path

    if feature_imp.shape[0] != X_ohe_map.shape[0]:
        print("feature_imp and mapping have not the same number of columns represented\n")

    # Create color scheme
    if color_dict is None:
        color_dict = dict(
            zip(X_ohe_map.source.unique(), sns.color_palette("pastel", n_colors=len(X_ohe_map.source.unique())))
        )
    X_ohe_map["color"] = X_ohe_map.source.map(color_dict)


    # Use short names if available
    # Add a smart name_print column if short_names=True, that takes short_name where available, else name
    if short_names:
        try:
            from pp import load_columngroups
            colgroups = load_columngroups(self.user_input)
            short_map = colgroups.set_index("column_name")["short_name"]
            X_ohe_map["short_name"] = X_ohe_map["name_print"].map(short_map)
        except Exception as e:
            print(f"Short name mapping failed: {e}")

    # Assign name_print to use short_name if present, else fallback to name
    X_ohe_map["name_print"] = X_ohe_map["short_name"].fillna(X_ohe_map["name_print"]) if "short_name" in X_ohe_map.columns else X_ohe_map["name_print"]


    # Prepare feature importance values for plotting
    plot_X = X_ohe_map.copy()

    if is_master_model:
        plot_X[feature_imp_all.columns] = np.float64(feature_imp_all.values)
        plot_X["feature_imp"] = plot_X["mean_feature_imp"]

        # Add std_feature_imp if we calculated it
        if std_feature_imp is not None:
            plot_X["std_feature_imp"] = std_feature_imp.values
    else:
        plot_X["feature_imp"] = np.float64(feature_imp)

    plot_X.sort_values(by=["feature_imp"], ascending=False, inplace=True)
    plot_X.drop(labels=features_to_exclude, axis=0, inplace=True)

    groups_with_features = plot_X["source"].unique()

    if n_features == "all":
        top_features = plot_X
    else:
        top_features = plot_X.iloc[:n_features, :]

    top_features = top_features.copy()
    top_features = top_features[["name_print", "source", "color", "feature_imp", "std_feature_imp"]]


    #limit n_feature to the number of features still in the plot
    actual_n_features = len(top_features)
    if n_features > actual_n_features:
        n_features = actual_n_features

    # if "feature_imp" not in plot_X.columns:
    #     plot_X.rename(columns={"mean_feature_imp": "feature_imp"}, inplace=True)
    plot_X["source_lit"] = plot_X.source.map(df_source_map_rename)
    bar_height = bar_height_factor * (n_features / 60)  # Define height of the plot according to number of features
    # Horizontal bar plot with feature importance for the top individual feature
    fig, ax = plt.subplots(figsize=(fig_width, bar_height))  # previously 20

    plt.subplots_adjust(top=0.98, bottom=0.05, left=0.2, right=0.98)


    max_imp = top_features["feature_imp"].max()

    # Add a buffer of 20% to maximum value
    x_max = max_imp * 1.1

    # Target for approximately 6-8 ticks on the x-axis
    target_ticks = 7

    # Calculate a reasonable tick interval to get ~target_ticks ticks
    raw_interval = x_max / target_ticks

    # Round to a nice number (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, etc.)
    if raw_interval < 0.01:
        tick_interval = 0.005
    elif raw_interval < 0.02:
        tick_interval = 0.01
    elif raw_interval < 0.05:
        tick_interval = 0.02
    elif raw_interval < 0.1:
        tick_interval = 0.05
    elif raw_interval < 0.2:
        tick_interval = 0.1
    elif raw_interval < 0.5:
        tick_interval = 0.2
    else:
        tick_interval = 0.5

    # Round up x_max to the nearest tick
    x_max = math.ceil(x_max / tick_interval) * tick_interval

    # Calculate how many ticks this will produce
    num_ticks = int(x_max / tick_interval) + 1

    # Adjust interval if we'd get too many ticks
    if num_ticks > 7:
        # Go to the next larger interval option
        if tick_interval == 0.005:
            tick_interval = 0.01
        elif tick_interval == 0.01:
            tick_interval = 0.02
        elif tick_interval == 0.02:
            tick_interval = 0.05
        elif tick_interval == 0.05:
            tick_interval = 0.1
        elif tick_interval == 0.1:
            tick_interval = 0.2
        elif tick_interval == 0.2:
            tick_interval = 0.5
        else:
            tick_interval = 1.0

        # Recalculate x_max with the new interval
        x_max = math.ceil(x_max / tick_interval) * tick_interval

    # Set the axis ticks and limits
    plt.xticks(np.arange(0, x_max + tick_interval/2, step=tick_interval))
    plt.xlim(0, x_max)

    if is_master_model and "std_feature_imp" in plot_X.columns:
        print("Plotting feature imp of master model with STD of 5 models")
        sns.barplot(
            data=top_features,
            x="feature_imp",
            estimator=np.mean,
            y="name_print",
            hue="source",
            errorbar=None, #Added manually
            saturation=1,
            dodge=False,
            palette=color_dict,
            ax=ax,
        )

        for i, (idx, row) in enumerate(top_features.iterrows()):
                    imp = row["feature_imp"]
                    std = row["std_feature_imp"]
                    ax.errorbar(
                        x=imp,
                        y=i,  # Position on the y-axis
                        xerr=std,
                        fmt="none",
                        ecolor="lightgray",
                        elinewidth=5,
                        capsize=0
                    )
    else:
        sns.barplot(
            data=top_features,
            x="feature_imp",
            y="name_print",
            hue="source",
            estimator=np.mean,
            errorbar=None,
            dodge=False,
            saturation=1,
            palette=color_dict,
            ax=ax,
        )

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.margins(x=0.02, y=0.015)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)

    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)

    if small_plot:
        ax.legend().set_visible(False)
        # add a second plot to the axis with mean/sum of feature group
        axadded = inset_axes(ax, width=size_smallplot, height=size_smallplot, loc=pos_smallplot, borderpad=borderpad)
        # Create the source order ensuring only sources in our data are used
        source_order = [df_source_map_rename[src] for src in groups_with_features if src in df_source_map_rename]
        plot_X_small = plot_X[plot_X['source_lit'].isin(source_order)].copy()
        #color_dict = {k.replace("\\n", "\n"): v for k, v in color_dict.items()}
        sns.barplot(
            data=plot_X_small,
            x="feature_imp",
            y="source_lit",
            hue="source",
            order=source_order,
            dodge=False,
            estimator=func_for_aggregation,
            palette=color_dict,
            ax=axadded,
            err_kws={'color': 'lightgray'},
        )

        axadded.set_ylabel("")
        axadded.set_xlabel("")
        axadded.set_yticklabels(axadded.get_yticklabels(), fontsize=fontsize-2)
        axadded.tick_params(axis='x', pad=-5)

        #Reduce the space between the ticks and the labels
        #axadded.set_xticklabels(axadded.get_xticklabels(), fontsize=fontsize-2, rotation=0, pad=0)

        # Instead of bar_label which puts numbers on bars, format differently
        #axadded.tick_params(reset=True, color="lightgray")
        for spine in axadded.spines.values():
            spine.set_linewidth(linewidth)
        axadded.legend().set_visible(False)
        axadded.set_title(
            f"{func_for_aggregation.__name__}".capitalize(), fontsize=plt.rcParams["legend.fontsize"]
        )

    if export:
        svg_path = os.path.join(fig_path, f"Feature Imp_{col_subset}_{row_subset}_{n_features}.svg")

        #Append the pl_suffix description if one is available
        suffix = getattr(self.user_input, "pl_suffix", None)
        if suffix:
            svg_path = f"{svg_path.rstrip('.svg')}_{suffix}.svg"
        os.makedirs(fig_path, exist_ok=True)
        fig.savefig(svg_path, format="svg", bbox_inches="tight", transparent=True)
        print(f"Feature importance plot saved to {svg_path}")

    return fig, ax, plot_X



def plot_roc_curve(test_scores, true_labels, ax=plt.axes, title=""):
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(true_labels, test_scores)
    roc_auc = auc(fpr, tpr)
    fpr_base = np.linspace(0, 1, 100)
    tpr = np.interp(fpr_base, fpr, tpr)
    # Create the ROC curve plot
    if ax == False:
        plt.plot(fpr_base, tpr, color="darkorange", lw=2, label="ROC curve (AUC = {:.2f})".format(roc_auc))
    else:
        ax.plot(fpr_base, tpr, color="darkorange", lw=2, label="ROC curve (AUC = {:.2f})".format(roc_auc))
        ax.set_title(title)
        ax.legend()
    return thresholds, fpr, tpr


def plot_rocs(tprs, aucs, n_splits, plot_all=True, y_amap=None, ax=plt.axes, fontsize=14, border_width=1):
    """Plot a composed ROC curve for the tprs and the corresponding mean with the AUROCcurves
    This function was written for a majority voting model with 5 individual models, to evaluate the performance on the individual testing dataset.

    Args:
        tprs (array or DataFrame): true positive rates
        plot_all (bool, optional): if True all rocs are plotted in gray and only the mean ROCcurve ist plotted. Defaults to True.
        y_amap (dataframe): if you want to include an other test (state of the art laboratory testing ... has to have the columns 'amap' and 'status'). Defaults to None.
    """
    # Compute mean ROC curve and AUC
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    base_fpr = np.linspace(0, 1, 100)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    # Plot ROC curves for each fold and mean ROC curve
    if plot_all == True:
        for i in range(n_splits):
            ax.plot(base_fpr, tprs[i], "b", alpha=0.15)
    ax.plot(base_fpr, mean_tprs, "b", label="Mean ROC curve (AUC = %0.2f)" % (np.mean(aucs)), linewidth=2)
    ax.fill_between(base_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.3)

    ax.plot([0, 1], [0, 1], "r--")
    if y_amap is not None:
        plot_roc_curve(test_scores=y_amap.amap, true_labels=y_amap.status)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("False Positive Rate", fontsize=fontsize*1.2)
    ax.set_ylabel("True Positive Rate", fontsize=fontsize*1.22)
    plt.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.legend(loc="lower right", fontsize=fontsize, frameon=False)

        # Add bounding box with customizable width
    for spine in ax.spines.values():
        spine.set_linewidth(border_width)
        spine.set_edgecolor('black')


    plt.tight_layout()


def save_colorbar(pip_self, orientation="vertical", figsize=(1, 6), cmap=plt.cm.Blues, font_size=20):
    """Helperfunction to export a colorbar as a separate svg once you run the conf matrices"""
    filename = os.path.join(pip_self.user_input.fig_path, "colorbar.svg")
    cmap = cmap  # Choose your colormap here
    fig, ax = plt.subplots(figsize=figsize)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cb1 = ColorbarBase(ax, cmap=cmap, norm=norm, orientation=orientation)
    # cb1.set_label('Score')
    tick_locator = plt.MaxNLocator(nbins=5)
    cb1.locator = tick_locator
    cb1.update_ticks()
    cb1.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x * 100)}%"))
    cb1.ax.tick_params(labelsize=font_size)

    fig.savefig(filename, format="svg", bbox_inches="tight")
    plt.close(fig)


def wrapper_eval_prediction_mono(
    pip_self,
    X,
    y_true,
    model,
    y_pred_proba=None,
    n_rows=2,
    n_cols=3,
    figsize=(15, 15),
    export=True,
    thresholds=[0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
    font_size=20,
    stratify=None
):
    ohe = pip_self.ohe
    DOI = pip_self.user_input.DOI
    col_subset = pip_self.user_input.col_subset
    row_subset = pip_self.user_input.row_subset
    fig_path = pip_self.user_input.fig_path

    if stratify is not None:
        if not isinstance(stratify, dict):
            raise ValueError("stratify should be a dictionary with 'column' and 'value' keys")

        column = stratify.get('column')
        value = stratify.get('value')

        if column not in X.columns:
            raise ValueError(f"Column '{column}' not found in X")

        mask = X[column] == value
        X = X[mask]
        y_true = y_true[mask]

        # Update the subset information for the plot title
        row_subset += f" ({column}={value})"


    def eval_prediction(
        y_true,
        threshold,
        ohe=ohe,
        y_pred_proba=None,
        y_pred=None,
        X=None,
        target_names=[f"No {DOI}", DOI],
        model=model,
        annotation_sz=15,
        axis=None,
        font_size=font_size,
    ):
        """Evaluate the prediction of a model passed or a vector of y_pred. or the model itself by model.predict against the y_true vector

        Args:
            X (featurematrix): ohe encoded feature matrix (test or train) to evaluate
            y_true (list/Series): True label of the cases
            model (model, optional): passed model can be used für the y_prediction if no y_pred is passed into the function. Defaults to model.
            y_pred (list/Series), optional): y_pred list or Series, f.e. if you want to evaluate different thresholds for the pred. prob. Defaults to None.
            target_names (list), optional): list of the y expressions.
            passed axis to plot plot onto: to plot multiple matrices in one plot

        Returns:
            series: containing ['TN', 'TP', 'FN', 'FP', 'sensitivity', 'specificity', 'pos_pred_val', 'neg_pred_val']
            matrix: confusion matrix output as heatmap

        hint: not ready yet for multithreshold evaluation -> use 6Maritx function instead!
        """
        results_df = pd.DataFrame()

        if y_pred is None and type(threshold) is float:
            try:
                y_pred = pd.DataFrame(model.predict_proba(X))[1] >= threshold
            except:
                y_pred = ((model.predict_proba(X)) >= threshold).astype(int)
        elif y_pred is None and type(threshold) is not float:
            y_pred = []
            if pd.DataFrame(model.predict_proba(X))[1] < threshold[0]:
                y_pred.append("low")
            elif (
                pd.DataFrame(model.predict_proba(X))[1] >= threshold[0]
                and pd.DataFrame(model.predict_proba(X))[1] <= threshold[1]
            ):
                y_pred.append("medium")
            elif pd.DataFrame(model.predict_proba(X))[1] > threshold[1]:
                y_pred.append("high")
        elif y_pred is None and y_pred_proba is not None:
            y_pred = int(y_pred_proba >= threshold)

        # print(classification_report(y_true, y_pred, target_names=target_names))
        TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()  # type: ignore
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        pos_pred_val = TP / (TP + FP)
        neg_pred_val = TN / (TN + FN)

        # ploting and export
        # Create Plot
        if axis == None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            ax = axis
            fig = None
        plt.rcParams.update({"font.size": annotation_sz})  # Set a default font size for all elements
        # Normalize the confusion matrix if normalize_rows is True
        cm = confusion_matrix(y_true, y_pred)  # type: ignore
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        annotations = np.array(
            [
                ["{}\n({:.2f}%)".format(cm[i, j], cm_normalized[i, j] * 100) for j in range(cm.shape[1])]
                for i in range(cm.shape[0])
            ]
        )
        sns.heatmap(
            cm_normalized, annot=annotations, fmt="", ax=ax, cmap="Blues", cbar=False, annot_kws={"size": font_size}
        )
        ax.set_xlabel("")
        ax.set_ylabel("True label")
        ax.set_xticklabels(target_names)
        ax.set_yticklabels(target_names, rotation=0)

        # Define text
        textstr = f"Sensitivity: {round(sensitivity, 2)}\nSpecificity: {round(specificity, 2)}"
        textstr2 = f"PPV: {round(pos_pred_val, 4)}"
        textstr2 += f"\nNPV: {round(neg_pred_val, 4)}"

        # Place text boxes
        props = dict(boxstyle="square", facecolor="none", alpha=0.5, edgecolor="none")  # setting properties
        ax.text(
            0.2,
            -0.15,
            textstr,
            transform=ax.transAxes,
            fontsize=annotation_sz*1.1,
            verticalalignment="top",
            horizontalalignment="center",
            bbox=props,
        )
        ax.text(
            0.8,
            -0.15,
            textstr2,
            transform=ax.transAxes,
            fontsize=annotation_sz*1.1,
            verticalalignment="top",
            horizontalalignment="center",
            bbox=props,
        )
        ax.set_xlabel("")
        output_series = pd.Series(
            [TN, TP, FN, FP, sensitivity, specificity, pos_pred_val, neg_pred_val],
            index=["TN", "TP", "FN", "FP", "sensitivity", "specificity", "pos_pred_val", "neg_pred_val"],
        )
        plt.tight_layout()
        return output_series, fig

    # Dictionary to store metrics for each threshold
    metrics_dict = {}

    # Create a 3x3 grid of subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)  # Adjust the figsize as needed
    axes = axes.ravel()  # Flatten the 3x3 grid into a 1D array for easier indexing

    # Iterate over the thresholds
    for index, threshold in zip(np.arange(0, 9), thresholds):
        print(f"Evaluating for threshold: {threshold}")
        # Evaluate prediction
        metric, Matrix = eval_prediction(
            model=model,
            X=pip_self.ohe.transform(X),
            threshold=threshold,
            y_true=y_true,
            axis=axes[index],
            y_pred_proba=y_pred_proba,
        )
        metrics_dict[threshold] = metric  # Store the metric in the dictionary
        axis = axes[index].set_title(f"Threshold: {threshold}")  # type: ignore

    plt.tight_layout()
    fig.text(0.5, -0.05, f"Conf. Matrices: {col_subset} {row_subset}", ha="center", fontsize=24)
    if export:

        svg_path = os.path.join(fig_path, f"4xConfMatrix_{col_subset}_{row_subset}.svg")
        suffix = getattr(pip_self.user_input, "pl_suffix", None)
        if suffix:
            svg_path = f"{svg_path.rstrip('.svg')}_{suffix}.svg"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")  # Save the entire 3x3 grid figure
    # return fig  # , metrics_dict





def wrapper_eval_prediction_multi(
    pip_self,
    X,
    y_true,
    model,
    export=True,
    n_rows=3,
    n_cols=2,
    figsize=(15, 17),
    thresholds=[(0.7, 0.8), (0.6, 0.7), (0.5, 0.7), (0.4, 0.6), (0.3, 0.4)],
    incorp_threh_in_y_label=False,
    font_size=20,
    target=None,
    stratify=None
):
    DOI = pip_self.user_input.DOI
    ohe = pip_self.ohe
    col_subset = pip_self.user_input.col_subset
    row_subset = pip_self.user_input.row_subset
    fig_path = pip_self.user_input.fig_path
    estimator = pip_self.name.split("_")[-1]

    if target is None:
        target = pip_self.user_input.target_to_validate_on # Assign the default target if target == None

    # This block subsets the data for a column and specific value (e.g. SEX == 1, representing only males in the dataset, for nuanced analysis)
    if stratify is not None:
        if not isinstance(stratify, dict):
            raise ValueError("stratify should be a dictionary with 'column' and 'value' keys")

        column = stratify.get('column')
        value = stratify.get('value')

        if column not in X.columns:
            raise ValueError(f"Column '{column}' not found in X")

        mask = X[column] == value
        X = X[mask]
        y_true = y_true[mask]

        # Update the subset information for the plot title
        row_subset += f" ({column}={value})"

    def matrix_6(
        thresholds,
        y_true,
        y_predicted,
        ax=None,
        map_true_label={0: f"No {DOI}", 1: DOI},
        incorp_threh_in_y_label=incorp_threh_in_y_label,
        font_size=font_size,
    ):
        """This Function can be used to generate a multidimentional version of the 2x2 table to evaluate a threshold(s).

        Args:
            threshods (Series or List): series or list of the thresholds to evaluate, first value corresponding to the lower threshold and the second one to the upper one
            y_true (List): true label, plotted on the y axis
            y_predicted (list): predicted label given by the model
            map_true_label (dict): translation of 0,1 of the ytrue vector to strings -> as well a as order of the yaxis in the plot
        Ret:
            fig and axis of sns.heatmap with the two thresholds to descriminate
        """
        if len(y_true) != len(y_predicted):
            raise ValueError("y_true and y_pred do not have the same length")
        df = pd.DataFrame()
        # mapping the values to the thresholds and defining the order that they should be displayed in the heatmap
        df["y_true"] = y_true[target]
        df["y_predicted"] = y_predicted

        prefix = ""
        risk_status = []

        for item in df.y_predicted:
            if item < thresholds[0]:
                risk_status.append(f"Low Risk\n[<{thresholds[0]}]")
            elif item >= thresholds[0] and item <= thresholds[1]:
                risk_status.append(f"Medium Risk\n[≥{thresholds[0]} & ≤ {thresholds[1]}]")
            elif item > thresholds[1]:
                risk_status.append(f"High Risk\n[>{thresholds[1]}]")
            else:
                print("item could not be mapped! please read the code:)")

        df["risk_status"] = risk_status
        df_dumm = pd.get_dummies(df.risk_status, prefix=prefix, prefix_sep="")
        df_out = df[df_dumm.columns] = df_dumm
        df["y_true"] = df.y_true.map(map_true_label)
        df.reset_index(inplace=True)
        # grouping by true label
        df_out = df.loc[:, :].groupby("y_true").sum(numeric_only=True)[df_dumm.columns]
        df_normalized = df_out.div(df_out.sum(axis=1), axis=0)
        df_out = df_out.astype(int)

        order = [
            df_out.columns[df_out.columns.str.startswith(prefix + "Low Risk")][0],
            df_out.columns[df_out.columns.str.startswith(prefix + "Medium Risk")][0],
            df_out.columns[df_out.columns.str.startswith(prefix + "High Risk")][0],
        ]
        try:
            df_out = df_out.loc[:, order]
        except:
            print("order did not work")

        row_order = [f"No {DOI}", DOI]
        # order dfs by column and row
        df_out = df_out[order].reindex(row_order)
        df_normalized = (df_normalized[order].reindex(row_order)) * 100

        # Configure custom annotations with absolute + relative values

        # Calculate PPV for Medium Risk and High Risk
        ppv_medium = df_out.iloc[1, 1] / (df_out.iloc[0, 1] + df_out.iloc[1, 1])
        ppv_high = df_out.iloc[1, 2] / (df_out.iloc[0, 2] + df_out.iloc[1, 2])
        annotations = df_out.astype(str) + "\n\n(" + np.round(df_normalized, 1).astype(str) + "%)"

        # create the figure object for the heatmap or use a given one
        if ax is False or ax is None:
            fig, ax = plt.subplots()
        # df_out=df_out.loc[list(map_true_label.values()),:]

        sns.heatmap(
            data=df_out.div(df_out.sum(axis=1), axis=0),
            annot=annotations,
            ax=ax,
            cbar=False,
            cmap="Blues",
            fmt="",
            annot_kws={"size": font_size},
        )
        ax.set_ylabel("True label")
        if not incorp_threh_in_y_label:
            ax.set_xticklabels([i.split("\n")[0] for i in df_out.columns])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        label_fontsize = plt.rcParams["xtick.labelsize"]  # get current xtick labelsize for transfer on text below
        # Add PPV as values below label
        ax.text(0.165, -0.23, "PPV:", ha="center", fontsize=label_fontsize, transform=ax.transAxes)
        ax.text(0.5, -0.23, f"{ppv_medium:.2%}", ha="center", fontsize=label_fontsize, transform=ax.transAxes)
        ax.text(0.825, -0.23, f"{ppv_high:.1%}", ha="center", fontsize=label_fontsize, transform=ax.transAxes)
        return ax, df_out

    # Create a 3x3 grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)  # Adjust the figsize as needed
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    axes = axes.ravel()  # Flatten the 3x3 grid into a 1D array for easier indexing

    # Iterate over the thresholds
    for index, threshold in zip(np.arange(0, len(thresholds)), thresholds):
        print(f"Evaluating for threshold: {threshold}")
        # Evaluate prediction
        try:
            matrix_6(
                y_true=y_true,
                y_predicted=[item[1] for item in model.predict_proba(ohe.transform(X))],
                thresholds=threshold,
                ax=axes[index],
            )
        except:
            matrix_6(
                y_true=y_true,
                y_predicted=model.predict_proba(ohe.transform(X)).tolist(),
                thresholds=threshold,
                ax=axes[index],
            )
    plt.tight_layout()
    fig.text(0.5, -0.05, f"Conf. Matrices: {col_subset} {row_subset}", ha="center", fontsize=24)

    if export:
        svg_path = os.path.join(fig_path, f"6xConfMatrix_{col_subset}_{row_subset}_{estimator}.svg")

        suffix = getattr(pip_self.user_input, "pl_suffix", None)
        if suffix:
            svg_path = f"{svg_path.rstrip('.svg')}_{suffix}.svg"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        print("SVG saved to: " ,svg_path)

#####Plotting actual prediction probabilities with violin plot #####

def create_violin_plot(
    pip_self,
    data,
    model,
    ohe,
    thresholds_choice,
    gap,
    width,
    show_thresholds=True,
    ylim=(0, 1),
    palette=None,
    ax=None,
    split=True,
    y_pred=None,
    truth="status",
    save_fig=True,
):
    DOI = pip_self.user_input.DOI
    row_subset_long = pip_self.user_input.row_subset_long
    col_subset = pip_self.user_input.col_subset
    row_subset = pip_self.user_input.row_subset
    try:
        estimator = pip_self.model_type
    except:
        estimator = "undefined model type"
    fig_path = pip_self.user_input.fig_path
    title = f"{col_subset} {row_subset} {estimator}"

    if palette is None:
        palette = {
            0: adjust_alpha(pip_self.mapper.color_groups_violin[pip_self.user_input.col_subset], 0.5),
            1: adjust_alpha(pip_self.mapper.color_groups_violin[pip_self.user_input.col_subset], 1),
        }
    # if y_pred is None:
    #     pred_probs = model.predict_proba(ohe.transform(pip_self.data.X_val))
    # else:
    #     pred_probs = y_pred

    # df = pd.DataFrame()
    # df["status"] = pip_self.data.y_val[truth]
    # if type(pred_probs) != np.ndarray:
    #     df["proba"] = pred_probs.values
    # else:
    #     df["proba"] = pd.DataFrame(pred_probs)[1].values
    if ax == None:
        fig, ax = plt.subplots(figsize=(3, 10))
        create_fig = True
    else:
        fig = ax.get_figure()
        create_fig = False

    sns.violinplot(
        data=data,
        y="y_pred",
        x=truth,
        split=split,
        inner="quart",
        gap=gap,
        width=width,
        dodge='auto',
        palette=palette,
        hue=truth,
        linecolor="white",
        linewidth=2,
        saturation=1,
        ax=ax,
    )
    ax.legend().set_visible(False)
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.set_ylabel("")
    ax.set_title(col_subset, fontsize=22)
    ax.tick_params(axis="y", labelsize=24)
    ax.text(0.1, -0.05, "Controls", color="black", ha="center", va="bottom", fontsize=20)
    ax.text(1, -0.05, "Cases", color="black", ha="center", va="bottom", fontsize=20)

    if show_thresholds:
        for threshold in thresholds_choice[1:-1]:  # Skip the first and last elements (0 and 1)
            plt.axhline(y=threshold, linestyle="--", color="gray")

        color_levels = ["green", "yellow", "darkred"]
        for i in range(3):
            ax.axhspan(
                thresholds_choice[i], thresholds_choice[i + 1], color=color_levels[i], alpha=0.2, joinstyle="round"
            )
        display = "thresholds"

    else:
        display = "raw"

    plt.ylabel("Predicted Probability", fontsize=26)
    plt.xlabel("")
    plt.ylim(ylim[0], ylim[1])
    plt.title(title, pad=20, fontsize=26)

    svg_path = os.path.join(fig_path, f"Violin_{col_subset}_{row_subset}.svg")
    if save_fig:
        fig.savefig(svg_path, format="svg", bbox_inches="tight")

    return fig, ax


########### Plotting AUCs over time ###########

def plot_auc_time(
    self,
    *,  # Enforce keyword-only arguments
    readouts_list=[3, 5, 6, 7, 8, 9, 10, "all"],
    time_var="difftime",
    plot_figure=True,
    cohort="val",
    target="status",
    metric="both",  # Can be "AUROC", "AUPRC", or "both"
    export_format="svg",
    annotate_cases=True,  # Option to annotate the number of cases
    font_size=18,  # Option to adjust font size
    y1_lim=None,  # Option to scale AUROC y-axis
    y2_lim=None,  # Option to scale AUPRC y-axis
    remarks=None,  # Add remarks below the plot
    resample=True,
    resample_n=100,
    save_fig=True,
    change_legend_colors=False,
    figsize=(10, 6)
):
    """
    Get time-dependent AUROC and AUPRC and optionally plot a combined figure with two y-axes.

    Args:
        readouts_list (list, optional): List of timepoints for evaluation. Defaults to [3,4,5,6,7,8,9,10,'all'].
        time_var (str, optional): Variable within your trained model. Defaults to 'difftime'.
        plot_figure (bool, optional): Whether to plot the figure. Defaults to False.
        cohort (str, optional): Cohort to evaluate. Defaults to "val".
        target (str, optional): Target variable. Defaults to "status".
        metric (str, optional): Metric to plot ('AUROC', 'AUPRC', or 'both'). Defaults to 'both'.
        export_path (str, optional): File path to save the plot. Defaults to None.
        export_format (str, optional): Format for saving the plot ('png', 'svg', or 'pdf'). Defaults to 'png'.
        annotate_cases (bool, optional): Whether to annotate the number of cases. Defaults to True.
        font_size (int, optional): Font size for the plot text. Defaults to 18.
        y1_lim (tuple, optional): Limits for the AUROC y-axis (e.g., (0.5, 1.0)). Defaults to None.
        y2_lim (tuple, optional): Limits for the AUPRC y-axis (e.g., (0.0, 1.0)). Defaults to None.
        remarks (str, optional): Remarks or notes to display below the plot. Defaults to None.
        save_fig (bool, optional): Whether to save the figure. Defaults to True.

    Returns:
        pd.DataFrame, plt.Axes: Exported DataFrame with metrics and plot axis if requested.
    """
    # import and adjust the functions
    from sklearn.metrics import roc_auc_score, average_precision_score
    import random
    from imblearn.under_sampling import RandomUnderSampler

    # Define custom colors for AUROC and AUPRC
    roc_color = "#5472beff"  # Default ROC color
    prc_color = "#be5454ff"  # Default PRC color

    if hasattr(self.user_input, "color_groups_all"):
        color_groups = getattr(self.user_input, "color_groups_all")
        if isinstance(color_groups, dict):
            roc_color = color_groups.get("Roc", roc_color)
            prc_color = color_groups.get("Prc", prc_color)


    # Get time-dependent status variable
    row = self.user_input.row_subset
    col = self.user_input.col_subset
    fig_path = self.user_input.fig_path
    name = f"{row}_{col}_{self.user_input.target_to_validate_on}_{self.name}"
    df = self.eval.test_train_pred.get(cohort).copy()

    if readouts_list == ["linspace"]:
        readouts = np.linspace(df[time_var].max() * -1, df[time_var].min() * -1, 20)[1:]
    else:
        readouts = readouts_list

    n_total = df.shape[0]

    for endpoint in readouts:
        if endpoint != "all":
            df[f"status_t{endpoint}"] = (df[target] == 1) & (df[time_var] * -1 <= endpoint)
        else:
            df[f"status_t{round(df[time_var].min()*-1)}"] = df[target]

    export_df = pd.DataFrame({"n_cases": df.loc[:, df.columns.str.startswith("status_t")].sum(), "n_total": n_total})
    export_df["n_controls"] = export_df.n_total - export_df.n_cases

    # Get time-dependent metrics
    aucs, auprcs, timepoints = [], [], []
    aucs_under, auprcs_under = [], []

    for endpoint in df.columns[df.columns.str.startswith("status_t")].tolist():
        timepoints.append(float(endpoint.lstrip("status_t")))
        aucs.append(roc_auc_score(df[endpoint], df.y_pred))
        auprcs.append(average_precision_score(df[endpoint], df.y_pred))

        # undersampling option
        if resample:
            a, p = [], []
            for i in random.sample(range(10000), resample_n):
                rus = RandomUnderSampler(random_state=i)
                X, y = df, df[endpoint]
                X_resampled, y_resampled = rus.fit_resample(X, y)
                a.append(roc_auc_score(X_resampled.status, X_resampled.y_pred))
                p.append(average_precision_score(X_resampled.status, X_resampled.y_pred))
            aucs_under.append(np.mean(a))
            auprcs_under.append(np.mean(p))


    if resample:
        export_df["AUROC"] = aucs
        export_df["mean_AUROC"] = aucs_under
        export_df["SD_AUROC"] = np.std(aucs_under)
        export_df["AUPRC"] = auprcs
        export_df["mean_AUPRC"] = auprcs_under
        export_df["SD_AUPRC"] = np.std(auprcs_under)
        export_df["readout [years]"] = timepoints
    else:
        export_df["AUROC"] = aucs
        export_df["AUPRC"] = auprcs
        export_df["readout [years]"] = timepoints

    # Plot the selected metric if requested
    if True:
        fig, ax1 = plt.subplots(figsize=figsize)

        if metric == "both":
            # Combined plot with two y-axes
            ax1.set_xlabel("Time (Years)", fontsize=font_size)
            ax1.set_ylabel("AUROC", color=roc_color, fontsize=font_size)
            sns.lineplot(
                data=export_df, x="readout [years]", y="AUROC", ax=ax1, marker="o", linewidth=2, color=roc_color
            )
            ax1.tick_params(axis="y", labelcolor=roc_color, labelsize=font_size)
            ax1.tick_params(axis="x", labelsize=font_size)
            if y1_lim:
                ax1.set_ylim(y1_lim)

            ax2 = ax1.twinx()
            ax2.set_ylabel("AUPRC", color=prc_color, fontsize=font_size)
            sns.lineplot(
                data=export_df, x="readout [years]", y="AUPRC", ax=ax2, marker="o", linewidth=2, color=prc_color
            )
            ax2.tick_params(axis="y", labelcolor=prc_color, labelsize=font_size)
            if y2_lim:
                ax2.set_ylim(y2_lim)

            # Annotate AUROC
            if annotate_cases:
                for i, row in export_df.iterrows():

                    label_text = f"{int(row['n_cases'])}"
                    if i == "status_t3":
                        label_text += "*"

                    ax1.annotate(
                        label_text,
                        (row["readout [years]"], row["AUROC"]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=font_size - 2,
                        color=roc_color,
                    )

        elif metric == "AUROC":
            sns.lineplot(
                data=export_df, x="readout [years]", y="AUROC", ax=ax1, marker="o", linewidth=2, color=roc_color
            )
            ax1.set_title("AUROC over Time", fontsize=font_size)
            ax1.set_xlabel("Time (Years)", fontsize=font_size)
            ax1.set_ylabel("AUROC", fontsize=font_size)
            ax1.tick_params(axis="both", labelsize=font_size)
            if y1_lim:
                ax1.set_ylim(y1_lim)

            if annotate_cases:
                for i, row in export_df.iterrows():
                    label_text = f"{int(row['n_cases'])}"
                    if i == "status_t3":
                        label_text += "*"

                    ax1.annotate(
                        label_text,
                        (row["readout [years]"], row["AUROC"]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=font_size - 2,
                        color=roc_color,
                    )

        elif metric == "AUPRC":
            sns.lineplot(
                data=export_df, x="readout [years]", y="AUPRC", ax=ax1, marker="o", linewidth=2, color=prc_color
            )
            ax1.set_title("AUPRC over Time", fontsize=font_size)
            ax1.set_xlabel("Time (Years)", fontsize=font_size)
            ax1.set_ylabel("AUPRC", fontsize=font_size)
            ax1.tick_params(axis="both", labelsize=font_size)
            if y2_lim:
                ax1.set_ylim(y2_lim)

            if annotate_cases:
                for i, row in export_df.iterrows():
                    label_text = f"{int(row['n_cases'])}"
                    if i == "status_t3":
                        label_text += "*"

                    ax1.annotate(
                        label_text,
                        (row["readout [years]"], row["AUPRC"]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=font_size - 2,
                        color=prc_color,
                    )
        if remarks:
            plt.figtext(
                0.97, 0.0, remarks, wrap=True, horizontalalignment="right",
                fontsize=font_size - 2, color="gray", style="italic"
            )

        fig.tight_layout()
        if save_fig:
            metric_suffix = metric.replace("-", "_")
            adjusted_path = os.path.join(fig_path, f"{name}_time_dep_AUC_{metric_suffix}.{export_format}")
            plt.savefig(adjusted_path, format=export_format, dpi=600, bbox_inches="tight", transparent=True)
            print(f"Plot saved to: {adjusted_path}")
        if plot_figure:
            plt.show()

    return export_df, ax1

def plot_ablation_dual_visual(
    df,
    x_col,
    y1_col,
    y2_col,
    y1_label="AUROC",
    y2_label="AUPRC",
    x_label="Number of included features",
    title="Scorer Results vs Number of Included Features",
    y1_color="blue",
    y2_color="orange",
    y1_ylim=None,
    y2_ylim=None,
    x_lim=None,  # Custom x-axis limits
    x_reverse=True,
    save_path=None,
    save_format="svg"
):
    """
    Creates a dual-axis line plot with two y-axes.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The column name for the x-axis.
        y1_col (str): The column name for the first y-axis (left).
        y2_col (str): The column name for the second y-axis (right).
        y1_label (str): Label for the left y-axis. Default is "AUROC".
        y2_label (str): Label for the right y-axis. Default is "AUPRC".
        x_label (str): Label for the x-axis.
        title (str): Title of the plot.
        y1_color (str): Line color for the first y-axis. Default is "blue".
        y2_color (str): Line color for the second y-axis. Default is "orange".
        y1_ylim (tuple): Limits for the left y-axis (AUROC). Default is None.
        y2_ylim (tuple): Limits for the right y-axis (AUPRC). Default is None.
        x_lim (tuple): Custom limits for the x-axis. Default is None.
        x_reverse (bool): Whether to reverse the x-axis. Default is True.
        save_path (str): Path to save the figure. If None, the plot is not saved.
        save_format (str): Format to save the figure. Default is "svg".

    Returns:
        None
    """

    # Create the figure and primary axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the first y-axis (AUROC)
    sns.lineplot(data=df, x=x_col, y=y1_col, ax=ax1, color=y1_color, label=y1_label, legend=False)
    ax1.set_ylabel(y1_label, color=y1_color)
    ax1.tick_params(axis='y', labelcolor=y1_color, colors=y1_color)
    ax1.spines['left'].set_color(y1_color)
    ax1.yaxis.label.set_color(y1_color)

    ax1.set_xlabel(x_label)

    # Set x-axis limits
    if x_lim:
        ax1.set_xlim(x_lim)
    elif x_reverse:
        ax1.set_xlim(df[x_col].max(), df[x_col].min())  # Reverse x-axis

    # Set y-axis limits if specified
    if y1_ylim:
        ax1.set_ylim(y1_ylim)

    fig.canvas.draw()

    # Create the secondary y-axis (AUPRC)
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x=x_col, y=y2_col, ax=ax2, color=y2_color, label=y2_label, legend=False)
    ax2.set_ylabel(y2_label, color=y2_color)
    ax2.tick_params(axis='y', labelcolor=y2_color, colors=y2_color)
    ax2.spines['right'].set_color(y2_color)

    if y2_ylim:
        ax2.set_ylim(y2_ylim)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=False)

    # Set title
    plt.title(title)
    # Save the figure if a path is specified
    if save_path:
        plt.savefig(f"{save_path}.{save_format}", format=save_format, dpi=300, bbox_inches="tight", transparent=True)
        print(f"Plot saved to: {save_path}.{save_format}")

    # Show the plot
    plt.show()



def plot_precision_recall_curves(
    dataframes, keys_ordered, colors, fig, ax, display='',
    xlim=(0,1), ylim=(0,1), fill_bet=False, title='',
    fig_path=None, line_style='-', dotted_keys=None,
    plot_legend=True, lw=2, font_size=26, truth="status_cancerreg",
    export_format="svg"  # Default export format
):
    """
    Plots overlaying precision-recall curves for multiple datasets.

    Parameters:
    - dataframes (dict): Dictionary of dataframes containing columns 'proba' and 'true'.
    - keys_ordered (list): List of keys in the order they should be plotted.
    - colors (dict): Dictionary of colors corresponding to each key.
    - fig (matplotlib.figure.Figure): Figure object to plot on.
    - ax (matplotlib.axes.Axes): Axes object to plot on.
    - display (str): Label for the display.
    - fill_bet (bool): Whether to fill the area between the standard deviation bounds.
    - title (str): Title of the plot.
    - fig_path (str): Path to save the figure.
    - line_style (str): Line style for the mean curve.
    - dotted_keys (list): List of keys to be displayed with dotted lines.
    - plot_legend (bool): Whether to include a legend.
    - lw (int): Line width of the curves.
    - font_size (int): Font size for labels and title.
    - truth (str): Name of the column representing the ground truth.
    - export_format (str): File format for export ("svg" or "png").
    """

    mean_precisions = []
    base_recall = np.linspace(0, 1, 100)
    dotted_keys = dotted_keys or []

    for key in keys_ordered:
        df = dataframes[key]
        precision, recall, _ = precision_recall_curve(df[truth], df["y_pred"])
        pr_auc = auc(recall, precision)

        # Ensure non-decreasing precision by taking cumulative max
        precision_inv = np.flip(precision)
        recall_inv = np.flip(recall)
        decreasing_max_precision = np.maximum.accumulate(precision_inv[::-1])[::-1]

        # Interpolating precision at standard recall levels
        mean_precision = np.interp(base_recall, recall[::-1], decreasing_max_precision)
        mean_precisions.append(mean_precision)

        linestyle = '--' if key in dotted_keys else line_style
        label = key.split('_')[-1]
        ax.plot(recall, precision, alpha=1, lw=lw, linestyle=linestyle,
                color=colors[key], label=f'{label} ({pr_auc:.2f})')

    fig.set_size_inches(5, 4.1)

    mean_precisions = np.array(mean_precisions)
    mean_precision = mean_precisions.mean(axis=0)
    std_precision = mean_precisions.std(axis=0)
    pr_auc = auc(base_recall, mean_precision)

    if fill_bet:
        precision_upper = np.minimum(mean_precision + std_precision, 1)
        precision_lower = mean_precision - std_precision
        ax.fill_between(base_recall, precision_lower, precision_upper, color='grey', alpha=0.2)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Recall (TP / (TP + FN))', fontsize=font_size)
    ax.set_ylabel('Precision (TP / (TP + FP))', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size, pad=-5)
    ax.set_title(f"{title}: {display}", fontsize=font_size, pad=5)
    ax.xaxis.set_tick_params(pad=-5)
    ax.yaxis.set_tick_params(pad=-5)

    # Font settings for legend
    if plot_legend:
        condensed_font = fm.FontProperties(family='Arial',style='normal',weight='normal', stretch='condensed')
        ax.legend(loc="upper right", bbox_to_anchor=(1.01, 1), fontsize=6, frameon=False, prop=condensed_font)


    # Set axis borders
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # Ensure export path exists
    if fig_path:
        file_name = f"Prec_Recall_{display}_{ylim}.{export_format}"
        export_file = os.path.join(fig_path, file_name)

        # Save figure with transparent background
        fig.savefig(export_file, format=export_format, bbox_inches='tight', transparent=True, dpi=300)
        print(f"Plot saved to: {export_file}")



def shap_analysis(
    self,
    sample_size=None,
    max_display=20,
    fig_size=(12, 8),
    export_path=None,
    export_format="svg",  # Can be "png", "svg", "pdf"
    plot_type="dot",  # Options: "dot", "bar"
    custom_color=None,  # Allows custom colormap
    short_names=True,  # Whether to use short names for features
    manual_name_substitutions={},  # Dictionary for manual name substitutions
    debug_mapping=False,
    save_fig=True
):
    """
    Generate a SHAP summary plot for the best-performing estimator.

    Args:
        sample_size (int, optional): Number of samples to use for SHAP analysis. If None, use all samples.
        max_display (int, optional): Maximum number of features to display.
        fig_size (tuple, optional): Size of the figure.
        export_path (str, optional): Path to save the figure.
        export_format (str, optional): File format for export ("png", "svg", "pdf").
        plot_type (str, optional): Type of SHAP plot ("dot" or "bar").
        custom_color (str, optional): Custom colormap for the SHAP plot.
        short_names (bool): Whether to use short names for features
        manual_name_substitutions (dict): Dictionary for manual name substitutions
        debug_mapping (bool): Show detailed name mapping information

    Returns:
        matplotlib.figure.Figure: SHAP summary plot.
    """
    import copy

    # Print function to help with debugging
    def debug_print(msg, data=None, enabled=debug_mapping):
        if enabled:
            print(f"SHAP DEBUG: {msg}")
            if data is not None:
                if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
                    if len(data) > 10:
                        print(data.head(5).to_string())
                        print("...")
                        print(data.tail(5).to_string())
                    else:
                        print(data.to_string())
                else:
                    print(data)
            print("-" * 50)


    # ✅ Extract Feature Names from OHE (One-Hot Encoder)
    feature_names = pd.Series(self.ohe.get_feature_names_out()).str.replace("remainder__", "").str.replace("one_hot_encoder__", "").str.replace("scaler__", "")
    debug_print("Initial feature names after removing prefixes:", feature_names)

    short_name_mapping = {}
    # ✅ Get better feature names
    if short_names:
        # First try to get short names from X_ohe_map if available
        if hasattr(self.data, "X_ohe_map") and "short_name" in self.data.X_ohe_map.columns:
            # Create mapping from original feature name to short name
            debug_print("X_ohe_map structure:", self.data.X_ohe_map[["name_print", "short_name"]].head())
            # Check for NaN values in short_name
            nan_count = self.data.X_ohe_map["short_name"].isna().sum()
            debug_print(f"NaN values in short_name column: {nan_count} out of {len(self.data.X_ohe_map)}")

            # Create mapping from original feature name to short name, filtering out NaN values
            valid_short_names = self.data.X_ohe_map[["name_print", "short_name"]].dropna(subset=["short_name"])
            short_name_mapping = dict(zip(valid_short_names["name_print"], valid_short_names["short_name"]))

            debug_print(f"Created short_name_mapping with {len(short_name_mapping)} entries:", short_name_mapping)

            # Check which feature names are in the mapping
            in_mapping = feature_names.map(lambda x: x in short_name_mapping)
            debug_print(f"Features in mapping: {in_mapping.sum()} out of {len(feature_names)}")

            # Show features missing from mapping
            missing_features = feature_names[~in_mapping]
            debug_print("Features missing from mapping:", missing_features)

            # Apply mapping to feature names, keep original if no match
            before_mapping = feature_names.copy()
            feature_names = feature_names.map(lambda x: short_name_mapping.get(x, x))

            # Show before/after for features that were mapped
            mapped_features = before_mapping != feature_names
            debug_print(f"Successfully mapped {mapped_features.sum()} features:")
            if mapped_features.sum() > 0:
                mapping_comparison = pd.DataFrame({
                    'Original': before_mapping[mapped_features],
                    'Mapped': feature_names[mapped_features]
                })
                debug_print("Mapping comparison:", mapping_comparison)


        # If not in X_ohe_map, try to load from columngroups.csv
        elif hasattr(self, "user_input"):
            debug_print("No X_ohe_map with short_name, trying columngroups.csv")
            try:
                from pp import load_columngroups
                colgroups = load_columngroups(self.user_input)
                debug_print("Loaded columngroups:", colgroups.columns)

                if "short_name" in colgroups.columns and "column_name" in colgroups.columns:
                    debug_print("Found short_name and column_name in columngroups")

                    # Check for NaN values
                    nan_count = colgroups["short_name"].isna().sum()
                    debug_print(f"NaN values in short_name column: {nan_count} out of {len(colgroups)}")

                    short_map = colgroups.set_index("column_name")["short_name"].dropna()
                    debug_print(f"Created short_map with {len(short_map)} entries:", short_map)

                    # Check which feature names are in the mapping
                    in_mapping = feature_names.map(lambda x: x in short_map)
                    debug_print(f"Features in mapping: {in_mapping.sum()} out of {len(feature_names)}")

                    # Apply mapping to feature names, keep original if no match
                    before_mapping = feature_names.copy()
                    feature_names = feature_names.map(lambda x: short_map.get(x, x))

                    # Show before/after for features that were mapped
                    mapped_features = before_mapping != feature_names
                    debug_print(f"Successfully mapped {mapped_features.sum()} features:")
                    if mapped_features.sum() > 0:
                        mapping_comparison = pd.DataFrame({
                            'Original': before_mapping[mapped_features],
                            'Mapped': feature_names[mapped_features]
                        })
                        debug_print("Mapping comparison:", mapping_comparison)
            except Exception as e:
                debug_print(f"Error loading columngroups: {e}")

    name_substitutions = manual_name_substitutions
    for old_name, new_name in name_substitutions.items():
        feature_names = feature_names.replace(old_name, new_name)

    X_val = pd.DataFrame(self.ohe.transform(self.data.X_val), columns=feature_names)
    X_val.rename(columns= {'AGE': 'Age', 'Gamma glutamyltransferase': 'Gamma GT'}, inplace=True)
    # ✅ Sample the dataset if needed
    if sample_size and len(X_val) > sample_size:
        X_val_sample = X_val.sample(sample_size, random_state=42)
    else:
        X_val_sample = X_val

    # ✅ Select the Best Model Efficiently
    best_model = max(self.master_RFC.models, key=lambda gs: gs.best_score_).best_estimator_

    if best_model is None:
        raise ValueError("No best estimator found in the model ensemble.")

    # ✅ Create SHAP Explainer
    try:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_val_sample)
    except Exception as e:
        raise RuntimeError(f"SHAP computation failed: {e}")

    # ✅ Handle Multi-Class Models by Choosing the Most Impactful Class
    # if isinstance(shap_values, list):
    #     class_contributions = [np.abs(values).mean() for values in shap_values]
    #     best_class = np.argmax(class_contributions)
    #     shap_values = shap_values[best_class]
    #     print(f"Using SHAP values for class index: {best_class}")

        # If shap_values is a list (for multi-class), take the positive class (index 1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        print(f"Using SHAP values for the positive class (if at Index 1)")

    # ✅ Create Figure
    fig, ax = plt.subplots(figsize=fig_size)

    # 🎨 Custom Colors
    if not custom_color:
       custom_color = "coolwarm"  # Default colormap

    # 🎨 Choose Plot Type (Dot or Bar)
    if plot_type == "dot":
        shap.summary_plot(shap_values, X_val_sample, plot_type="dot", max_display=max_display, show=False)
    elif plot_type == "bar":
        shap.summary_plot(shap_values, X_val_sample, plot_type="bar", max_display=max_display, show=False)
    else:
        raise ValueError("Invalid plot_type. Choose 'dot' or 'bar'.")

    # ✅ Adjust Layout & Labels
    plt.title(f"{self.user_input.col_subset} {self.user_input.row_subset}", fontsize=16, pad=20)
    plt.xlabel("SHAP value", fontsize=16)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    # Reduce space between lines and adjust y-axis limits
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min + 0.5, y_max - 0.5)

    plt.subplots_adjust(left=0.5, right=0.8, top=1, bottom=0.35, hspace=0)
    # 💾 Save Figure if Export Path is Provided
    export_path = self.user_input.fig_path
    svg_path = f"SHAP_{self.user_input.col_subset}_{self.user_input.row_subset}_{max_display}.{export_format}"
    if save_fig:
        suffix = getattr(self.user_input, "pl_suffix", None)
        if suffix:
            svg_path = f"{svg_path.rstrip(export_format)}_{suffix}.{export_format}"
        export_file = os.path.join(export_path, svg_path)
        plt.savefig(export_file, format=export_format, dpi=300, bbox_inches="tight", transparent=True)
        print(f"SHAP summary plot saved to: {export_file}")
    return plt.gca()




# ###################################################################################
# # only for a composed model


# def feature_importances_(self, ohe):
#     """get the mean(feature importance) of the best estimator as a pd.Series"""
#     export = pd.DataFrame()
#     for model, name in zip(self.models, np.arange(len(self.models))):
#         name = f"model_{str(name)}"
#         feature_imp = model.best_estimator_.feature_importances_
#         export[name] = feature_imp
#     export["mean_feature_imp"] = export.mean(axis=1)
#     export.set_axis(labels=ohe.get_feature_names_out().tolist())
#     return export


def feature_imp_aggregation(self, func_for_aggregation=np.sum, height_factor=0.5):
        """
        Creates a small aggregated feature importance plot for the trained model in the Pipeline.

        Parameters:
        - func_for_aggregation (function): Aggregation method (np.sum or np.mean).
        - height_factor (float): Adjusts plot height dynamically.

        Returns:
        - fig, ax: Matplotlib figure and axis objects.
        """

        if not hasattr(self, "master_RFC"):
            raise ValueError("No trained model found in pipeline. Run 'build_master_RFC()' first.")

        try:
            feature_imp_all = self.master_RFC.feature_importances_()
            feature_imp = feature_imp_all["mean_feature_imp"]
        except Exception as e:
            print(f"Exception occurred: {e}")
            feature_imp = self.master_RFC.feature_importances_

        if feature_imp.shape[0] != self.data.X_ohe_map.shape[0]:
            print("Warning: Feature importance and mapping have different lengths.")

        # Define source mapping
        df_source_map_rename = {
            "df_covariates": "Demography\n& Lifestyle",
            "df_diagnosis": "EHR",
            "df_blood": "Blood count\n& Serum",
            "df_snp": "Genomics",
            "df_genomics": "Genomics",
            "df_metabolomics": "Metabolomics",
            "df_metadata": "Metadata",
        }

        # Create DataFrame for plotting
        plot_X = self.data.X_ohe_map.copy()
        plot_X["feature_imp"] = np.float64(feature_imp)
        plot_X["source_lit"] = plot_X.source.map(df_source_map_rename)

        # Compute aggregated feature importance
        aggregated_imp = plot_X.groupby("source_lit")["feature_imp"].agg(func_for_aggregation).reset_index()

        # Remove Metadata df for cleaner representation of Data Modalities
        aggregated_imp = aggregated_imp[aggregated_imp["source_lit"] != "Metadata"]

        categories_in_data = aggregated_imp["source_lit"].unique()
        # Define custom order
        custom_order = ["Demography\n& Lifestyle","EHR","Blood count\n& Serum", "Genomics", "Metabolomics"]
         # Filter custom order to only include categories that are in the data
        filtered_order = [category for category in custom_order if category in categories_in_data]
         # Set the custom order for the plot
        aggregated_imp["source_lit"] = pd.Categorical(aggregated_imp["source_lit"], categories=filtered_order, ordered=True)
        # Fix the missing keys issue in color_dict
        unique_sources = aggregated_imp["source_lit"].unique()

        # Adjust plot height dynamically
        bar_height = max(3, len(unique_sources) * height_factor)  # Ensure height scales properly

        # Create standalone small plot
        fig, ax = plt.subplots(figsize=(5, bar_height))
        sns.barplot(
            data=aggregated_imp,
            x="feature_imp",
            y="source_lit",
            palette=self.mapper.color_groups,
            hue="source_lit",
            legend=False,
            ax=ax,
            err_kws={'color': 'lightgray'},
        )

        ax.set_ylabel("")
        ax.set_xlabel("Feature Importance (Aggregated)")
        ax.set_title(f"{func_for_aggregation.__name__.capitalize()} Feature Importance by Group")

        # Save plot
        fig_path = os.path.join(self.user_input.fig_path, f"Feature_Importance_Summary_{self.user_input.col_subset}_{func_for_aggregation.__name__}.svg")
        fig.savefig(fig_path, format="svg", bbox_inches="tight", transparent=True)

        print(f"Feature importance aggregation plot saved to: {fig_path}")

        return fig, ax


def plot_kaplan_meier(
    self,
    thresholds: Tuple[float, float] = (0.4, 0.6),
    color_dict: Dict[str, str] = {"Low Risk": "green", "Medium Risk": "orange", "High Risk": "red"},
    cohort: str = "val",
    target: str = "status",
    time_variable: str = "date_of_diag",
    risk_score_var: str = "y_pred",
    x_scale: str = "y",  # "d", "m", "y"
    y_mode: str = "survival",  # or "risk"
    font_size: int = 20,
    save_fig: bool = True,
    y_scale: Union[str, Tuple[float, float]] = "default",
    fig_path: Optional[str] = None,
    doi_label: Optional[str] = None
) -> KaplanMeierFitter:
    """
    Unified Kaplan-Meier plotting function combining log-rank test, customizable axes, risk groups, and flexible input.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import os

    # 1. Risk group assignment
    def assign_risk_groups(scores, thresholds):
        return np.select(
            [scores < thresholds[0],
             (scores >= thresholds[0]) & (scores < thresholds[1]),
             scores >= thresholds[1]],
            ['Low Risk', 'Medium Risk', 'High Risk']
        )

    # 2. Get cohort data
    try:
        z = self.eval.test_train_pred.get(cohort)
    except AttributeError:
        print(f"KM_plot - Using df_y_orig as fallback for cohort={cohort}.")
        z = self.data.df_y_orig.copy()

    # 3. Time calculation
    censor_date = pd.Timestamp(2024, 1, 1)
    z[time_variable] = pd.to_datetime(z[time_variable].fillna(censor_date))
    assessment_col = "assessment" if "assessment" in z else "Date of assessment"
    z[assessment_col] = pd.to_datetime(z[assessment_col])
    timedelta = z[time_variable] - z[assessment_col]
    z["time_to_event_d"] = timedelta.dt.days
    z["time_to_event_m"] = timedelta.dt.days / 30
    z["time_to_event_y"] = timedelta.dt.days / 365.25
    time_col = f"time_to_event_{x_scale}"

    # 4. Ensure risk score exists
    if risk_score_var not in z:
        print(f"{risk_score_var} not found, recomputing risk scores...")
        z[risk_score_var] = self.master_RFC.predict_proba(self.ohe.transform(self.data.X_val))[:, 1]

    # 5. Assign risk groups
    z["risk_group"] = assign_risk_groups(z[risk_score_var], thresholds)
    print("Risk group distribution:\n", z["risk_group"].value_counts())

    # 6. Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    kmf = KaplanMeierFitter()
    label_ranges = {
        "Low Risk": f"< {thresholds[0]}",
        "Medium Risk": f"{thresholds[0]} – {thresholds[1]}",
        "High Risk": f"≥ {thresholds[1]}"
    }

    for group in ["Low Risk", "Medium Risk", "High Risk"]:
        group_data = z[z["risk_group"] == group]
        if not group_data.empty:
            label = f"{group} ({label_ranges[group]}) (n={len(group_data)})"
            kmf.fit(durations=group_data[time_col], event_observed=group_data[target], label=label)
            kmf.plot_survival_function(ax=ax, color=color_dict.get(group, "gray"), alpha=0.8, linewidth=2)

    # Axis setup
    time_unit = {"d": "Days", "m": "Months", "y": "Years"}[x_scale]
    ylabel = f"1 - Survival Probability"
    if doi_label:
        ylabel = f"1 - Probability of {doi_label} [%]"

    ax.set_xlabel(f"Time ({time_unit})", fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_title(f"Time to {self.user_input.DOI} per Risk Group", fontsize=font_size)
    ax.legend(fontsize=font_size * 0.7, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=font_size * 0.8)
    if y_scale != "default":
        ax.set_ylim(y_scale)

    # Log-rank test
    try:
        high = z[z["risk_group"] == "High Risk"]
        low = z[z["risk_group"] == "Low Risk"]
        if not high.empty and not low.empty:
            res = logrank_test(high[time_col], low[time_col], event_observed_A=high[target], event_observed_B=low[target])
            ax.text(0.02, 0.2, f"Log-rank p = {res.p_value:.4f}", transform=ax.transAxes, fontsize=font_size * 0.6,
                    bbox=dict(facecolor='white', alpha=0.7))
    except Exception as e:
        print("Log-rank test failed:", e)

    plt.tight_layout()

    # Save
    if save_fig:
        if fig_path is None:
            fig_path = os.path.join(self.user_input.fig_path, f"KaplanMeier_{self.user_input.col_subset}_{y_mode}.svg")
        fig.savefig(fig_path, format="svg", bbox_inches="tight", transparent=True)
        print(f"Figure saved to {fig_path}")

    plt.show()
    return kmf





# 15 Export interpolated Precision Recall curves


def export_interpolated_pr_curves(pl, path=None, cohort='val', save_format="joblib", biobank=None):
    """
    Export interpolated precision-recall curves for external validation using the
    test_train_pred DataFrame, similar to your TPRS_combined.joblib approach.

    Parameters:
    -----------
    pl_ext : pipeline object
        Your pipeline object containing eval.test_train_pred data
    cohort : str
        Which dataset to export ('train', 'test', 'val')
    save_format : str
        File format: 'joblib', 'csv', or 'xlsx'

    Returns:
    --------
    pd.DataFrame : DataFrame containing interpolated precision-recall data
    """

    print(f"Starting export_interpolated_pr_curves for {cohort} cohort using format: {save_format}")

    # Get the DataFrame from test_train_pred
    if cohort not in pl.eval.test_train_pred:
        print(f"{cohort} data not available in test_train_pred")
        return pd.DataFrame()

    df = pl.eval.test_train_pred[cohort].copy()
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")

    # Define recall range (100 points from 0 to 1, as commonly used)
    recall_base = np.linspace(0, 1, 100)

    # Initialize DataFrame to store interpolated precisions
    prc_interpolated = pd.DataFrame()

    # Get model prediction columns (y_pred_val_model_0, y_pred_val_model_1, etc.)
    model_cols = [col for col in df.columns if col.startswith('y_pred_val_model_')]

    # If no specific model columns, use the main y_pred column
    if not model_cols and 'y_pred' in df.columns:
        model_cols = ['y_pred']

    # True labels - assuming 'status' is the target variable
    if 'status' not in df.columns:
        print("Warning: 'status' column not found. Available columns:", df.columns.tolist())
        return pd.DataFrame()

    y_true = df['status']

    print(f"Found {len(model_cols)} model prediction columns: {model_cols}")

    # Process each model's predictions
    for model_col in model_cols:
        if model_col in df.columns:
            y_pred = df[model_col]

            # Remove any NaN values
            valid_mask = ~(pd.isna(y_true) | pd.isna(y_pred))
            y_true_clean = y_true[valid_mask]
            y_pred_clean = y_pred[valid_mask]

            if len(y_true_clean) == 0:
                print(f"Warning: No valid data for {model_col}")
                continue

            # Calculate precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_true_clean, y_pred_clean)

            # Sort by recall (ascending) for proper interpolation
            sorted_indices = np.argsort(recall)
            recall_sorted = recall[sorted_indices]
            precision_sorted = precision[sorted_indices]

            # Interpolate precision at the base recall points
            precision_interp = np.interp(recall_base, recall_sorted, precision_sorted)

            # Store in DataFrame with appropriate column name
            col_name = model_col.replace('y_pred_val_model_', 'model_').replace('y_pred', 'Precision')
            prc_interpolated[col_name] = precision_interp

            print(f"Processed {model_col} -> {col_name}")

    if prc_interpolated.empty:
        print("No valid model predictions found")
        return pd.DataFrame()

    # Save using the same structure as save_performance_combination

    row_subset = pl.user_input.row_subset
    col_subset = pl.user_input.col_subset
    estimator = pl.model_type
    identifier = f"{biobank}_{row_subset}_{col_subset}_{estimator}"



    #Either save the PRC in the pipeline specific output path or in the path given by the user
    if path is None:
        combined_output_path = os.path.join(
            pl.pipeline_output_path,
            f"combined_output/{cohort}"
    )

    else:
          combined_output_path = path

    os.makedirs(combined_output_path, exist_ok=True)

    # Rename columns with identifier (same pattern as TPRS)
    prc_export = prc_interpolated.rename(columns=lambda x: f"{identifier}_{str(x)}")

    if save_format == "joblib":
        prc_combined_path = os.path.join(combined_output_path, "PRC_combined.joblib")

        # Load existing file or create new DataFrame
        if os.path.exists(prc_combined_path):
            prc_combined = joblib.load(prc_combined_path)
        else:
            prc_combined = pd.DataFrame()

        # Remove any existing columns with the same identifier
        prc_combined = prc_combined.drop(columns=[col for col in prc_combined.columns
                                                  if col.startswith(f"{identifier}_")], errors='ignore')

        # Add new data
        prc_combined = pd.concat([prc_combined, prc_export], axis=1)

        # Save
        joblib.dump(prc_combined, prc_combined_path)
        print(f"PRC data saved to {prc_combined_path}")

    else:
        # Handle CSV/Excel formats
        prc_combined_path = os.path.join(combined_output_path, "PRC_combined.xlsx")

        if os.path.exists(prc_combined_path):
            if save_format == "csv":
                prc_combined = pd.read_csv(prc_combined_path.replace(".xlsx", ".csv"))
            else:
                prc_combined = pd.read_excel(prc_combined_path)
        else:
            prc_combined = pd.DataFrame()

        # Remove existing columns with same identifier
        current_cols = [col for col in prc_combined.columns if col.startswith(f"{identifier}_")]
        if current_cols:
            prc_combined = prc_combined.drop(columns=current_cols)

        # Add new data
        prc_combined_export = pd.concat([prc_combined, prc_export], axis=1)

        # Save in requested format
        if save_format == "csv":
            csv_path = prc_combined_path.replace(".xlsx", ".csv")
            prc_combined_export.to_csv(csv_path, index=False)
            print(f"PRC data exported to {csv_path}")
        else:
            prc_combined_export.to_excel(prc_combined_path, index=False)
            print(f"PRC data exported to {prc_combined_path}")

    print(f"Exported PRC data shape: {prc_export.shape}")
    return prc_export