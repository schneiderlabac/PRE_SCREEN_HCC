# Plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
import shap


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


def feature_imp_barplot(
    model,
    X_ohe_map,
    pip_self,
    n_features=60,
    xlabel="Feature importance",
    ylabel=None,
    small_plot=True,
    fontsize_smallplot=3,
    pos_smallplot="lower right",
    size_smallplot="30%",
    func_for_aggregation=np.sum,
    features_to_exclude=[],
    export=True,
    color_dict=None,
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
        model (model, optional): Model to evaluate and draw the feature importance from. Defaults to model.
        X_ohe_map (df, optional): df for the column mapping with the featuresorce as sorce and name as. Defaults to X_ohe_map.
        n_features (int, or 'all'): n features to plot in the big plot, the aggregation in the smaller one is always referring to the whole dataset passed. Defaults to 60.
        xlabel (str, optional): xlabel. Defaults to 'Feature importance'.
        ylabel (str, optional): ylabel. Defaults to 'Features'.
        small_plot (bool, optional): plot the small additional plot in the biger one?. Defaults to True.
        fontsize_smallplot (int, optional): Defaults to 7.
        pos_smallplot (str, optional): plot the small aggregation of the biger figure. Defaults to 'center right'.
        size_smallplot (str, optional): percentage of the big figure used for the small one. Defaults to '30%'.
        func_for_aggregation (func, optional): function for the aggregation f.i. np.sum or np.mean. Defaults to np.sum.
        featrures_to_exclude (list): list of features to exclude from the plotting
    Returns:
        fig, ax, : the figure and axis object of the created plots
    """
    try:
        feature_imp_all = model.feature_importances_()
        feature_imp = feature_imp_all["mean_feature_imp"]
        is_master_model = True
    except Exception as e:
        print(f"Got this Exception {e}")
        print("For the feature_imp. plot only a single model was used.".center(100, "-"))
        feature_imp = model.feature_importances_

    ## test for consistency of maping structure with the passed model
    col_subset = pip_self.user_input.col_subset
    row_subset = pip_self.user_input.row_subset
    fig_path = pip_self.user_input.fig_path

    if feature_imp.shape[0] != X_ohe_map.shape[0]:
        print("feature_imp and mapping have not the same number of columns represented\n")

    # create color sheme and add to the X_ohe_map plot
    if color_dict is None:
        color_dict = dict(
            zip(X_ohe_map.source.unique(), sns.color_palette("pastel", n_colors=len(X_ohe_map.source.unique())))
        )
    X_ohe_map["color"] = X_ohe_map.source.map(color_dict)

    # create df for plotting
    plot_X = X_ohe_map.copy()
    if is_master_model:
        plot_X[feature_imp_all.columns] = np.float64(feature_imp_all.values)
        plot_X.sort_values(by=["mean_feature_imp"], ascending=False, inplace=True)
    else:
        plot_X["feature_imp"] = np.float64(feature_imp)
        plot_X.sort_values(by=["feature_imp"], ascending=False, inplace=True)

    plot_X.drop(labels=features_to_exclude, axis=0, inplace=True)

    groups_with_features = plot_X["source"].unique()

    if n_features == "all":
        top_features = plot_X
    else:
        top_features = plot_X.iloc[:n_features, :]

    series = top_features.set_index(["source", "name_print", "color"]).stack()
    top_features = pd.DataFrame(series).reset_index()
    top_features.rename(columns={0: "feature_imp"}, inplace=True)
    plot_X.rename(columns={"mean_feature_imp": "feature_imp"}, inplace=True)
    plot_X["source_lit"] = plot_X.source.map(df_source_map_rename)
    bar_height = 17 * (n_features / 60)  # Define height of the plot according to number of features
    # Horizontal bar plot with feature importance for the top individual features
    fig, ax = plt.subplots(figsize=(10, bar_height))  # previously 20

    if is_master_model:
        sns.barplot(
            data=top_features,
            x="feature_imp",
            estimator=np.mean,
            y="name_print",
            hue="source",
            errorbar=("ci", 95),
            errcolor="lightgray",
            # err_kws={'alpha':0.2}, # only in newer versions
            saturation=1,
            dodge=False,
            palette=color_dict,
            ax=ax,
        )
    else:
        sns.barplot(
            x="feature_imp",
            y="name_print",
            hue="source",
            data=top_features,
            errorbar=None,
            dodge=False,
            saturation=1,
            palette=color_dict,
            ax=ax,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if small_plot:

        ax.legend().set_visible(False)
        # add a second plot to the axis with mean/sum of feature group
        axadded = inset_axes(ax, width=size_smallplot, height=size_smallplot, loc=pos_smallplot, borderpad=2)
        sns.barplot(
            x="feature_imp",
            y="source_lit",
            hue="source",
            order=[df_source_map_rename[src] for src in groups_with_features if src in df_source_map_rename],  # type: ignore
            dodge=False,
            estimator=func_for_aggregation,
            data=plot_X,
            # palette=dict(zip(pd.Series(color_groups.keys()).map({df_source_map_rename}),list(color_groups.values()))),
            palette=color_dict,
            ax=axadded,
            errcolor="lightgray",
        )
        axadded.bar_label(
            ax.containers[0],  # type: ignore
        )  # , fontsize=fontsize_smallplot)
        axadded.set_ylabel("")
        axadded.set_xlabel("")
        axadded.tick_params(reset=True, color="lightgray")
        axadded.legend().set_visible(False)
        axadded.set_title(
            f"{func_for_aggregation.__name__} by group".capitalize(), fontsize=plt.rcParams["legend.fontsize"]
        )

        svg_path = os.path.join(fig_path, f"Feature Imp_{col_subset}_{row_subset}_{n_features}.svg")
        fig.savefig(svg_path, format="svg", bbox_inches="tight")

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


def plot_rocs(tprs, aucs, n_splits, plot_all=True, y_amap=None, ax=plt.axes):
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
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    plt.tick_params(axis="both", which="major", labelsize=13)
    ax.legend(loc="lower right", fontsize=12)
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
    stratify=None
):
    DOI = pip_self.user_input.DOI
    ohe = pip_self.ohe
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
        df["y_true"] = y_true
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
        svg_path = os.path.join(fig_path, f"6xConfMatrix_{col_subset}_{row_subset}.svg")
        fig.savefig(svg_path, format="svg", bbox_inches="tight")


def create_violin_plot(
    pip_self,
    X_val,
    y_val,
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
):
    DOI = pip_self.user_input.DOI
    row_subset_long = pip_self.user_input.row_subset_long
    col_subset = pip_self.user_input.col_subset
    row_subset = pip_self.user_input.row_subset
    estimator = pip_self.model_type
    fig_path = pip_self.user_input.fig_path
    title = f"{col_subset} {row_subset} {estimator}"

    if palette is None:
        palette = {
            0: adjust_alpha(pip_self.mapper.color_groups_violin[pip_self.user_input.col_subset], 0.5),
            1: adjust_alpha(pip_self.mapper.color_groups_violin[pip_self.user_input.col_subset], 1),
        }

    pred_probs = model.predict_proba(ohe.transform(X_val))

    df = pd.DataFrame()
    df["status"] = y_val.status
    if type(pred_probs) != np.ndarray:
        df["proba"] = pred_probs.values
    else:
        df["proba"] = pd.DataFrame(pred_probs)[1].values
    if ax == None:
        fig, ax = plt.subplots(figsize=(3, 10))
        create_fig = True
    else:
        fig = ax.get_figure()
        create_fig = False

    sns.violinplot(
        data=df,
        y="proba",
        x="status",
        split=True,
        inner="quart",
        gap=gap,
        width=width,
        dodge="auto",
        palette=palette,
        hue="status",
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
    ax.text(0.1, -0.05, "TN", color="black", ha="center", va="bottom", fontsize=22)
    ax.text(1, -0.05, "TP", color="black", ha="center", va="bottom", fontsize=22)

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
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    return fig, ax



def plot_shap_summary(pip_self, shap_values, X, feature_names, max_display=20, fig_size= (12, 8)):
    """
    Create a SHAP summary plot for feature importance visualization.

    Args:
    pip_self (Pipeline): The Pipeline object containing user input information.
    shap_values (array or list): SHAP values calculated by the explainer.
    X (DataFrame): The input feature matrix.
    feature_names (list): List of feature names.
    max_display (int): Maximum number of features to display in the plot.

    Returns:
    matplotlib.figure.Figure: The figure object containing the SHAP summary plot.
    """
    col_subset = pip_self.user_input.col_subset
    row_subset = pip_self.user_input.row_subset
    fig_path = pip_self.user_input.fig_path


    # Check the type of shap_values and handle accordingly
    if isinstance(shap_values, list):
        # For multi-class models, use the mean absolute SHAP values across all classes
        shap_values = np.abs(np.array(shap_values)).mean(0)
    elif shap_values.ndim == 3:
        # For binary classification, use the positive class (index 1)
        shap_values = shap_values[:,:,1]

    # Create the SHAP summary plot
    fig, ax = plt.subplots(figsize=fig_size)
    shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=max_display, show=False, plot_type="bar")

    plt.tight_layout()

    # Save the figure
    if pip_self.user_input.visual_export:
        svg_path = os.path.join(fig_path, f"Shapley_{col_subset}_{row_subset}.svg")
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        print(f"SHAP summary plot saved to: {svg_path}")

    return fig


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
