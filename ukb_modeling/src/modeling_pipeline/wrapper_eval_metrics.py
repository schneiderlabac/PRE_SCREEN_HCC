import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
import seaborn as sns
import os
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from scipy import stats
from itertools import combinations
from modeling_pipeline.pipeline import *
import os
import yaml
import warnings
warnings.filterwarnings("ignore", message="indexing past lexsort depth may impact performance")




########################################################################################
def plot_metric_tradeoff(
    df,
    model_filter,
    dataset_filter,
    x_col="Recall",
    y_col="NNS",
    x_label="Recall (Sensitivity)",
    y_label="Number Needed to Screen (NNS)",
    thresholds_to_label=[0.4, 0.5, 0.6],
    font_size=14,
    dot_size=150,
    fig_path="./visuals/",
    name_suffix="",
    figsize=(10, 6),
    title=True
):
    """
    Create a plot showing the trade-off between two metrics across different thresholds.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the threshold-dependent metrics
    model_filter : str
        String to filter model names (e.g., "Model_TOP15")
    dataset_filter : str
        Dataset to filter (e.g., "par" or "all")
    x_col : str
        Column name to plot on x-axis
    y_col : str
        Column name to plot on y-axis
    x_label : str
        Label for x-axis
    y_label : str
        Label for y-axis
    thresholds_to_label : list
        List of threshold values to annotate on the plot
    font_size : int
        Base font size for plot elements
    dot_size : int
        Size of scatter points
    fig_path : str
        Directory path to save figure
    name_suffix : str
        Additional text to add to filename

    Returns:
    --------
    plt.Figure
        The matplotlib figure object
    """
    # Filter the dataframe
    df_filtered = df[
        (df["Model"].str.contains(model_filter)) &
        (df["Dataset"] == dataset_filter)
    ]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create scatter plot
    scatter = sns.scatterplot(
        data=df_filtered,
        x=x_col,
        y=y_col,
        hue="Threshold",
        palette="viridis",
        s=dot_size,
        alpha=0.8,
        ax=ax
    )

    # Connect points with a line
    plt.plot(df_filtered[x_col], df_filtered[y_col], 'k--', alpha=0.3, linewidth=2)

    # Annotate key threshold points
    for threshold in thresholds_to_label:
        point = df_filtered[df_filtered["Threshold"] == threshold]
        if not point.empty:
            plt.annotate(
                f"{threshold}",
                (point[x_col].values[0], point[y_col].values[0]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=font_size,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", linewidth=1.5)
            )

    # Format plot
    if title:
        ax.set_title(f"Trade-off between {x_col} and {y_col}\n{model_filter}, {dataset_filter}",
                 fontsize=font_size+4)
    ax.set_xlabel(x_label, fontsize=font_size+2)
    ax.set_ylabel(y_label, fontsize=font_size+2)
    ax.tick_params(axis='both', labelsize=font_size)
    ax.grid(False)

    # Adjust legend
    legend = ax.legend(title="Threshold", fontsize=font_size, title_fontsize=font_size,
                      frameon=False, loc='best')

    # Ensure directory exists
    os.makedirs(fig_path, exist_ok=True)

    # Create filename
    filename = f"{model_filter}_{dataset_filter}_{x_col}_vs_{y_col}"
    if name_suffix:
        filename += f"_{name_suffix}"
    filepath = os.path.join(fig_path, f"{filename}.svg")

    # Save the figure
    plt.tight_layout()
    plt.savefig(filepath, format="svg", bbox_inches="tight", dpi=300, transparent=True)

    return fig