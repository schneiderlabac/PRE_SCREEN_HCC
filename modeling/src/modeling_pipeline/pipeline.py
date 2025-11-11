import numpy as np
import pandas as pd
from joblib import dump, load
from modeling_pipeline.ablation import *
from modeling_pipeline.calibration import *
from modeling_pipeline.wrapper_roc_analysis import *
from modeling_pipeline.wrapper_violins_prcs import *
from modeling_pipeline.export_tables import *
from modeling_pipeline.helpers import *
import modeling_pipeline.pp as pp
import modeling_pipeline.plot as plot
import modeling_pipeline.training.models as models
import modeling_pipeline.training.train_test as train_test
from modeling_pipeline.external_validation import *

import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from typing import Literal,List, Dict, Optional, Tuple
import yaml
import time
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
        )



from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def load_Pipeline(path_to_load_from="."):
    """still in here just to keep the old functionallity; loads a pipeline from a joblib file

    Args:
        path_to_load_from (str, optional): _description_. Defaults to ".".

    Returns:
        _type_: _description_
    """
    self = load(path_to_load_from)
    self.user_input.path_loaded_from = path_to_load_from
    return self


def inspect_object(obj):
    """inspect an object to get a better understanding of its structure and function

    Args:
        obj (obj): to inspect
    """
    print("Attributes:\n")
    [print(func) for func in dir(obj) if not callable(getattr(obj, func)) and not func.startswith("__")]

    [print(func + "()") for func in dir(obj) if callable(getattr(obj, func)) and not func.startswith("__")]


################## just for the external validation -> work in progress!
class trained_model_ext_val:
    def __init__(self, models: dict, ohe=None, Master_RFC=None) -> None:
        if Master_RFC != None:
            self.model_with_info = {}
        if (
            list(models.values())[0] != dict
        ):  # adjust the dict notation for the use in the pipeline -> if dict only contains models and keys....
            self.model_with_info = {key: {"model": value} for key, value in models.items()}
        else:
            self.model_with_info = models
        pass


class data_ext_val:

    def __init__(self, X, y, ohe, columngroups_df) -> None:
        self.X_val = X
        self.y_val = y
        self.y_val_orig = y.copy()
        self.columngroups_df = columngroups_df
        self.X_ohe_df = pd.DataFrame(ohe.transform(X), columns=ohe.get_feature_names_out().tolist())
        pass


###################


class Pipeline:
    def __init__(
        self,
        project_vars: dict = {},
        ext_val_obj=None,
        dataframes={}, # you can pass y_inner, y_outer (y_test), X_inner, X_outer (X_test) as a dict.
        external_validation_mode=False # needed for inheritance in external validation class
    ):
        """Init the Pipeline; load and preprocess the data specified in the user_input
        - one hot encode the data
        - adjust the source mapper to the ohe encoded dataframe
        - load all data into the data instance of the pipeline [Pipeline.data]
        - folder and cohort are only needed if external validation is called

        Args:
            project_vars (dict): project vars, that are defined in the dict, as row of interest, col of interest or DOI
        """
        # get the user_input:
        if ext_val_obj is None:
            self.user_input = pp.pp_user_input(project_vars)
            if external_validation_mode:
                return
            self.name = f"Pipeline_{getattr(self.user_input, 'DOI', None)}_{getattr(self.user_input, 'row_subset', None)}_{getattr(self.user_input, 'col_subset', None)}"
        else:
            vars(self).update(vars(ext_val_obj))
            self.data = None

        if ext_val_obj is None:
            ## load the needed data append table X to z_val
            self.data = pp.data(self.user_input,dataframes=dataframes)
            # add mappings for the data for instance the color that sould be used for a specific source_df
            self.mapper = pp.mapper(self)  # type: ignore
            # construct the One Hot Encoder for the Table X!!
            self.ohe = pp.fit_ohe(self.data, scale_data=self.user_input.method_scaling_remainders)  # type: ignore
            # compose a better readeble format for the X and adjust the source mapping for the encoded dataframe
            self.data.andjust_enc_X_and_map(self.ohe)

        if (
            "reduce_columns" not in self.user_input.__dict__
        ):  # Use all columns per modality by default if no reduction specified
            self.user_input.reduce_columns = None

        self.plot = {}
        self.model_type = "not_trained"
        self.pipeline_output_path = ""
        #self.calibration = {}

    def create_violin_plot(self,
                        cohort='val',
                        figsize=(6, 8),
                        color='#4A90E2',
                        alpha=0.7,
                        show_points=True,
                        point_size=6,
                        point_alpha=0.3,
                        title=None,
                        save_path=None,
                        dpi=300):
        """
        Create a clean, minimalistic violin plot showing prediction probability distribution.

        Args:
            cohort (str): Which data to use ('val', 'test', 'train')
            figsize (tuple): Figure size (width, height)
            color (str): Color for the violin plot
            alpha (float): Transparency for violin plot
            show_points (bool): Whether to overlay individual data points
            point_size (float): Size of individual points
            point_alpha (float): Transparency of individual points
            title (str): Custom title (if None, auto-generates)
            save_path (str): Path to save figure (if None, uses pipeline default)
            dpi (int): DPI for saved figure

        Returns:
            fig, ax: matplotlib figure and axis objects
        """

        # Get data based on cohort
        if cohort == 'val':
            if not hasattr(self.data, 'X_val'):
                raise ValueError("Validation data not available")
            X_data = self.data.X_val.copy()
        elif cohort == 'test':
            raise NotImplementedError("Test cohort not implemented")
        elif cohort == 'train':
            raise NotImplementedError("Train cohort not implemented")
        else:
            raise ValueError(f"Unknown cohort: {cohort}")

        # Generate predictions
        if not hasattr(self, 'master_RFC'):
            raise ValueError("Model must be trained first")

        predictions = self.master_RFC.predict_proba(self.ohe.transform(X_data))
        if predictions.ndim > 1:
            predictions = predictions[:, 1]  # Take positive class probability

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create violin plot
        violin_parts = ax.violinplot(
            [predictions],
            positions=[0],
            widths=0.6,
            showmeans=False,
            showmedians=True,
            showextrema=False
        )

        # Style the violin
        violin_parts['bodies'][0].set_facecolor(color)
        violin_parts['bodies'][0].set_alpha(alpha)
        violin_parts['bodies'][0].set_edgecolor('white')
        violin_parts['bodies'][0].set_linewidth(2)

        # Style median line
        violin_parts['cmedians'].set_colors('white')
        violin_parts['cmedians'].set_linewidth(3)

        # Add individual points if requested
        if show_points:
            # Add small random jitter to x-position
            x_jitter = np.random.normal(0, 0.02, len(predictions))
            ax.scatter(x_jitter, predictions,
                    s=point_size, alpha=point_alpha,
                    color=color, edgecolors='white', linewidth=0.3)

        # Customize axes
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_ylabel('Predicted Probability', fontsize=14, fontweight='bold')

        # Add sample size annotation
        ax.text(0, -0.08, f'n = {len(predictions)}',
            ha='center', va='top', transform=ax.transData,
            fontsize=12, color='gray')

        # Clean up plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

        # Set title
        if title is None:
            title = f'Prediction Distribution\n{self.user_input.col_subset} ({cohort.capitalize()})'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        # Tight layout
        plt.tight_layout()

        # Save if requested
        if save_path is None and hasattr(self.user_input, 'fig_path'):
            save_path = os.path.join(self.user_input.fig_path,
                                    f'violin_plot_{self.name}_{cohort}.svg')

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
            print(f"Violin plot saved to: {save_path}")

        # Store in pipeline object
        if not hasattr(self, 'plots'):
            self.plots = {}

        self.plots['violin'] = {
            'figure': fig,
            'axis': ax,
            'cohort': cohort,
            'color': color,
            'parameters': {
                'figsize': figsize,
                'alpha': alpha,
                'show_points': show_points,
                'point_size': point_size,
                'point_alpha': point_alpha
            }
        }

        return fig, ax



    def plot_correlations(self, subsettings_startswith=[None], subsettings_isin=[], print_columns=False):
        """plot the correlation matrix; uses the data in the X_ohe_df for the ploting

        Args:
            subsettings_startswith (list, optional): _description_. Defaults to [None] -> with None the isin can be used.
            subsettings_isin (list, optional): _description_. Defaults to [].
            print_columns (bool, optional): _description_. Defaults to False -> if set to True the fuction will return a list of the columns of X_ohe_df and not the figure! .

        Returns:
            _type_: _description_
        """
        return pp.correlation_matrix_plot(
            self.data,
            subsets_startwith=subsettings_startswith,
            subset_isin=subsettings_isin,
            print_columns=print_columns,
        )

    def training(self, model_type: str, split_group_on="split_int"):
        """_summary_

        Args:
            model_type (str): RFC, GBC, neuron, ensemble [neuron,rfc]
            cv_method (_type_, optional): grouped, stratified
            split_group_on: on what var. to split the groups on if grouped == cv_method
        """
        hyperparams = self.user_input.hyperparameters[model_type] #Loads the hyperparams e.g. specific for "Random Forest Classifier", indexed by "RFC" in user_input.yaml
        fixed_params = hyperparams["params_fixed"] #Loads the fixed params e.g. specific for "Random Forest Classifier", indexed by "RFC" in user_input.yaml
        grid_params = hyperparams["params_grid"]   #Loads the grid params, with different options that will be assessed in gridsearch
        pipeline_params = hyperparams["params_pipeline"]

        self.model_type = model_type
        self.trained_model = train_test.trained_model()  # init the class
        self.trained_model.five_fold_cross_train(
            self_pip=self,
            estimator=models.get_estimator(model_type, fixed_params),
            hyperparams=hyperparams,
            grouped_split_on=split_group_on
        ) # type: ignore
        self.name = self.name + "_" + model_type
        self.model_type = model_type
        self.pipeline_output_path = self.user_input.model_path + f"/Pipelines/{model_type}"

    def time_dep_training_eval_tube(self, model_type: str, cv_method="grouped", split_group_on="split_int"):
        """Add a time dependent readout to the case definition during the training and the evaluation. By firth adding the time dependent "status" and "status_cancerreg"(if avail.) orig dataframes, so far only works if y_val is available, feel free to add some alternative

        Args:
            model_type (str): _description_
            cv_method (str, optional): _description_. Defaults to "grouped".
            split_group_on (str, optional): _description_. Defaults to "split_int".
        """
        import copy
        start_msg="""

        Starting the training with a time dependent readout. To save the time dependent readout in the evaluation, the col_subset and row_subset will be adjusted, athought the original row and column subset will be used! please be aware of this!\n"""

        print(start_msg,f'\nThe options that are evaluated as readout timepoints are: {self.user_input.time_of_readout}')
        self.model_type = model_type

        orig_row_subset = getattr(self.user_input, 'row_subset', None)
        orig_col_subset = getattr(self.user_input, 'col_subset', None)
        time_dependent_training = {}

        # adjust the y_orig and y_val
        for df in [self.data.y_orig,self.data.y_val_orig]:
            for stat in [self.user_input.target, self.user_input.target_to_validate_on]:
                for ro in self.user_input.time_of_readout:
                    df[f'{stat}_{ro}']= ((df[stat]==1)& (df['difftime']*-1<=ro)).astype(int)

        self_copy=copy.deepcopy(self)

        # loop over the timepoints and train the model duplicate the pipline, del the data class at the end to avoid memory issues
        for readout in self.user_input.time_of_readout:
            # adjust the metadata for the time_setting
            print(f"\nStarting to train on readout timepoint: {readout}")
            copy_pl = copy.deepcopy(self_copy)
            copy_pl.user_input.row_subset = f"{orig_row_subset}_{readout}"
            copy_pl.data.y=pd.DataFrame(copy_pl.data.y_orig[f'{self.user_input.target}_{readout}'])
            copy_pl.data.y_val=pd.DataFrame(copy_pl.data.y_val_orig[f'{self.user_input.target_to_validate_on}_{readout}'])
            copy_pl.name = copy_pl.name + "_" + model_type + f"_t{readout}"
            copy_pl.user_input.target = f'{self.user_input.target_to_validate_on}_{readout}'
            copy_pl.user_input.target_to_validate_on = f'{self.user_input.target_to_validate_on}_{readout}'


            # lets train
            copy_pl.trained_model = train_test.trained_model()  # init the class
            copy_pl.trained_model.five_fold_cross_train(
                self_pip=copy_pl,
                estimator=models.get_estimator(model_type),
                type_cv=cv_method,
                grouped_split_on=split_group_on,
                )
            copy_pl.pipeline_output_path = copy_pl.user_input.model_path + f"/Pipelines/{model_type}"
            print(f"\nFinished training on readout timepoint: {self.user_input.target_to_validate_on}_{readout}")

            # lets evaluate
            copy_pl.build_master_RFC()
            copy_pl.master_RFC.get_best_params()
            copy_pl.evaluation()

            copy_pl.data = None
            time_dependent_training[f"{self.user_input.target_to_validate_on}_{readout}"]=copy_pl
        self.time_dep_train=time_dependent_training
        return time_dependent_training

    def build_master_RFC(self):
        self.master_RFC = models.master_model_RFC(self.trained_model.model_with_info, ohe=self.ohe)

    def feature_imp_barplot(self, n_features=50, func_for_aggregation=np.mean, fontsize=16, short_names=True, bar_height_factor=17, fig_width=7, borderpad=2, linewidth=1.5):
        ### We should move this to plot.py @Paul (dixit Jan) ###


        # Collect all features you don't want in the feature plot
        self.data.andjust_enc_X_and_map(self.ohe)
        features_to_exclude = self.data.X_ohe_map[self.data.X_ohe_map["name_print"] == "split_int"].index.tolist()
        self.data.X_ohe_map.loc[self.data.X_ohe_map['name_print'] == 'Gamma glutamyltransferase', 'name_print'] = 'Gamma GT'


        fig, ax, plotX = plot.feature_imp_barplot(
            model=self.master_RFC,
            small_plot=True,
            func_for_aggregation=func_for_aggregation,
            n_features=n_features,
            features_to_exclude=features_to_exclude,
            color_dict=self.mapper.color_groups,
            X_ohe_map=self.data.X_ohe_map,
            self=self,
            fontsize=fontsize,
            short_names=short_names,
            bar_height_factor=bar_height_factor,
            fig_width=fig_width,
            borderpad=borderpad,
            linewidth=linewidth
        )

        try:
            self.eval.feature_imp.update({"PlotX": plotX})
        except:
            print("Could not append PlotX, please run evaluation before this plot...")

        # Clean feature names by removing prefixes
        clean_feature_names = plotX.index.str.replace(r"^(remainder__|one_hot_encoder__|scaler__)", "", regex=True)

        # Create a DataFrame with all features and their mean importance
        feature_imp_df = pd.DataFrame(
            {"Feature": clean_feature_names, "Mean Importance": plotX["feature_imp"], "Source": plotX["source"]}
        ).sort_values("Mean Importance", ascending=False)

        # Create a summary DataFrame
        summary_df = (
            feature_imp_df.groupby("Source")["Mean Importance"].sum().sort_values(ascending=False).reset_index()
        )

        # Export to Excel
        excel_path = os.path.join(
            self.user_input.fig_path,
            f"Feature_Importance_{self.user_input.col_subset}_{self.user_input.row_subset}.xlsx",
        )

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # Write Feature Importance sheet
            feature_imp_df.to_excel(writer, sheet_name="Feature Importance", index=False, float_format="%.10f")

            # Write Summary by Source sheet
            summary_df.to_excel(writer, sheet_name="Summary by Source", index=False, float_format="%.10f")

            # Access the workbook and the worksheets
            workbook = writer.book
            feature_sheet = workbook["Feature Importance"]
            summary_sheet = workbook["Summary by Source"]

            # Set column widths
            for sheet in [feature_sheet, summary_sheet]:
                for column in sheet.columns:
                    max_length = 0
                    column = [cell for cell in column]
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(cell.value)
                        except:
                            pass
                    adjusted_width = max_length + 2
                    sheet.column_dimensions[column[0].column_letter].width = adjusted_width

        print(f"Feature importance data exported to: {excel_path}")

        return fig, ax, plotX

    def shap_analysis(self, sample_size=None, max_display=20, fig_size=(12, 8)):

        ### We should move this to plot.py @Paul (dixit Jan) ###

        """
        Perform SHAP analysis on the best performing estimator from the ensemble.

        Args:
        sample_size (int, optional): Number of samples to use for SHAP analysis. If None, use all samples.
        max_display (int): Maximum number of features to display in the SHAP summary plot.

        Returns:
        matplotlib.figure.Figure: Figure object containing the SHAP summary plot.
        """

        # Find the best performing estimator
        best_score = -np.inf
        best_estimator = None
        for index,grid_search in enumerate(self.master_RFC.models):
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_estimator = grid_search.best_estimator_
                best_index = index

        if best_estimator is None:
            raise ValueError("No best estimator found")

        if hasattr(self.data, "X_val"):
            X_val = pd.DataFrame(
                data=self.ohe.transform(self.data.X_val),
                columns=pd.Series(self.ohe.get_feature_names_out())
                .str.replace("remainder__", "")
                .str.replace("one_hot_encoder__", "")
                .str.replace("scaler__", ""),
            )
        else:
            print('No validation data available. using the testing data for the choosen estimator.')
            X_val = pd.DataFrame(
                data=self.ohe.transform(self.master_RFC.models_with_eids_of_datasets.get(f'model_{best_index}').get('X_train_inner')),
                columns=pd.Series(self.ohe.get_feature_names_out())
                .str.replace("remainder__", "")
                .str.replace("one_hot_encoder__", "")
                .str.replace("scaler__", ""),
            )


        # Sample the validation set if sample_size is specified
        if sample_size is not None and len(X_val) > sample_size:
            X_val_sample = X_val.sample(sample_size, random_state=42)
        else:
            X_val_sample = X_val

        # Create SHAP explainer for the best estimator
        explainer = shap.TreeExplainer(best_estimator)
        shap_values = explainer.shap_values(X_val_sample)
        explainer_general = explainer(X_val_sample)

        # If shap_values is a list (for multi-class), take the positive class (index 1)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Create a single figure
        fig, ax = plt.subplots(figsize=fig_size)

        # Create the SHAP summary plot
        shap.summary_plot(shap_values, X_val_sample, plot_type="dot", max_display=max_display, show=False)

        # Get the current axis
        ax = plt.gca()

        # Adjust plot layout
        plt.tight_layout()
        plt.subplots_adjust(left=0.3, top=0.95, bottom=0.05)

        # Add title
        plt.title(f"SHAP Summary Plot - Best Estimator ({self.user_input.col_subset})", fontsize=16, pad=20)

        # Adjust colorbar
        cbar = plt.gcf().axes[-1]  # The colorbar should be the last axes object
        if cbar is not None:
            cbar.tick_params(labelsize=12)
            cbar.set_ylabel("Feature value", fontsize=14)

        # Reduce space between lines and adjust y-axis limits
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min + 0.5, y_max - 0.5)

        # Save the figure if visual_export is True
        if self.user_input.visual_export:
            svg_path = os.path.join(
                self.user_input.fig_path,
                f"SHAP_{self.user_input.col_subset}_{self.user_input.row_subset}_{max_display}.svg",
            )
            plt.savefig(svg_path, format="svg", bbox_inches="tight", dpi=300)
            print(f"SHAP summary plot saved to: {svg_path}")
    def shap_plot_waterfalls(self, df,index=0):
            index=1
            df=pd.DataFrame(self.ohe.transform(pd.DataFrame(df.iloc[index,:]).T))
            df.columns=self.ohe.get_feature_names_out()
            def plot_waterfall(df, pl=self):
                idx=0
                X=df
                iterator=0
                for estimator in pl.trained_model.models:
                    estimator=estimator[0].best_estimator_

                    explainer = shap.explainers.TreeExplainer(estimator)
                    sv = explainer(X)
                    exp = shap.Explanation(sv.values[:,:,1],
                                    sv.base_values[:,1],
                                    data=X.values,
                                    feature_names=X.columns,)
                    shap.plots.waterfall(exp[idx],show=True)
                    iterator+=1
            plot_waterfall(df)


    #return plt.gcf() #, explainer, shap_values, explainer_general


    def summarize_data(self):
        """
        Print comprehensive summary statistics of the data in the model pipeline,
        including sex-stratified statistics for features and targets.
        Handles unique EID reporting and proper treatment of categorical variables.
        """
        def print_info(name, obj):
            print(f"\n{name}:")
            if obj is None:
                print("  None")
                return

            # Report on unique EIDs if available
            if isinstance(obj, pd.DataFrame) and 'eid' in obj.columns:
                total_rows = len(obj)
                unique_eids = obj['eid'].nunique()
                print(f"  Shape: {obj.shape} ({unique_eids} unique patients, {total_rows-unique_eids} duplicates)")
                if total_rows != unique_eids:
                    print(f"  WARNING: {total_rows-unique_eids} duplicate EIDs detected!")
            elif hasattr(obj, "shape"):
                print(f"  Shape: {obj.shape}")

            # Analyze data types more intelligently
            if isinstance(obj, pd.DataFrame):
                # Count dtypes by category
                numeric_cols = obj.select_dtypes(include=['number']).columns.tolist()
                categorical_cols = obj.select_dtypes(include=['category', 'object', 'bool']).columns.tolist()
                date_cols = obj.select_dtypes(include=['datetime']).columns.tolist()

                print(f"  Columns: {len(obj.columns)} total "
                      f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical, {len(date_cols)} datetime)")

                # Identify binary columns being treated as numeric
                binary_treated_as_numeric = []
                for col in numeric_cols:
                    if set(obj[col].dropna().unique()).issubset({0, 1}):
                        binary_treated_as_numeric.append(col)

                if binary_treated_as_numeric:
                    print(f"  Note: {len(binary_treated_as_numeric)} binary columns are treated as numeric:")
                    print(f"    {', '.join(binary_treated_as_numeric[:5])}" +
                          (f" and {len(binary_treated_as_numeric)-5} more" if len(binary_treated_as_numeric) > 5 else ""))

            # Show clean summary statistics with appropriate handling
            if isinstance(obj, pd.DataFrame) and len(obj) > 0:
                try:
                    # For numeric columns excluding binary
                    numeric_non_binary = [col for col in numeric_cols if col not in binary_treated_as_numeric]
                    if numeric_non_binary and len(numeric_non_binary) <= 8:  # Limit to avoid excessive output
                        print("\n  Summary of numeric features:")
                        desc = obj[numeric_non_binary].describe().transpose()
                        # Format to 2 decimal places
                        desc = desc.round(2)
                        print(desc)

                    # For binary/categorical columns
                    categorical_all = categorical_cols + binary_treated_as_numeric
                    if categorical_all and len(categorical_all) <= 8:
                        print("\n  Summary of categorical features:")
                        for col in categorical_all:
                            value_counts = obj[col].value_counts(dropna=False)
                            print(f"    {col}: {dict(value_counts.head(3))}" +
                                  (f" and {len(value_counts)-3} more values" if len(value_counts) > 3 else ""))
                except Exception as e:
                    print(f"  Could not generate summary statistics: {e}")

        # Helper for target counts with unique EID handling - MODIFIED to report both status types
        def count_positives(y_data, x_data=None, prefix=""):
            if y_data is None:
                return

            try:
                # Handle different data structures with EID awareness
                if isinstance(y_data, pd.DataFrame):
                    # First check for duplicate eids if applicable
                    if 'eid' in y_data.columns:
                        total = len(y_data)
                        unique_total = y_data['eid'].nunique()
                        if total != unique_total:
                            print(f"{prefix}Note: {total} rows but only {unique_total} unique patients")

                    # Count positives for BOTH status and status_cancerreg
                    status_columns = ['status', 'status_cancerreg']

                    for col in status_columns:
                        if col in y_data.columns:
                            # Count on both row level and unique EID level if possible
                            count = y_data[col].sum()
                            total = len(y_data)

                            # If we have EIDs, also report unique patient counts
                            if 'eid' in y_data.columns:
                                unique_positives = y_data[y_data[col] == 1]['eid'].nunique()
                                unique_total = y_data['eid'].nunique()

                                # Label appropriately
                                label_suffix = " (cancer registry confirmed)" if col == 'status_cancerreg' else " (overall HCC)"
                                print(f"{prefix}Count of {self.user_input.DOI}{label_suffix}: {unique_positives}/{unique_total} unique patients ({unique_positives/unique_total:.2%})")
                            else:
                                label_suffix = " (cancer registry confirmed)" if col == 'status_cancerreg' else " (overall HCC)"
                                print(f"{prefix}Count of {self.user_input.DOI}{label_suffix}: {count}/{total} ({count/total:.2%})")

                    # Also check for the target column specified in user_input
                    target_col = getattr(self.user_input, 'target', None)
                    if target_col and target_col in y_data.columns and target_col not in status_columns:
                        count = y_data[target_col].sum()
                        total = len(y_data)
                        if 'eid' in y_data.columns:
                            unique_positives = y_data[y_data[target_col] == 1]['eid'].nunique()
                            unique_total = y_data['eid'].nunique()
                            print(f"{prefix}Count of {self.user_input.DOI} ({target_col}): {unique_positives}/{unique_total} unique patients ({unique_positives/unique_total:.2%})")
                        else:
                            print(f"{prefix}Count of {self.user_input.DOI} ({target_col}): {count}/{total} ({count/total:.2%})")

                elif isinstance(y_data, pd.Series):
                    count = y_data.sum()
                    total = len(y_data)
                    print(f"{prefix}Count of {self.user_input.DOI}: {count}/{total} ({count/total:.2%})")
                else:  # Assume numpy array
                    count = np.sum(y_data == 1)
                    total = len(y_data)
                    print(f"{prefix}Count of {self.user_input.DOI}: {count}/{total} ({count/total:.2%})")

                # If we have corresponding X data with EIDs but Y doesn't have them
                if x_data is not None and isinstance(x_data, pd.DataFrame) and 'eid' in x_data.columns:
                    if not (isinstance(y_data, pd.DataFrame) and 'eid' in y_data.columns):
                        print(f"{prefix}Note: X data has {x_data['eid'].nunique()} unique patients")

            except Exception as e:
                print(f"{prefix}Could not count positives: {e}")

        # Helper for sex-stratified analysis with unique EID awareness
        def analyze_by_sex(X_data, y_data, prefix=""):
            if X_data is None or y_data is None:
                return

            try:
                # Determine if SEX column exists and how it's formatted
                sex_col = None
                for col_name in ['SEX', 'Sex', 'sex', 'gender', 'Gender']:
                    if col_name in X_data.columns:
                        sex_col = col_name
                        break

                if sex_col is None:
                    print(f"{prefix}No sex/gender column found for stratification")
                    return

                # Get unique values and their labels
                unique_vals = X_data[sex_col].unique()

                # Check if SEX is encoded as 0/1 or with string labels
                sex_labels = {}
                if set(unique_vals).issubset({0, 1, '0', '1', 'Male', 'Female', 'M', 'F'}):
                    for val in unique_vals:
                        if val in [0, '0', 'Female', 'F']:
                            sex_labels[val] = "Female"
                        else:
                            sex_labels[val] = "Male"
                else:
                    # Use the values as is
                    sex_labels = {val: str(val) for val in unique_vals}

                print(f"\n{prefix}Sex-stratified analysis:")

                # Function to get target count by sex, with unique EID awareness
                def count_by_sex(sex_value, label):
                    # Get available target columns (both status and status_cancerreg)
                    target_columns = []
                    if isinstance(y_data, pd.DataFrame):
                        for col in ['status', 'status_cancerreg']:
                            if col in y_data.columns:
                                target_columns.append(col)

                        # Also add user-specified target if different
                        target_col = getattr(self.user_input, 'target', None)
                        if target_col and target_col in y_data.columns and target_col not in target_columns:
                            target_columns.append(target_col)

                    if not target_columns and isinstance(y_data, pd.DataFrame):
                        target_columns = [y_data.columns[0]]  # Fallback to first column
                    elif isinstance(y_data, (pd.Series, np.ndarray)):
                        target_columns = ['target']  # Generic name for non-DataFrame data

                    # Check if we can join by EID for more accurate statistics
                    has_eids = isinstance(X_data, pd.DataFrame) and isinstance(y_data, pd.DataFrame) and \
                              'eid' in X_data.columns and 'eid' in y_data.columns

                    for target_col in target_columns:
                        if isinstance(y_data, pd.DataFrame) and target_col in y_data.columns:
                            if has_eids:
                                # Merge data to ensure we're matching correctly
                                sex_data = X_data[['eid', sex_col]].merge(y_data[['eid', target_col]], on='eid')
                                sex_subset = sex_data[sex_data[sex_col] == sex_value]

                                # Count both total rows and unique patients
                                count = sex_subset[target_col].sum()
                                total = len(sex_subset)
                                unique_count = sex_subset[sex_subset[target_col] == 1]['eid'].nunique()
                                unique_total = sex_subset['eid'].nunique()

                                prevalence_pct = (unique_count/unique_total * 100) if unique_total > 0 else 0

                                # Label appropriately
                                label_suffix = " (cancer registry)" if target_col == 'status_cancerreg' else " (overall)" if target_col == 'status' else f" ({target_col})"
                                print(f"  {label} - {self.user_input.DOI}{label_suffix}: {unique_count}/{unique_total} unique patients ({prevalence_pct:.2f}%)")
                            else:
                                # Without EIDs, use indices to match
                                if isinstance(X_data, pd.DataFrame):
                                    sex_indices = X_data[sex_col] == sex_value
                                    sex_subset = y_data.loc[sex_indices]
                                    count = sex_subset[target_col].sum()
                                    total = len(sex_subset)
                                else:
                                    sex_subset = y_data[X_data[sex_col] == sex_value]
                                    count = sex_subset[target_col].sum()
                                    total = len(sex_subset)

                                prevalence_pct = (count/total * 100) if total > 0 else 0
                                label_suffix = " (cancer registry)" if target_col == 'status_cancerreg' else " (overall)" if target_col == 'status' else f" ({target_col})"
                                print(f"  {label} - {self.user_input.DOI}{label_suffix}: {count}/{total} cases ({prevalence_pct:.2f}%)")

                        elif isinstance(y_data, (pd.Series, np.ndarray)):
                            # If y is not a DataFrame, it's likely a Series or array
                            sex_subset = y_data[X_data[sex_col] == sex_value]
                            if isinstance(sex_subset, pd.Series):
                                count = sex_subset.sum()
                            else:
                                count = np.sum(sex_subset == 1)
                            total = len(sex_subset)

                            prevalence_pct = (count/total * 100) if total > 0 else 0
                            print(f"  {label} - {self.user_input.DOI}: {count}/{total} cases ({prevalence_pct:.2f}%)")

                        if total > 0 and count > 0:
                            print(f"    Prevalence ratio: 1:{total/count:.1f}")

                # Print counts for each sex category
                for sex_value, label in sex_labels.items():
                    count_by_sex(sex_value, label)

            except Exception as e:
                print(f"{prefix}Error in sex-stratified analysis: {e}")

        # Check for duplicate EIDs across all relevant dataframes
        def check_eid_consistency():
            relevant_dfs = []

            # Build a list of all dataframes with EIDs
            if hasattr(self.data, "X") and isinstance(self.data.X, pd.DataFrame) and 'eid' in self.data.X.columns:
                relevant_dfs.append(("X", self.data.X))

            if hasattr(self.data, "y") and isinstance(self.data.y, pd.DataFrame) and 'eid' in self.data.y.columns:
                relevant_dfs.append(("y", self.data.y))

            if hasattr(self.data, "X_val") and isinstance(self.data.X_val, pd.DataFrame) and 'eid' in self.data.X_val.columns:
                relevant_dfs.append(("X_val", self.data.X_val))

            if hasattr(self.data, "y_val") and isinstance(self.data.y_val, pd.DataFrame) and 'eid' in self.data.y_val.columns:
                relevant_dfs.append(("y_val", self.data.y_val))

            if hasattr(self.data, "z_val") and isinstance(self.data.z_val, pd.DataFrame) and 'eid' in self.data.z_val.columns:
                relevant_dfs.append(("z_val", self.data.z_val))

            if len(relevant_dfs) < 2:
                return  # Not enough dataframes to compare

            print("\nCHECKING EID CONSISTENCY ACROSS DATASETS:")

            # Check for overlaps and differences
            for i, (name_i, df_i) in enumerate(relevant_dfs):
                eids_i = set(df_i['eid'])
                for j, (name_j, df_j) in enumerate(relevant_dfs[i+1:], i+1):
                    eids_j = set(df_j['eid'])

                    # Calculate overlap and unique sets
                    overlap = eids_i.intersection(eids_j)
                    only_i = eids_i - eids_j
                    only_j = eids_j - eids_i

                    print(f"  {name_i} vs {name_j}:")
                    print(f"    Common EIDs: {len(overlap)}")
                    print(f"    Only in {name_i}: {len(only_i)}")
                    print(f"    Only in {name_j}: {len(only_j)}")

        # NEW: Overall data summary function
        def print_overall_summary():
            print("\nOVERALL DATA SUMMARY:")
            print("="*40)

            # Combine all available data for overall statistics
            all_data_sources = []

            # Training data
            X = getattr(self.data, "X", None)
            y = getattr(self.data, "y", None)
            if X is not None and y is not None:
                all_data_sources.append(("Training", X, y))

            # Validation data
            X_val = getattr(self.data, "X_val", None)
            y_val = getattr(self.data, "y_val", None)
            if X_val is not None and y_val is not None:
                all_data_sources.append(("Validation", X_val, y_val))

            # Additional validation info
            z_val = getattr(self.data, "z_val", None)
            if z_val is not None:
                all_data_sources.append(("Additional_validation", None, z_val))

            # Calculate overall statistics
            total_unique_patients = set()
            total_overall_hcc = set()
            total_cancerreg_hcc = set()
            total_rows = 0

            for source_name, X_data, y_data in all_data_sources:
                if y_data is not None and isinstance(y_data, pd.DataFrame) and 'eid' in y_data.columns:
                    # Get unique patients from this source
                    source_patients = set(y_data['eid'])
                    total_unique_patients.update(source_patients)
                    total_rows += len(y_data)

                    # Count HCC cases
                    if 'status' in y_data.columns:
                        hcc_patients = set(y_data[y_data['status'] == 1]['eid'])
                        total_overall_hcc.update(hcc_patients)

                    if 'status_cancerreg' in y_data.columns:
                        cancerreg_patients = set(y_data[y_data['status_cancerreg'] == 1]['eid'])
                        total_cancerreg_hcc.update(cancerreg_patients)

            # Print overall summary
            print(f"Total unique patients across all datasets: {len(total_unique_patients)}")
            print(f"Total rows across all datasets: {total_rows}")

            if total_overall_hcc:
                overall_prevalence = len(total_overall_hcc) / len(total_unique_patients) * 100
                print(f"Total {self.user_input.DOI} cases (overall): {len(total_overall_hcc)}/{len(total_unique_patients)} ({overall_prevalence:.2f}%)")

            if total_cancerreg_hcc:
                cancerreg_prevalence = len(total_cancerreg_hcc) / len(total_unique_patients) * 100
                print(f"Total {self.user_input.DOI} cases (cancer registry confirmed): {len(total_cancerreg_hcc)}/{len(total_unique_patients)} ({cancerreg_prevalence:.2f}%)")

            # Show overlap between overall and cancer registry if both exist
            if total_overall_hcc and total_cancerreg_hcc:
                overlap = total_overall_hcc.intersection(total_cancerreg_hcc)
                only_overall = total_overall_hcc - total_cancerreg_hcc
                print(f"HCC cases confirmed by cancer registry: {len(overlap)}/{len(total_overall_hcc)} ({len(overlap)/len(total_overall_hcc)*100:.1f}%)")
                print(f"HCC cases not confirmed by cancer registry: {len(only_overall)}")

        # Print basic info about the pipeline
        print(f"\n{'='*60}")
        print(f"DATA SUMMARY FOR MODEL: {getattr(self.user_input, 'col_subset', 'Unknown')}")
        print(f"COHORT: {getattr(self.user_input, 'row_subset', 'Unknown')}")
        print(f"DOI: {getattr(self.user_input, 'DOI', 'Unknown')}")
        print(f"{'='*60}")

        # NEW: Add overall summary at the beginning
        print_overall_summary()

        # Training data
        X = getattr(self.data, "X", None)
        y = getattr(self.data, "y", None)
        X_val = getattr(self.data, "X_val", None)
        y_val = getattr(self.data, "y_val", None)

        print("\nTRAINING DATA:")
        print_info("X (training features)", X)
        print_info("y (training target)", y)
        count_positives(y, X, "  ")
        analyze_by_sex(X, y, "  ")

        print("\nVALIDATION DATA:")
        print_info("X_val (validation features)", X_val)
        print_info("y_val (validation target)", y_val)
        count_positives(y_val, X_val, "  ")
        analyze_by_sex(X_val, y_val, "  ")

        # Check if we have z_val available (for additional info)
        z_val = getattr(self.data, "z_val", None)
        if z_val is not None:
            print("\nADDITIONAL VALIDATION INFO (z_val):")
            print_info("z_val", z_val)
            count_positives(z_val, prefix="  ")

        # Check EID consistency across datasets
        check_eid_consistency()

        # Print model info if available
        if hasattr(self, "model_type") and self.model_type != "not_trained":
            print(f"\nMODEL TYPE: {self.model_type}")

            if hasattr(self, "master_RFC") and hasattr(self.master_RFC, "models"):
                print(f"  Number of models in ensemble: {len(self.master_RFC.models)}")

                # Try to get some performance metrics if available
                if hasattr(self, "eval") and hasattr(self.eval, "val"):
                    try:
                        aucs = self.eval.val.get("aucs", [])
                        auprcs = self.eval.val.get("auprcs", [])
                        if aucs:
                            print(f"  Mean validation AUROC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
                        if auprcs:
                            print(f"  Mean validation AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
                    except Exception as e:
                        print(f"  Could not extract performance metrics: {e}")

        print(f"\n{'='*60}")


    def evaluation(self, only_val=False):
        self.eval = train_test.eval(self, only_val=only_val)
        self.eval.get_pv_test_train(self, )

    def roc_auc_test_train(self, fontsize=14, title=True, border_width=1, save_fig=False):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
        ax = ax.ravel()
        n_splits = self.user_input.hyperparameters[self.model_type]["params_pipeline"]["n_splits"]

        plot.plot_rocs(
            self.eval.train["tprs"].transpose(),
            self.eval.train["aucs"],
            n_splits,
            ax=ax[0],
            fontsize=fontsize,
            border_width=border_width,
        )
        plot.plot_rocs(
            self.eval.test["tprs"].transpose(),
            self.eval.test["aucs"],
            n_splits,
            ax=ax[1],
            fontsize=fontsize,
            border_width=border_width
        )
        plot.plot_rocs(
            self.eval.val["tprs"].transpose(),
            self.eval.val["aucs"],
            n_splits,
            ax=ax[2],
            fontsize=fontsize,
            border_width=border_width
        )
        for axi, tit in zip(ax, ["Training", "Testing", "Validation"]):
            axi.set_title(tit)

        if title:

            suffix = getattr(self.user_input, "pl_suffix", None)
            if suffix:
                fig.suptitle(f"Evaluation of {self.name} — {suffix}", fontsize=fontsize+6)
            else:
                fig.suptitle(f"Evaluation of {self.name}", fontsize=fontsize+6)

        if save_fig:
            suffix = getattr(self.user_input, "pl_suffix", None)
            if suffix:
                svg_path = os.path.join(self.user_input.fig_path, f"ROCs_Train_Test_Val_{self.user_input.col_subset}_{self.user_input.row_subset}_{suffix}.svg")
            else:
                svg_path = os.path.join(self.user_input.fig_path, f"ROCs_Train_Test_Val_{self.user_input.col_subset}_{self.user_input.row_subset}.svg")

            fig.savefig(svg_path, format="svg", bbox_inches="tight")

        plt.tight_layout()

        return fig, ax

    def evaluation_summary_independent(self, output_path=None):
        """
        Perform evaluation summary for threshold-independent metrics and save/update results in Excel.

        Args:
        output_path (str, optional): Path to save the Excel file. If None, uses a default path.

        Returns:
        pd.DataFrame: DataFrame containing the evaluation results.
        """

        # def round_columns(df, columns, precision):
        #     df[columns] = df[columns].map(lambda x: round(x, precision) if isinstance(x, (int, float)) else x)
        #     return df

        columns_precision = {"Mean": 3, "Std. Dev.": 3, "Fold 1": 3, "Fold 2": 3, "Fold 3": 3, "Fold 4": 3, "Fold 5": 3}

        # Ensure evaluation has been performed
        if not hasattr(self, "eval"):
            self.evaluation()

        aucs = self.eval.val["aucs"]
        auprcs = self.eval.val["auprcs"]
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        mean_auprc = np.mean(auprcs)
        std_auprc = np.std(auprcs)

        # Create evaluation tables for AUROC and AUPRC
        evaluation_results = pd.DataFrame(
            {   #add x2 to all metadata columns as AUROC and AUPRC will be reported separately
                "DOI": [self.user_input.DOI]*2,
                "Target": [self.user_input.target]*2,
                "Model": [self.user_input.col_subset]*2,
                "Estimator" : [self.model_type]*2,
                "Dataset": [self.user_input.row_subset] * 2,
                "Metric": ["AUROC", "AUPRC"],
                "Mean": [mean_auc, mean_auprc],
                "Std. Dev.": [std_auc, std_auprc],
                "Fold 1": [aucs[0], auprcs[0]],
                "Fold 2": [aucs[1], auprcs[1]],
                "Fold 3": [aucs[2], auprcs[2]],
                "Fold 4": [aucs[3], auprcs[3]],
                "Fold 5": [aucs[4], auprcs[4]],
            }
        )

        evaluation_results = evaluation_results.round(
            {col: prec for col, prec in columns_precision.items()}
        )

        # Store results in pipeline object
        if not hasattr(self, "evaluation_results"):
            self.evaluation_results = {}
        self.evaluation_results["independent"] = evaluation_results

        # Save to Excel, updating existing file if it exists
        if output_path is None:
            output_path = os.path.join(
                self.user_input.model_path,
                # "Pipelines",
                # self.model_type,
                # "combined_output",
                # "val",
                "all_evaluation_results.xlsx",
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Read existing Excel file if it exists
        if os.path.exists(output_path):
            with pd.ExcelFile(output_path) as xls:
                if "Independent metrics" in xls.sheet_names:
                    existing_data = pd.read_excel(xls, "Independent metrics")
                    # Remove rows where all 4 columns match the current values
                    # Fixed mask logic: use & instead of | for AND condition
                    mask = (
                        (existing_data["DOI"] == self.user_input.DOI) &
                        (existing_data["Target"] == self.user_input.target) &
                        (existing_data["Model"] == self.user_input.col_subset) &
                        (existing_data["Dataset"] == self.user_input.row_subset) &
                        (existing_data["Estimator"] == self.model_type)
                    )
                    existing_data = existing_data[~mask]
                    evaluation_results = pd.concat([existing_data, evaluation_results], ignore_index=True)

        # Write to Excel - handle existing vs new file
        if os.path.exists(output_path):
            # Read all existing sheets
            existing_sheets = {}
            with pd.ExcelFile(output_path) as xls:
                for sheet_name in xls.sheet_names:
                    if sheet_name != "Independent metrics":
                        existing_sheets[sheet_name] = pd.read_excel(xls, sheet_name)

            # Write all sheets back
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                evaluation_results.to_excel(writer, sheet_name="Independent metrics", index=False)
                for sheet_name, sheet_data in existing_sheets.items():
                    sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            # Create new file
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                evaluation_results.to_excel(writer, sheet_name="Independent metrics", index=False)

        print(f"Threshold-independent evaluation results saved to: {output_path}")
        return evaluation_results



    def evaluation_summary_threshold_dependent(self, output_path=None, thresholds=np.arange(0.6, 0.29, -0.01), beta=10):
        """
        Perform evaluation summary for threshold-dependent metrics and save/update results in Excel.
        """


        columns_precision = {
            "Precision": 3,
            "Recall": 2,
            "Accuracy": 2,
            "F1 Score": 3,
            f"F-beta Score (beta={beta})": 3,
            "Balanced Accuracy": 2,
            "PPV": 4,
            "NPV": 4,
            "NNS": 1,
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "TN": 0,
            "TP %": 3,
            "FN %": 3,
            "Youden Index": 3,
        }

        if thresholds is None:
            thresholds = np.arange(0.6, 0.29, -0.01)

        # Ensure evaluation has been performed
        if not hasattr(self, "eval"):
            self.evaluation()

        threshold_evaluation_results = pd.DataFrame()

        proba = self.master_RFC.predict_proba(self.ohe.transform(self.data.X_val))
        y_true = self.data.z_val["status_cancerreg"]

        for threshold in thresholds:
            if proba.ndim == 1:
                y_pred = (proba >= threshold).astype(int)
            else:
                y_pred = (proba[:, 1] >= threshold).astype(int)

            # Calculate metrics
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            f_beta = fbeta_score(y_true, y_pred, beta=beta)
            balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            # Calculate percentages
            tp_percentage = tp / (tp + fn) if (tp + fn) > 0 else 0
            fn_percentage = fn / (tp + fn) if (tp + fn) > 0 else 0

            # Calculate Youden Index
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            youden_index = recall + specificity - 1

            # Create evaluation table for the threshold
            evaluation_table_threshold = pd.DataFrame(
                {   "DOI": [self.user_input.DOI],
                    "Target": [self.user_input.target],
                    "Model": [self.user_input.col_subset],
                    "Dataset": [self.user_input.row_subset],
                    "Estimator" : [self.model_type],
                    "Threshold": [threshold],
                    "Precision": [precision],
                    "Recall": [recall],
                    "Accuracy": [accuracy],
                    "F1 Score": [f1],
                    f"F-beta Score (beta={beta})": [f_beta],
                    "Balanced Accuracy": [balanced_accuracy],
                    "PPV": [ppv],
                    "NPV": [npv],
                    "NNS": [1 / ppv if ppv > 0 else float("inf")],
                    "TP": [tp],
                    "FP": [fp],
                    "FN": [fn],
                    "TN": [tn],
                    "TP %": [tp_percentage],
                    "FN %": [fn_percentage],
                    "Youden Index": [youden_index],
                }
            )

            threshold_evaluation_results = pd.concat(
                [threshold_evaluation_results, evaluation_table_threshold], ignore_index=True
            )

        # Apply rounding to all columns based on the specified precision
        round_dict = {col: prec for col, prec in columns_precision.items()
                  if col in threshold_evaluation_results.columns}
        threshold_evaluation_results = threshold_evaluation_results.round(round_dict)

        # Store results in the Pipeline object
        if not hasattr(self, "evaluation_results"):
            self.evaluation_results = {}
        self.evaluation_results["threshold_dependent"] = threshold_evaluation_results

        # Save to Excel, updating existing file if it exists
        if output_path is None:
            output_path = os.path.join(
                self.user_input.model_path,
                #"Pipelines",
                # self.model_type,
                # "combined_output",
                # "val",
                "all_evaluation_results.xlsx",
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Read existing Excel file if it exists
        if os.path.exists(output_path):
            with pd.ExcelFile(output_path) as xls:
                if "Threshold metrics_all_thresholds" in xls.sheet_names:
                    existing_data = pd.read_excel(xls, "Threshold metrics_all_thresholds")
                    # Remove rows where all 5 columns match the current values
                    # Fixed mask logic: use & instead of | for AND condition
                    mask = (
                        (existing_data["DOI"] == self.user_input.DOI) &
                        (existing_data["Target"] == self.user_input.target) &
                        (existing_data["Model"] == self.user_input.col_subset) &
                        (existing_data["Dataset"] == self.user_input.row_subset) &
                        (existing_data["Estimator"] == self.model_type)
                    )
                    existing_data = existing_data[~mask]
                    threshold_evaluation_results = pd.concat(
                        [existing_data, threshold_evaluation_results], ignore_index=True
                    )

        # Sort the results by Model, Dataset, and Threshold
        threshold_evaluation_results = threshold_evaluation_results.sort_values(["Model", "Dataset", "Threshold"])

        # Write to Excel - handle existing vs new file
        if os.path.exists(output_path):
            # Read all existing sheets
            existing_sheets = {}
            with pd.ExcelFile(output_path) as xls:
                for sheet_name in xls.sheet_names:
                    if sheet_name != "Threshold metrics_all_thresholds":
                        existing_sheets[sheet_name] = pd.read_excel(xls, sheet_name)

            # Write all sheets back
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                threshold_evaluation_results.to_excel(writer, sheet_name="Threshold metrics_all_thresholds", index=False)
                for sheet_name, sheet_data in existing_sheets.items():
                    sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            # Create new file
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                threshold_evaluation_results.to_excel(writer, sheet_name="Threshold metrics_all_thresholds", index=False)

        print(f"Threshold-dependent evaluation results saved to: {output_path}")
        return threshold_evaluation_results


    def save_values_for_combined_plot(self, only_val=False, save_format="joblib"):
        """
        Save the prediction values of testing, training, and validation cohorts,
        along with their TPRS data, into joblib/csv/xlsx files for combined plotting.

        Args:
            only_val (bool, optional): If True, only save the validation cohort data.
            save_format (str, optional): Format to save the outputs ('joblib', 'csv', 'xlsx'). Defaults to 'joblib'.
        """
        if only_val:
            train, test, val = (
                None,
                None,
                self.eval.val["predicted_values"] if hasattr(self.eval, "val") else None,
            )
        else:
            train = self.eval.test_train_pred.get("train")
            test = self.eval.test_train_pred.get("test")
            val = self.eval.test_train_pred.get("val")

        # --- Validation ---
        if val is not None and hasattr(self.eval, "val") and "tprs" in self.eval.val:
            self.eval.save_performance_combination(
                pip_self=self,
                tprs=self.eval.val["tprs"],
                pred_values=val.get("y_pred") if isinstance(val, pd.DataFrame) else None,
                y_true=val.get("status") if isinstance(val, pd.DataFrame) else None,
                true_cancerreg=(
                    getattr(self.data.z_val, "status_cancerreg", None) if hasattr(self.data, "z_val") else None
                ),
                cohort="val",
                save_format=save_format,
            )
        else:
            print("Validation data not available. Skipping validation performance combination.")

        if not only_val:
            # --- Training ---
            if isinstance(train, pd.DataFrame) and "y_pred" in train and self.user_input.target in train:
                tprs_train = self.eval.train.get("tprs") if hasattr(self.eval, "train") else None
                self.eval.save_performance_combination(
                    pip_self=self,
                    tprs=tprs_train,
                    pred_values=train["y_pred"],
                    y_true=train[self.user_input.target],
                    cohort="train",
                    save_format=save_format,
                )
            else:
                print("Training data incomplete. Skipping training performance combination.")

            # --- Testing ---
            if isinstance(test, pd.DataFrame) and "y_pred" in test and self.user_input.target in test:
                tprs_test = self.eval.test.get("tprs") if hasattr(self.eval, "test") else None
                self.eval.save_performance_combination(
                    pip_self=self,
                    tprs=tprs_test,
                    pred_values=test["y_pred"],
                    y_true=test[self.user_input.target],
                    cohort="test",
                    save_format=save_format,
                )
            else:
                print("Testing data incomplete. Skipping testing performance combination.")

    def save_values_for_validation(self):
        if hasattr(self.data, "z_val") and hasattr(self.data.z_val, "status_cancerreg"):
            self.eval.save_performance_combination(
                self,
                self.eval.val["tprs"],
                self.master_RFC.predict_proba(self.ohe.transform(self.data.X_val)),
                self.data.y_val,
                self.data.z_val.status_cancerreg,
                cohort="val",
            )
        else:
            self.eval.save_performance_combination(
                self,
                self.eval.val["tprs"],
                self.master_RFC.predict_proba(self.ohe.transform(self.data.X_val)),
                self.data.y_val,
                cohort="val",
            )

    def validation(self):
        self.eval.get_pv_test_train(self)
        print("Performance on the validation dataset is saved to the eval class.")

    def external_validation(self, X_val, y_val):
        self.data = data_ext_val(X_val, y_val, ohe=self.ohe, columngroups_df=self.columngroups_df)

    def save_Pipeline(self, compress=('zlib', 3)):
        path = self.user_input.model_path + f"/Pipelines/{self.model_type}/"
        os.makedirs(path, exist_ok=True)

        # Add suffix if specified
        name_with_suffix = self.name
        if hasattr(self.user_input, 'pl_suffix') and self.user_input.pl_suffix is not None:
            name_with_suffix = f"{self.name}_{self.user_input.pl_suffix}"

        model_save_to = path + name_with_suffix + ".joblib"
        dump(self, model_save_to, compress=compress)
        print("Pipeline saved to:\n", model_save_to)


    def save_Pipeline_and_comb_outputs(self, only_val=False, compress=('zlib', 3)):
        self.save_values_for_combined_plot(only_val=only_val)
        path = self.user_input.model_path + f"/Pipelines/{self.model_type}/"
        os.makedirs(path, exist_ok=True)

        # Add suffix if specified
        name_with_suffix = self.name
        if hasattr(self.user_input, 'pl_suffix') and self.user_input.pl_suffix is not None:
            name_with_suffix = f"{self.name}_{self.user_input.pl_suffix}"

        model_save_to = path + name_with_suffix + ".joblib"
        print("Type of self:", type(self))
        print("Defined in module:", self.__class__.__module__)
        dump(self, model_save_to, compress=compress)
        print("Pipeline saved to:\n", model_save_to)

    def get_permu_feature_imp(self,cohort="test"):
        for i,a in self.trained_model.model_with_info.items():
            if cohort != 'val':
                X= self.ohe.transform(a.get(f"X_{cohort}_inner"))
                y= a.get(f"y_{cohort}_inner")
            else:
                X=self.ohe.transform(self.data.X_val)
                y=self.data.y_val
            a.get('model').best_estimator_.permutate_feature_imp(X=X,y=y)

    def run_ablation_study(
            self,
            only_pre_ablation: bool = False,
            to_n_features: int = 1,
            n_features_elim_tail_at_start: float = 0.8,
            override_existing: bool = False,
            plot_all: bool = True,
            use_median_hyperparams: bool = True,
            allowed_features=None,
            allowed_features_mode="raw"
        ):
        if hasattr(self, 'ablation_study') and override_existing is False:
            print("Ablation study already exists. Skipping creation or set override_existing to True.")
            return

        n_features=len(self.master_RFC.feature_importances_())
        orig_hyperparams = copy.copy(self.user_input.hyperparameters)

        def get_type(k,orig_hyperparams=orig_hyperparams):
            if k in orig_hyperparams[self.model_type]["params_grid"]:
                return type(orig_hyperparams[self.model_type]["params_grid"][k][0])
            else:
                return type(orig_hyperparams[self.model_type]["params_pipeline"][k])

        if use_median_hyperparams:
            _=self.master_RFC.get_best_params()
            self.user_input.hyperparameters[self.model_type]["params_grid"]={k:[get_type(k)(v)] for k,v in zip(_.columns,np.median(_,axis=0))}
            print("Using median hyperparameters for ablation study.", self.user_input.hyperparameters[self.model_type]["params_grid"])
            self.user_input.hyperparameters[self.model_type]['params_ablation'] = self.user_input.hyperparameters[self.model_type]
        else:
            self.user_input.hyperparameters[self.model_type]['params_ablation'] = self.user_input.hyperparameters[self.model_type]
            print("Using original hyperparameters for ablation study.", self.ablation_study.hyperparams_used)
        if only_pre_ablation:
            dimension_of_small_model= 5
            n_to_reduce=n_features-dimension_of_small_model
            pre_ab=ablation(self, to_n_features=dimension_of_small_model,n_features_elim_tail_at_start=n_to_reduce)
            pre_ab.run_ablation()
            return pre_ab.get_scorer_df()

        self.ablation_study = ablation(self, to_n_features=to_n_features, step_size=1, n_features_elim_tail_at_start=n_features_elim_tail_at_start,
                                        allowed_features=allowed_features, allowed_features_mode=allowed_features_mode)
        self.ablation_study.run_ablation()
        self.ablation_study.scorer_df=self.ablation_study.get_scorer_df()

        if plot_all:
            self.ablation_study.plot_feature_rank_lineplot(export_path=self.user_input.fig_path+ f"/ablation_feature_rank_lineplot_{self.name}.svg")

            plot.plot_ablation_dual_visual(
                df=self.ablation_study.scorer_df,
                x_col='len_included',
                y1_col='mean_auc',
                y2_col= 'mean_auprc',
                y1_label="AUROC",
                y2_label="AUPRC",
                x_label="Number of Included Features",
                y1_color="blue",
                y2_color="red",
                title=None,
                y1_ylim=(0.5, 1),
                y2_ylim=(0, 0.1),
                x_lim=(20,0),  # Custom x-axis limits
                x_reverse=False,  # Don't reverse x-axis
                save_path=self.user_input.fig_path+ f"/ablation_study_{self.name}",
                save_format="svg"
            )


        self.user_input.hyperparameters = orig_hyperparams  # Restore original hyperparameters


    def add_calibration(self, method='sigmoid', cv=5, X_holdout=None, y_holdout=None):
        """
        Add and calibrate the model using the holdout/validation set.
        """
        if not hasattr(self, 'master_RFC'):
            raise ValueError(
                "Pipeline must be trained first. "
                "Run training() and build_master_RFC() before adding calibration."
            )

        # CHANGE THIS LINE - use hasattr to handle old pipelines
        if not hasattr(self, 'calibration') or self.calibration is None:
            self.calibration = CalibrationLayer(self)
            print("✓ Calibration layer created")

        # Calibrate on holdout data (your X_val)
        self.calibration.calibrate_on_holdout(
            method=method,
            cv=cv,
            X_holdout=X_holdout,
            y_holdout=y_holdout
        )

        return self.calibration


    def save_to(self, path: str = ".",compress=('zlib', 3)):
        save_dir = os.path.join(self.user_input.path, "Models", "Validation_Objects")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{self.name}_external_val.joblib")
        dump(self, file_path,compress=compress)

        # Print confirmation message
        print(f"External validation object has been saved to: {os.path.abspath(file_path)}")




def plot_KM_improved(
    pl,
    thresholds_limits: List[float] = [0.4, 0.5],
    color_dict: Dict[str, str] = {"low": "green", "medium": "orange", "high": "red"},
    cohort: str = "val",
    target: str = "status_cancerreg",
    time_variable: str = "date_of_diag",
    risk_score_var: str = "y_pred",
    figsize: Tuple[int, int] = (12, 8),
    show_stats: bool = True
) -> KaplanMeierFitter:
    """
    Create Kaplan-Meier survival plots with improved functionality.

    Parameters:
    - pl: Pipeline object containing data and models
    - thresholds_limits: Risk score thresholds for grouping [low_threshold, high_threshold]
    - color_dict: Colors for each risk group
    - cohort: Dataset to use ('val', 'test', 'train')
    - target: Target variable name for events
    - time_variable: Time variable name
    - risk_score_var: Risk score variable name
    - figsize: Figure size tuple
    - show_stats: Whether to show statistical tests

    Returns:
    - KaplanMeierFitter object with fitted data
    """

    def _get_risk_groups(pred_prob: np.array, thresholds: List[float]) -> List[str]:
        """Assign risk groups based on prediction probabilities."""
        conditions = [
            pred_prob < thresholds[0],
            (pred_prob >= thresholds[0]) & (pred_prob < thresholds[1]),
            pred_prob >= thresholds[1]
        ]
        choices = ['low', 'medium', 'high']
        return np.select(conditions, choices, default='unknown')

    def _prepare_survival_data(pl, cohort: str, time_variable: str, target: str, risk_score_var: str) -> pd.DataFrame:
        """Prepare and clean survival analysis data."""
        # Get data based on cohort
        try:
            z_val = pl.data.z_val.copy()
        except AttributeError:
            print(f"Warning: Could not find z_{cohort}, using df_y_orig")
            z_val = pl.data.df_y_orig.copy()

        # Handle missing diagnosis dates
        time_censoring = pd.Timestamp(year=2024, month=1, day=1)
        z_val.loc[z_val[time_variable].isna(), time_variable] = time_censoring

        # Calculate time to event in days, then convert to months
        timedelta = pd.to_datetime(z_val[time_variable]) - pd.to_datetime(z_val["Date of assessment"])
        z_val["time_to_event_months"] = timedelta.dt.days / 365.25  # Average days per month

        # Generate risk scores if not present
        if risk_score_var not in z_val.columns:
            print(f"Generating risk scores using {risk_score_var}")
            try:
                z_val[risk_score_var] = pl.master_RFC.predict_proba(pl.ohe.transform(pl.data.X_val)).values
            except Exception as e:
                print(f"Error generating risk scores: {e}")
                print("Please ensure risk scores are pre-calculated or check model/data compatibility")
                raise

        return z_val

    def _plot_km_curves(z_val: pd.DataFrame, target: str, color_dict: Dict, figsize: Tuple, show_stats: bool, thresholds_limits: List[float]):
        """Create the Kaplan-Meier plots."""
        fig, ax = plt.subplots(figsize=figsize)

        kmf = KaplanMeierFitter()
        groups = z_val['risk_group'].unique()

        # Fit and plot each group
        for group in ['low', 'medium', 'high']:  # Ensure consistent order
            if group in groups:
                group_data = z_val[z_val['risk_group'] == group]
                if group == 'low':
                    label = f'Low risk (< {thresholds_limits[0]:.2f}) (n={len(group_data)})'
                elif group == 'medium':
                    label = f'Medium risk ({thresholds_limits[0]:.2f}–{thresholds_limits[1]:.2f}) (n={len(group_data)})'
                elif group == 'high':
                    label = f'High risk (≥ {thresholds_limits[1]:.2f}) (n={len(group_data)})'
                else:
                    label = f'{group.capitalize()} risk (n={len(group_data)})'

                kmf.fit(
                    durations=group_data['time_to_event_months'],
                    event_observed=group_data[target],
                    label=label
                )
                kmf.plot_survival_function(
                    ax=ax,
                    color=color_dict.get(group, 'gray'),
                    alpha=0.7,
                    linewidth=2
                )

        # Customize plot
        ax.set_xlabel('Time (years)', fontsize=24)
        ax.set_ylabel('Survival Probability', fontsize=24)
        ax.set_title(f'Kaplan-Meier Survival Curves - {cohort.capitalize()} Cohort', fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=16, loc="lower left")

        # Add statistics if requested
        if show_stats and len(groups) >= 2:
            _add_statistical_tests(z_val, target, ax)

        plt.tight_layout()
        plt.show()

        return kmf

    def _add_statistical_tests(z_val: pd.DataFrame, target: str, ax):
        """Add log-rank test results to the plot."""
        try:
            # Compare high vs low risk groups
            high_risk = z_val[z_val['risk_group'] == 'high']
            low_risk = z_val[z_val['risk_group'] == 'low']

            if len(high_risk) > 0 and len(low_risk) > 0:
                results = logrank_test(
                    high_risk['time_to_event_months'], low_risk['time_to_event_months'],
                    high_risk[target], low_risk[target]
                )

                p_value = results.p_value
                ax.text(0.02, 0.19, f'Log-rank p-value: {p_value:.4f}',
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        except Exception as e:
            print(f"Could not perform statistical test: {e}")

    # Main execution
    print(f"Creating Kaplan-Meier plot for {cohort} cohort")
    print(f"Thresholds: {thresholds_limits}")

    # Prepare data
    survival_data = _prepare_survival_data(pl, cohort, time_variable, target, risk_score_var)

    # Assign risk groups
    survival_data['risk_group'] = _get_risk_groups(survival_data[risk_score_var], thresholds_limits)

    # Print group distribution
    group_counts = survival_data['risk_group'].value_counts()
    print("Risk group distribution:")
    for group, count in group_counts.items():
        print(f"  {group}: {count} patients")

    # Create plots
    kmf_result = _plot_km_curves(survival_data, target, color_dict, figsize, show_stats, thresholds_limits)

    return kmf_result


# Example usage:
# estimator = plot_KM_improved(
#     pl,
#     thresholds_limits=[0.4, 0.6],
#     cohort="val",
#     figsize=(14, 8),
#     show_stats=True
# )

# class export_ext_val:
#     def __init__(self, pl) -> None:
#         self.user_input = pl.user_input
#         self.master_RFC = pl.master_RFC
#         self.list_estimators = [i.best_estimator_ for i in pl.master_RFC.models]
#         self.name = pl.name
#         self.ohe = pl.ohe
#         self.mapper = pl.mapper
#         self.columngroups_df = pl.data.columngroups_df
#         self.plots = pl.plots
#         self.pipeline_output_path = "."  # setzt den export path auf das dir in dem wir uns befinden !!!! adjust if needed on external server to the folder you want
#         ## export the trained_model class partly:
#         for key, value_orig in pl.trained_model.model_with_info.items():
#             value_new = {"model": value_orig.get("model")}
#             value_orig.clear()
#             value_orig.update(value_new)
#         self.trained_model = pl.trained_model

#     def save_to(self, path: str = ".",compress=('zlib', 3)):
#         save_dir = os.path.join(self.user_input.path, "Models", "Validation_Objects")
#         os.makedirs(save_dir, exist_ok=True)
#         file_path = os.path.join(save_dir, f"{self.name}_external_val.joblib")
#         dump(self, file_path,compress=compress)

#         # Print confirmation message
#         print(f"External validation object has been saved to: {os.path.abspath(file_path)}")

def export_ext_val(pl):
    """Wrapper function that creates ExportExtVal instance"""
    return ExportExtVal(pl)

# Function to set up modules before loading
def setup_module_mappings():
    """Call this before loading any pipeline objects"""
    if 'modeling_pipeline' not in sys.modules:
        modeling_pipeline = types.ModuleType('modeling_pipeline')
        sys.modules['modeling_pipeline'] = modeling_pipeline

        submodules = [
            'ablation', 'wrapper_roc_analysis', 'wrapper_violins_prcs',
            'export_tables', 'pp', 'plot', 'training', 'training.models',
            'training.train_test', 'pipeline'
        ]

        for submodule_name in submodules:
            dummy_module = types.ModuleType(f'modeling_pipeline.{submodule_name}')

            if '.' in submodule_name:
                parts = submodule_name.split('.')
                current = modeling_pipeline
                for i, part in enumerate(parts[:-1]):
                    if not hasattr(current, part):
                        setattr(current, part, types.ModuleType(f'modeling_pipeline.{".".join(parts[:i+1])}'))
                    current = getattr(current, part)
                setattr(current, parts[-1], dummy_module)
            else:
                setattr(modeling_pipeline, submodule_name, dummy_module)

            sys.modules[f'modeling_pipeline.{submodule_name}'] = dummy_module

        # Add the export class to the dummy pipeline module
        if hasattr(modeling_pipeline, 'pipeline'):
            modeling_pipeline.pipeline.export_ext_val = ExportExtVal

def iterate_over_pipelines(path, func=None):

    pipeline_path= path #Define the path to your pipeline folder, where all the joblib files are stored
    #get all joblib pipeline fieles from the pipeline_folder
    root_folder = Path(pipeline_path)
    joblib_files = list(root_folder.rglob('*.joblib'))

    # Convert to string paths if needed
    joblib_file_paths = [str(p) for p in joblib_files if p.is_file() and 'Pipeline' in str(p.name)]

    with open("../project_template/user_input.yaml") as file:
        user_input = yaml.load(file, Loader=yaml.FullLoader)


    #Loop over all pipeline objects
    for _path in joblib_file_paths:
        pl=load(_path)
        print(pl.name)
        #pl.whatever_you_want_to_do_with_the_pipeline_object()
        diff_dict={k:v for k,v in user_input.items() if k not in vars(pl.user_input)}
        for k,v in diff_dict.items():
            print(f"{k}: {v} is not in the pipeline object, but in the user input file")
            setattr(pl.user_input, k, v)

        if func is not None:
            try:
                func(pl)
            except Exception as e:
                print(f"Error applying function to pipeline {pl.name}: {e}")

        pl.save_Pipeline()



class fix_trained_model:
    '''
    Hack from All Of Us Instance to fix trained_model class when loading old pipeline objects, as somehow in AOU trained_model class was a dict instead of .-callable class object'''
    #TODO: Build a cleaner version of this fix if possible
    def __init__(self, trained_model_self):
        for k, v in trained_model_self.items():
            setattr(self, k, v)


