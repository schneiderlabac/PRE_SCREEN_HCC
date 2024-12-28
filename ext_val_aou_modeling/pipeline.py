import numpy as np
from joblib import dump, load
from pp import *
import pp as pp
import plot as plot
import training.models as models
import training.train_test as train_test
import shap

# TODO:
# - make external validation more elegant


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
    print('Attributes:\n')
    [print(func) for func in dir(obj) if not callable(getattr(obj, func)) and not func.startswith("__")]

    [print(func+'()') for func in dir(obj) if callable(getattr(obj, func)) and not func.startswith("__")]


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
        self.y_val_orig=y.copy()
        self.columngroups_df = columngroups_df
        self.X_ohe_df = pd.DataFrame(ohe.transform(X), columns=ohe.get_feature_names_out().tolist())
        pass


###################


class Pipeline:
    def __init__(
        self,
        project_vars: dict = {},
        ext_val_obj=None,
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
            self.name = f"Pipeline_{self.user_input.DOI}_{self.user_input.row_subset}_{self.user_input.col_subset}"
        else:
            vars(self).update(vars(ext_val_obj))
            self.data = None

        if ext_val_obj is None:
            ## load the needed data append table X to z_val
            self.data = pp.data(self.user_input)
            # add mappings for the data for instance the color that sould be used for a specific source_df
            self.mapper = pp.mapper(self)  # type: ignore
            # construct the One Hot Encoder for the Table X!!
            self.ohe = pp.fit_ohe(self.data)
            # compose a better readeble format for the X and adjust the source mapping for the encoded dataframe
            self.data.andjust_enc_X_and_map(self.ohe)

        if (
            "reduce_columns" not in self.user_input.__dict__
        ):  # Use all columns per modality by default if no reduction specified
            self.user_input.reduce_columns = None

        self.plot = {}
        self.model_type = "not_trained"
        self.pipeline_output_path = ""

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

    def training(self, model_type: str, cv_method="grouped", split_group_on="split_int"):
        """_summary_

        Args:
            model_type (str): RFC, GBC, neuron, ensemble [neuron,rfc]
            cv_method (_type_, optional): grouped, stratified
            split_group_on: on what var. to split the groups on if grouped == cv_method
        """
        self.trained_model = train_test.trained_model()  # init the class
        self.trained_model.five_fold_cross_train(
            self_pip=self,
            estimator=models.get_estimator(model_type),
            type_cv=cv_method,
            grouped_split_on=split_group_on,
        )
        self.name = self.name + "_" + model_type
        self.model_type = model_type
        self.pipeline_output_path = self.user_input.model_path + f"/Pipelines/{model_type}"

    def build_master_RFC(self):
        self.master_RFC = models.master_model_RFC(self.trained_model.model_with_info, ohe=self.ohe)

    def feature_imp_barplot(self, n_features=50, func_for_aggregation=np.mean):
        # Collect all features you don't want in the feature plot
        features_to_exclude = self.data.X_ohe_map[self.data.X_ohe_map["name_print"] == "split_int"].index.tolist()



        fig, ax, plotX = plot.feature_imp_barplot(
            model=self.master_RFC,
            small_plot=True,
            func_for_aggregation=func_for_aggregation,
            n_features=n_features,
            features_to_exclude=features_to_exclude,
            color_dict=self.mapper.color_groups,
            X_ohe_map=self.data.X_ohe_map,
            pip_self=self,
        )

        try:
            self.eval.feature_imp.update({"PlotX": plotX})
        except:
            print("Could not append PlotX, please run evaluation before this plot...")

        # Clean feature names by removing prefixes
        clean_feature_names = plotX.index.str.replace(r"^(remainder__|one_hot_encoder__)", "", regex=True)

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
        """
        Perform SHAP analysis on the best performing estimator from the ensemble.

        Args:
        sample_size (int, optional): Number of samples to use for SHAP analysis. If None, use all samples.
        max_display (int): Maximum number of features to display in the SHAP summary plot.

        Returns:
        matplotlib.figure.Figure: Figure object containing the SHAP summary plot.
        """
        X_val = pd.DataFrame(
            data=self.ohe.transform(self.data.X_val),
            columns=pd.Series(self.ohe.get_feature_names_out())
            .str.replace("remainder__", "")
            .str.replace("one_hot_encoder__", ""),
        )

        # Sample the validation set if sample_size is specified
        if sample_size is not None and len(X_val) > sample_size:
            X_val_sample = X_val.sample(sample_size, random_state=42)
        else:
            X_val_sample = X_val

        # Find the best performing estimator
        best_score = -np.inf
        best_estimator = None
        for grid_search in self.master_RFC.models:
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_estimator = grid_search.best_estimator_

        if best_estimator is None:
            raise ValueError("No best estimator found")

        # Create SHAP explainer for the best estimator
        explainer = shap.TreeExplainer(best_estimator)
        shap_values = explainer.shap_values(X_val_sample)

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

        return plt.gcf()

    def evaluation(self, only_val=False):
        self.eval = train_test.eval(self, only_val=only_val)
        self.eval.get_pv_test_train(self)

    def roc_auc_test_train(self):
        fig, ax = plt.subplots(1, 3, figsize=(15, 8), sharey=False)
        ax = ax.ravel()
        plot.plot_rocs(
            self.eval.train["tprs"].transpose(),
            self.eval.train["aucs"],
            n_splits=self.user_input.training["n_splits"],
            ax=ax[0],
        )
        plot.plot_rocs(
            self.eval.test["tprs"].transpose(),
            self.eval.test["aucs"],
            n_splits=self.user_input.training["n_splits"],
            ax=ax[1],
        )
        plot.plot_rocs(
            self.eval.val["tprs"].transpose(),
            self.eval.val["aucs"],
            n_splits=self.user_input.training["n_splits"],
            ax=ax[2],
        )
        for axi, tit in zip(ax, ["Training", "Testing", "Validation"]):
            axi.set_title(tit)
        fig.suptitle(f"Evaluation of {self.name}")
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
        def round_columns(df, columns, precision):
            df[columns] = df[columns].applymap(lambda x: round(x, precision) if isinstance(x, (int, float)) else x)
            return df

        columns_precision = {
            'Mean': 3,
            'Std. Dev.': 3,
            'Fold 1': 3,
            'Fold 2': 3,
            'Fold 3': 3,
            'Fold 4': 3,
            'Fold 5': 3
        }

        # Ensure evaluation has been performed
        if not hasattr(self, 'eval'):
            self.evaluation()

        aucs = self.eval.val["aucs"]
        auprcs = self.eval.val["auprcs"]
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        mean_auprc = np.mean(auprcs)
        std_auprc = np.std(auprcs)

        # Create evaluation tables for AUROC and AUPRC
        evaluation_results = pd.DataFrame({
            'Model': [self.user_input.col_subset] * 2,
            'Dataset': [self.user_input.row_subset] * 2,
            'Metric': ['AUROC', 'AUPRC'],
            'Mean': [mean_auc, mean_auprc],
            'Std. Dev.': [std_auc, std_auprc],
            'Fold 1': [aucs[0], auprcs[0]],
            'Fold 2': [aucs[1], auprcs[1]],
            'Fold 3': [aucs[2], auprcs[2]],
            'Fold 4': [aucs[3], auprcs[3]],
            'Fold 5': [aucs[4], auprcs[4]]
        })

        # Round the columns
        for column, precision in columns_precision.items():
            evaluation_results = round_columns(evaluation_results, [column], precision)


        #Store results in pipeline object
        if not hasattr(self, 'evaluation_results'):
            self.evaluation_results = {}
        self.evaluation_results['independent'] = evaluation_results

        # Save to Excel, updating existing file if it exists
        if output_path is None:
            output_path = os.path.join('combined_output', 'val', 'all_evaluation_results.xlsx')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Check if file exists
        file_exists = os.path.exists(output_path)

        if file_exists:
            # Read existing data if file exists
            with pd.ExcelFile(output_path) as xls:
                if 'Independent metrics' in xls.sheet_names:
                    existing_data = pd.read_excel(xls, 'Independent metrics')
                    # Remove existing entries for this model and dataset
                    existing_data = existing_data[
                        (existing_data['Model'] != self.user_input.col_subset) |
                        (existing_data['Dataset'] != self.user_input.row_subset)
                    ]
                    evaluation_results = pd.concat([existing_data, evaluation_results], ignore_index=True)

            # Append to existing file
            with pd.ExcelWriter(output_path, mode='a', if_sheet_exists="replace") as writer:
                evaluation_results.to_excel(writer, sheet_name='Independent metrics', index=False)
        else:
            # Create new file
            with pd.ExcelWriter(output_path, mode='w') as writer:
                evaluation_results.to_excel(writer, sheet_name='Independent metrics', index=False)

        print(f"Threshold-independent evaluation results saved to: {output_path}")
        return evaluation_results





    def evaluation_summary_threshold_dependent(self, output_path=None, thresholds=np.arange(0.6, 0.29, -0.01), beta=10):
        from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix, fbeta_score
        import numpy as np

        columns_precision = {
            'Precision': 3,
            'Recall': 2,
            'Accuracy': 2,
            'F1 Score': 3,
            f'F-beta Score (beta={beta})': 3,
            'Balanced Accuracy': 2,
            'PPV': 4,
            'NPV': 4,
            'NNS': 1,
            'TP': 0,
            'FP': 0,
            'FN': 0,
            'TN': 0,
            'TP %': 3,
            'FN %': 3
        }

        if thresholds is None:
            thresholds = np.arange(0.6, 0.29, -0.01)

        # Ensure evaluation has been performed
        if not hasattr(self, 'eval'):
            self.evaluation()

        threshold_evaluation_results = pd.DataFrame()

        proba = self.master_RFC.predict_proba(self.ohe.transform(self.data.X_val))
        y_true = self.data.y_val.values 

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

            # Create evaluation table for the threshold
            evaluation_table_threshold = pd.DataFrame({
                'Model': [self.user_input.col_subset],
                'Dataset': [self.user_input.row_subset],
                'Threshold': [threshold],
                'Precision': [precision],
                'Recall': [recall],
                'Accuracy': [accuracy],
                'F1 Score': [f1],
                f'F-beta Score (beta={beta})': [f_beta],
                'Balanced Accuracy': [balanced_accuracy],
                'PPV': [ppv],
                'NPV': [npv],
                'NNS': [1 / ppv if ppv > 0 else float('inf')],
                'TP': [tp],
                'FP': [fp],
                'FN': [fn],
                'TN': [tn],
                'TP %': [tp_percentage],
                'FN %': [fn_percentage]
            })

            threshold_evaluation_results = pd.concat([threshold_evaluation_results, evaluation_table_threshold], ignore_index=True)

        # Apply rounding to all columns based on the specified precision
        for column, precision in columns_precision.items():
            if column in threshold_evaluation_results.columns:
                threshold_evaluation_results[column] = threshold_evaluation_results[column].apply(
                    lambda x: round(x, precision) if isinstance(x, (int, float)) else x
                )
            else:
                print(f"Warning: Column '{column}' not found in the results DataFrame.")

        # Store results in the Pipeline object
        if not hasattr(self, 'evaluation_results'):
            self.evaluation_results = {}
        self.evaluation_results['threshold_dependent'] = threshold_evaluation_results


        # Save to Excel, updating existing file if it exists
        if output_path is None:
            output_path = os.path.join('combined_output', 'val', 'all_evaluation_results.xlsx')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Check if file exists
        file_exists = os.path.exists(output_path)

        if file_exists:
            # Read existing data if file exists
            with pd.ExcelFile(output_path) as xls:
                if 'Threshold metrics_all_thresholds' in xls.sheet_names:
                    existing_data = pd.read_excel(xls, 'Threshold metrics_all_thresholds')
                    # Remove existing entries for this model and dataset
                    existing_data = existing_data[
                        (existing_data['Model'] != self.user_input.col_subset) |
                        (existing_data['Dataset'] != self.user_input.row_subset)
                    ]
                    threshold_evaluation_results = pd.concat([existing_data, threshold_evaluation_results], ignore_index=True)

            # Sort the results by Model, Dataset, and Threshold
            threshold_evaluation_results = threshold_evaluation_results.sort_values(['Model', 'Dataset', 'Threshold'])

            # Append to existing file
            with pd.ExcelWriter(output_path, mode='a', if_sheet_exists="replace") as writer:
                threshold_evaluation_results.to_excel(writer, sheet_name='Threshold metrics_all_thresholds', index=False)
        else:
            # Create new file
            with pd.ExcelWriter(output_path, mode='w') as writer:
                threshold_evaluation_results.to_excel(writer, sheet_name='Threshold metrics_all_thresholds', index=False)

        print(f"Threshold-dependent evaluation results saved to: {output_path}")
        return threshold_evaluation_results





    def save_values_for_combined_plot(self, only_val=False):
        """Save the prediction values of testing, training and validation together with the eids in an excel
        as well as the tprs and prcs

        Args:
            only_val (bool, optional): run the export only for the validation cohort. Defaults to False.
        """
        if only_val:
            train, test, val = None, None, self.eval.val['predicted_values'] if hasattr(self.eval, 'val') else None
        else:
            train, test, val = self.eval.test_train_pred.get('train'),self.eval.test_train_pred.get('test'),self.eval.test_train_pred.get('val')

        # For the validation cohort
        if val is not None and hasattr(self.eval, 'val') and 'tprs' in self.eval.val.keys():
            self.eval.save_performance_combination(
                pip_self=self,
                tprs=self.eval.val["tprs"],
                pred_values=val["y_pred"] if type(val)== pd.DataFrame and "y_pred" in val.columns else None,
            y_true=val.status if isinstance(val, pd.DataFrame) else None,
            true_cancerreg=getattr(self.data.z_val, 'status_cancerreg', None) if hasattr(self.data, 'z_val') else None,
            cohort="val",
        )
        else:
            print("Validation data not available. Skipping validation performance combination.1")

        if not only_val:
            # For the training
            if isinstance(train, pd.DataFrame) and "y_pred" in train and "y_true" in train:
                self.eval.save_performance_combination(
                    self, self.eval.train["tprs"], train.y_pred, train.status, None, cohort="train"
                )
            else:
                print("Training data incomplete. Skipping training performance combination.2")

            # For the testing
            if isinstance(test, pd.DataFrame) and "y_pred" in test and "y_true" in test:
                self.eval.save_performance_combination(
                    self, self.eval.test["tprs"], test.y_pred, test.status, None, cohort="test"
                )
            else:
                print("Testing data incomplete. Skipping testing performance combination.3")



    def save_values_for_validation(self):
        if hasattr(self.data, 'z_val') and hasattr(self.data.z_val, 'status_cancerreg'):
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

    def save_Pipeline(self):
        path = self.user_input.model_path + f"/Pipelines/{self.model_type}/"
        os.makedirs(path, exist_ok=True)
        model_save_to = path + self.name + ".joblib"
        dump(self, model_save_to)
        print("Pipeline saved to:\n", model_save_to)

    def save_Pipeline_and_comb_outputs(self, only_val=False):
        self.save_values_for_combined_plot(only_val=only_val)
        path = self.user_input.model_path + f"/Pipelines/{self.model_type}/"
        os.makedirs(path, exist_ok=True)
        model_save_to = path + self.name + ".joblib"
        dump(self, model_save_to)
        print("Pipeline saved to:\n", model_save_to)


def plot_KM(pl, thrsholds_limits: list = [0.4, 0.5], color_dict={"low": "green", "medium": "yellow", "high": "red"}):
    import pandas as pd
    import numpy as np
    from lifelines import KaplanMeierFitter
    import matplotlib.pyplot as plt

    def kaplan_meier_analysis(time, event, group, group_labels=None, plot=True, color_dict=color_dict):
        """
        Perform Kaplan-Meier analysis and compare survival curves between two groups.,
        Parameters:,
        - time: Array-like object containing the time to event or censoring.,
        - event: Array-like object indicating whether an event occurred (1) or not (0).,
        - group: Array-like object specifying the group each observation belongs to (e.g., treatmentcontrol).,
        - group_labels: List of labels for the two groups (optional).,
        - plot: Boolean indicating whether to plot the survival curves (default is True).,
        Returns:,
        - kmf: KaplanMeierFitter object containing the survival estimates for each group.,
        """
        # Creating a DataFrame from the provided data,
        df = pd.DataFrame({"time": list(time), "event": list(event), "group": list(group)})
        # Initializing KaplanMeierFitter,
        kmf = KaplanMeierFitter()
        # Group-wise analysis,
        for i, grp in enumerate([i for i in df["group"].unique()]):
            data = df.loc[df["group"] == grp, :]
            kmf.fit(data["time"], event_observed=data["event"], label=group_labels[i] if group_labels else f"{grp}")
            if plot:
                kmf.plot(color=color_dict.get(grp, "green"), alpha=0.6)

        if plot:
            plt.xlabel("Time [d]"),
            plt.ylabel(f"Probability to survive without a {pl.user_input.DOI} [%]"),
            plt.title(f"Kaplan-Meier estimator - Time to {pl.user_input.DOI}"),
            plt.legend(),
            plt.show(),
            plt.tight_layout(),
        return kmf

    def get_group(pred_prob, thresholds):
        groups = []
        for i in pred_prob:
            if i < thresholds[0]:
                groups.append("low")
            elif i >= thresholds[0] and i < thresholds[1]:
                groups.append("medium")
            elif i >= thresholds[1]:
                groups.append("high")
        return groups

    time_censoring = pd.Timestamp(year=2024, day=1, month=1)
    z_val = pl.data.z_val

    z_val.loc[z_val.date_of_diag.isna(), "date_of_diag"] = time_censoring
    timedelta = pd.to_datetime(z_val.date_of_diag) - (pd.to_datetime(z_val["Date of assessment"]))
    z_val["time_to_event_d"] = [i.days for i in timedelta]
    z_val["pred_prob"] = pl.master_RFC.predict_proba(pl.ohe.transform(pl.data.X_val)).values

    z_val["risk_group"] = get_group(z_val.pred_prob, thresholds=thrsholds_limits)
    print("The Thresholds are:", thrsholds_limits)
    print(z_val.risk_group.value_counts())
    fig, ax = plt.subplots(figsize=(20, 20))
    estimator = kaplan_meier_analysis(z_val.time_to_event_d, event=z_val.status, group=z_val.risk_group)
    estimator.plot_survival_function()
    return estimator
