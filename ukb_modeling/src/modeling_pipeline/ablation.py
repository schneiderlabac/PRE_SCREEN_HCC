import pandas as pd
import numpy as np
import copy
import os
import sys
import joblib
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt


class ohe_w_filt:
            def __init__(self, ohe, cols_to_exclde=[]):
                """
                Initialize the one-hot encoder wrapper with feature filtering capability.

                Feature Filtering: Allows selective exclusion of features without modifying the original data
                Transparent Wrapping: Passes through all relevant methods to the underlying encoder
                Feature Tracking: Maintains lists of both included and excluded features
                Compatible Interface: Implements the same interface as the standard OneHotEncoder

                Parameters:
                -----------
                ohe : OneHotEncoder
                    The original one-hot encoder to wrap
                cols_to_exclde : list
                    List of column names to exclude from the encoding
                """
                self.ohe = copy.copy(ohe)
                for key, val in vars(ohe).items():
                    setattr(self, key, val)
                self._names_original = pd.Series(ohe.get_feature_names_out())
                self.cols_to_exclde = cols_to_exclde
                self.index_cols_to_exclude = self._names_original.loc[
                    self._names_original.isin(self.cols_to_exclde)
                ].index

                # get the cols that are included
                self.get_included_cols = pd.Series(self.ohe.get_feature_names_out()).loc[
                    ~pd.Series(self.ohe.get_feature_names_out()).isin(cols_to_exclde)
                ]
                self.get_excluded_cols_literal = pd.Series(self.ohe.get_feature_names_out()).loc[
                    pd.Series(self.ohe.get_feature_names_out()).isin(cols_to_exclde)
                ]

            def get_feature_names_out(self, *args, **kwargs):
                return np.array(
                    pd.Series(self.ohe.get_feature_names_out(*args, **kwargs)).drop(self.index_cols_to_exclude)
                )

            def transform(self, *args, **kwargs):
                df = pd.DataFrame(self.ohe.transform(*args, **kwargs))
                df = df.loc[:, ~df.columns.isin(self.index_cols_to_exclude)]
                return df.values

            def get_params(self, *args, **kwargs):
                return self.ohe.get_params(*args, **kwargs)

            def set_params(self, *args, **kwargs):
                return self.ohe.set_params(*args, **kwargs)

            def fit_transform(self, *args, **kwargs):
                return self.ohe.fit_transform(*args, **kwargs)

            def set_cols_to_exclude(self, cols_to_exclde):
                self.cols_to_exclde = cols_to_exclde
                self.index_cols_to_exclude = self._names_original.loc[
                    self._names_original.isin(self.cols_to_exclde)
                ].index

class ablation:
    def __init__(
        self, pip_self, to_n_features=1, step_size=1, n_features_elim_tail_at_start=0, export_temp_results=False,
        allowed_features=None, allowed_features_mode="raw"
    ):
        """Run ablation analysis on the mean voting Model. Eliminates features in a stepwise manner and evaluates the performance of the model after each elimination.
        For a random forest it makes sense to only drop one feature at a time.

        -> make sure that you fitted the model before running this function ones on the whole dataset you have available for training and built the master model (might be the RFC Master function), but not on the testing set(old-notation-validation)

        #### Methods

        - **`run_ablation()`**: Performs the ablation analysis by:
        1. Creating a wrapped one-hot encoder to filter features
        2. Iteratively removing the least important features
        3. Retraining the model after each feature reduction
        4. Evaluating performance metrics at each step
        5. Saving the ablation object for later use
        6. loading a new instance of the ablation object from previous


        Args:
            to_n_features (int, optional): _description_. Defaults to 1. -> will run for all features:) you might see a nice peek of performance on the
            steps (int, optional): depends on the model you have and the featurespace, and the time you would like to spent:). Defaults to 1.
            n_features_elim_tail_at_start (int, optional): you can just eliminate the tail (10 or whatever) features from the start to lower the computational expense. Defaults to 0.
            export_temp_results (bool, optional): Export the temporary results (the whole estimator) so have a look later. Defaults to False.
            allowed_features (list, optional): List of feature names to restrict ablation to. If provided, all other features are excluded from start. Defaults to None.
            allowed_features_mode (str, optional): Mode for interpreting allowed_features. 'raw' for original column names, 'encoded' for exact OHE names. Defaults to 'raw'.
        """
        self.pl = copy.copy(
            pip_self
        )  # not to change the original pipeline object TODO: check if this is too memory consuming
        self.feature_names_in_lit_to_num = {
            k: v for v, k in pd.Series(pip_self.ohe.get_feature_names_out()).to_dict().items()
        }
        self.feature_names_in_num_to_lit = {
            v: k for v, k in pd.Series(pip_self.ohe.get_feature_names_out()).to_dict().items()
        }
        self.features_excluded = {}
        self.to_n_features = to_n_features
        self.step_size = step_size
        self.export_temp_results = export_temp_results
        self.results = {}
        self.eval_function_results={}
        self.feature_importances = {}
        self.scorer = []
        self.allowed_features_mode = allowed_features_mode # 'raw' or 'encoded'
        self.allowed_features = allowed_features # default: None

        # Handle allowed feature subset
        if allowed_features is not None:
            all_encoded = list(self.feature_names_in_lit_to_num.keys())

            if allowed_features_mode == "encoded":
                allowed_encoded = [f for f in all_encoded if f in allowed_features]
                missing = set(allowed_features) - set(allowed_encoded)
                if missing:
                    print(f"[ablation] Warning: {len(missing)} encoded feature names not found: {list(missing)[:5]}...")
            elif allowed_features_mode == "raw":
                # Leverage existing ohe wrapper mapping logic
                temp_wrapper = ohe_w_filt(self.pl.ohe)
                allowed_encoded = temp_wrapper.map_raw_to_encoded(allowed_features)
                if not allowed_encoded:
                    raise ValueError("None of the provided raw feature names matched encoded columns.")
                print(f"[ablation] Mapped {len(allowed_features)} raw features to {len(allowed_encoded)} encoded features")
            else:
                raise ValueError("allowed_features_mode must be 'raw' or 'encoded'.")

            initially_excluded = set(all_encoded) - set(allowed_encoded)
            if len(initially_excluded) == len(all_encoded):
                raise ValueError("All features would be excluded. Check allowed_features input.")

            # Register exclusions with iteration key -2 (before baseline -1)
            self.features_excluded.update({
                f"-2_{self.feature_names_in_lit_to_num[name]}": name for name in initially_excluded
            })
            print(
                f"[ablation] Restricting to {len(allowed_encoded)}/{len(all_encoded)} features. Pre-excluding {len(initially_excluded)} features."
            )

            if to_n_features > len(allowed_encoded):
                print(f"[ablation] Adjusting to_n_features from {to_n_features} to {len(allowed_encoded)}")
                self.to_n_features = len(allowed_encoded)


        # adjust the elimination at the start for proportion
        if n_features_elim_tail_at_start > 0 and n_features_elim_tail_at_start < 1:
            self.n_features_elim_tail_at_start = np.round(
                len(self.feature_names_in_lit_to_num.keys()) * n_features_elim_tail_at_start
            ).astype(int)
        else:
            self.n_features_elim_tail_at_start = n_features_elim_tail_at_start

        # check if model is trained
        if pip_self.model_type == "not_trained" and hasattr(pip_self, "eval"):
            raise Exception(
                "Model not trained yet. Please train and evaluate the model first. and build the master model"
            )

    def save(self, path, ablation_object_name="ablation_object.joblib",compress=('zlib', 3)):
        """
        Save the ablation object to a file using joblib for later use.

        Parameters:
        filepath (str): Path where the object will be saved
        """
        # Check if the ablation_object_name ends with ".joblib" or ".pkl"
        if not (ablation_object_name.endswith(".joblib") or ablation_object_name.endswith(".pkl")):
            raise ValueError("The ablation_object_name must end with '.joblib' or '.pkl'")

        directory = os.path.join(path, "Ablation_Objects")
        if not os.path.exists(directory):
            os.makedirs(directory)

        filepath = os.path.join(directory, ablation_object_name) #Change this to include the details of this ablation object later
        print(filepath)
        joblib.dump(self, filepath,compress=compress)
        print(f"Ablation object saved to {filepath}")


    #Load a previously generated ablation object from a file
    #The purpose of load is to create a new ablation object from a saved file,
    # so it doesn't make sense to call it on an existing instance of the class.
    @classmethod
    def load(cls, path, ablation_object_name="ablation_object.joblib"):
        """
        Load an ablation object from a file using joblib.

        Parameters:
        filepath (str): Path to the saved ablation object

        Returns:
        ablation: The loaded ablation object
        """

        filepath = os.path.join(path, "Ablation_Objects", ablation_object_name) #Change this to include the details of this ablation object later
        obj = joblib.load(filepath)
        print(f"Ablation object loaded from {filepath}")
        return obj

    def get_pipeline(self, model_iteration):
        """
        Get the pipeline object with only the features excluded up to a specific iteration.
        """
        if model_iteration not in self.results:
            raise ValueError(f"Model iteration {model_iteration} not found in results")

        # Create a fresh copy of the pipeline
        pl = copy.deepcopy(self.pl)

        # Get features excluded only for this specific iteration
        excluded_features = []

        # Only collect features that were excluded up to and including the requested iteration
        for i in range(model_iteration + 1):
            for key, val in self.features_excluded.items():
                if key.startswith(f"{i}_"):
                    excluded_features.append(val)

        print(f"Total excluded features: {len(excluded_features)}")

        # Create the ohe wrapper with the excluded features
        ohe_orig = copy.copy(pl.ohe)
        pl.ohe = ohe_w_filt(ohe_orig, cols_to_exclde=excluded_features)

        # Check how many features remain
        remaining_features = pl.ohe.get_feature_names_out()
        print(f"Remaining features: {len(remaining_features)}")
        print(f"Feature names: {remaining_features}")

        return pl

    def run_ablation(self,func_eval=None):
        """The reduction in the columns is not done by reducing X or X_val, but by modifying the ohe
        -> this wrapper will leave out the least important features
        func_eval: function to evaluate the model, if None the default scorer will be used
        """

        if func_eval is None:
            print("No evaluation function provided. Using default scorer, no evaluation function will be saved")



        def scorer(pl):
            """
            Default scorer function to evaluate the model performance.
            Returns:
            - mean_auc: Mean AUC score from the validation set
            - mean_auprc: Mean AUPRC score from the validation set
            - best_params: Best parameters from the Random Forest Classifier
            - excluded_cols: Columns that were excluded from the one-hot encoding
            - included_cols: Columns that were included in the one-hot encoding"""
            mean_auc = np.mean(pl.eval.val["aucs"])
            best_params = pl.master_RFC.get_best_params()
            mean_auprc = np.mean(pl.eval.val["auprcs"])
            excluded_cols=pl.ohe.get_excluded_cols_literal
            included_cols=pl.ohe.get_included_cols
            #balanced_accuracy = balanced_accuracy_score(pl.eval.val.get("predicted_values")[pl.user_input.target_to_validate_on], pl.eval.val.get("predicted_values")["y_pred"])
            return (mean_auc, mean_auprc, best_params,excluded_cols,included_cols)

        def get_ranks(self, feature_imp: pd.DataFrame, ohe, step_size=None):
            if step_size is None:
                return_nth_least_imp_feature = self.step_size
            else:
                return_nth_least_imp_feature = step_size
            feature_imp = feature_imp.copy()
            mean_ranks = pd.DataFrame()
            for col in feature_imp.columns[~feature_imp.columns.str.startswith("mean")]:
                feature_imp["rank" + col] = feature_imp[col].rank(ascending=False)
                mean_ranks["rank"] = feature_imp[col].rank(ascending=False)
            feature_imp["mean_rank"] = mean_ranks
            feature_imp.index = ohe.get_included_cols.tolist()
            try:
                n_th = (
                    feature_imp["mean_rank"]
                    .sort_values(ascending=True)
                    .tail(return_nth_least_imp_feature)
                    .index.tolist()
                )
            except:
                n_th = (
                    feature_imp["mean_rank"]
                    .sort_values(ascending=True)
                    .tail(feature_imp.shape[0] - self.to)
                    .index.tolist()
                )
            return feature_imp, n_th

        ################## code starts here ##################
        ohe_orig = copy.copy(self.pl.ohe)
        counter = -1
        while self.pl.ohe.get_feature_names_out().shape[0] > self.to_n_features:
            # set a counte for the iterations:
            counter += 1
            # for the first iteration we will eliminate the tail of the features
            if counter == 0:
                # 1. gat a trained model -> save evaluation and feature importances at 'baseline'
                self.pl.ohe = ohe_w_filt(
                    ohe_orig, cols_to_exclde=list(self.features_excluded.values())
                )  # no excluded values at this time

                # eliminate the n last features for it it the first iteration
                # first - adjust the ohe -> get the n last features to drop and update excluded features
                print(
                    f"Step {counter-1}: Eliminating {self.n_features_elim_tail_at_start} features with this iteration"
                )
                feature_imp, features_to_drop = get_ranks(
                    self,
                    copy.copy(self.pl.master_RFC.feature_importances_()),
                    self.pl.ohe,
                    step_size=self.n_features_elim_tail_at_start,
                )
                self.feature_importances.update({counter - 1: feature_imp})
                if func_eval is not None:
                    self.eval_function_results.update({counter - 1:func_eval(self.pl)})
                self.results.update({counter - 1: self.pl.eval})
                self.scorer.append(scorer(self.pl))
                self.features_excluded.update(
                    {
                        f"{counter-1}_{v}": i
                        for indexx, (i, v) in enumerate(self.feature_names_in_lit_to_num.items())
                        if i in features_to_drop
                    }
                )


            print(f"Step {counter}: Eliminating {self.step_size} features with this iteration")
            self.pl.ohe = ohe_w_filt(ohe_orig, cols_to_exclde=list(self.features_excluded.values()))
            # retrain the model on the smaller feature space
            self.pl.training(self.pl.model_type)
            self.pl.build_master_RFC()
            self.pl.evaluation()
            # log your results
            feature_imp, features_to_drop = get_ranks(
                self, copy.copy(self.pl.master_RFC.feature_importances_()), self.pl.ohe, step_size=self.step_size
            )
            self.feature_importances.update({counter: feature_imp})
            self.results.update({counter: self.pl.eval})
            self.scorer.append(scorer(self.pl))
            self.features_excluded.update(
                {
                    f"{counter}_{v}": i
                    for indexx, (i, v) in enumerate(self.feature_names_in_lit_to_num.items())
                    if i in features_to_drop
                }
            )
    def get_scorer_df(self,save_path=None):
        """
        Extracts performance metrics into a DataFrame.
        Returns:
            pd.DataFrame: DataFrame containing the performance metrics
        """
        df = pd.DataFrame(self.scorer, columns=["mean_auc", "mean_auprc", "best_params", "excluded_cols", "included_cols"])
        df['len_excluded'] = df['excluded_cols'].apply(lambda x: len(x))
        df['len_included'] = df['included_cols'].apply(lambda x: len(x))

        if save_path is not None:
            df.to_csv(os.path.join(save_path, "ablation_df.csv"))
            print(f"Scorer DataFrame saved to {os.path.join(save_path, 'ablation_df.csv')}")
        return df
    def plot_feature_rank_lineplot(self,ax=None,export_path=None):
        f_imp_all=pd.DataFrame()
        for i in self.feature_importances.values():
            f_imp_all=pd.concat([f_imp_all,i['mean_rank']],axis=1)
        f_imp_all.columns=self.feature_importances.keys()

        # filter all with only nans
        f_imp_all=f_imp_all.dropna(subset=f_imp_all.columns[1:],how='all')
        f_imp_all=(f_imp_all-f_imp_all.max().max())*-1
        if ax is None:
            fig, ax = plt.subplots(figsize=(20,10))
        plot_df=f_imp_all.T
        plot_df.plot.line(ax=ax)
        plot_df.columns=plot_df.columns.str.replace('scaler__','').str.replace('one_hot_encoder__','')
        ax.set_xticklabels(list())
        # adjust the labeling to not use the legend
        counter=0
        for i,v in ((plot_df.T.isna().sum(1).sort_values(ascending=True)-max(plot_df.T.isna().sum(1)))*-1).items():
            ax.annotate(i, (v+0.1,v-0.5), textcoords="offset points", xytext=(0,10), ha='left', color='gray',fontsize=12)
        # adjust the ax labels:
        ax.set_xlabel('iteration of ablation study ->')
        ax.set_ylabel('rank of feature importance with higher ranges being more important')
        ax.legend().remove()
        return ax
### Helper Functions
def get_scorer_df(ab):
    # Extracts performance metrics into a DataFrame
    df=pd.DataFrame(ab.scorer)
    df[['len_excluded','len_included']]=df[[3,4]].map (lambda x: len(x))
    return df