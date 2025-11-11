import array
import pandas as pd
import numpy as np
import os
from joblib import load
from joblib import dump
from datetime import datetime

# For RandomForestClassifier
from sklearn import neural_network
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from collections import defaultdict
from sklearn.inspection import permutation_importance
from catboost import CatBoostClassifier
#from tabpfn import TabPFNClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import torch


def get_survival_y(y: pd.DataFrame, col_event_observed="osstat", col_time_to_event="ostm", drop_na=True):
    """generate the survival array for the survival models

    Args:
        y (pd.DataFrame): DataFrame with id as index and column with status and one with time to event
        col_event_observed (str, optional): _description_. Defaults to "osstat".
        col_time_to_event (str, optional): _description_. Defaults to "ostm".
        drop_na (bool, optional): _description_. Defaults to True. -> if you have a model that can handle nan values, set to False.

    Returns:
        y_array: array with the status and time to event as a structured array
        index -> index of the dropped y so all ids that are included in the structured array

    """
    if drop_na:
        y = y.dropna(subset=[col_event_observed, col_time_to_event], inplace=False)
    y["status"] = [(i, a) for i, a in zip(y[col_event_observed] == 1, y[col_time_to_event])]
    y_array = np.array(y["status"], dtype=[("Status", "?"), ("Survival_in_days", "<f8")])
    return y_array, y.index.tolist()


def get_estimator(label, fixed_params={}):
    random_state = fixed_params["random_state"] #pass fixed hyperparameters to the model via local variables (as **fixed_params did not work)

    rsf = RandomSurvivalForest
    gbs = GradientBoostingSurvivalAnalysis
    # rsf.predict_proba = rsf.predict.to_series()

    cox = CoxPHSurvivalAnalysis
    # class TabPFNClassifierWrapper(BaseEstimator, ClassifierMixin):
    #     """
    #     Scikit-learn compatible wrapper for TabPFNClassifier.

    #     Allows use in sklearn Pipelines, cross_val_score, GridSearchCV, etc.

    #     Parameters:
    #     -----------
    #     device : str
    #         'cuda' or 'cpu' depending on hardware availability.
    #     """

    #     def __init__(self, device='cpu',random_state=42,**kwargs):
    #         if not torch.cuda.is_available() and device == 'cuda':
    #             if torch.backends.mps.is_available():
    #                 device = 'mps'  # Use Metal Performance Shaders on macOS
    #             else:
    #                 device = 'cpu'
    #             #raise Warning(f"CUDA does not seem to be available. Using {device} instead.")
    #         self.device = device
    #         self.model_ = None
    #         self.random_state = random_state
    #         self.kwargs = kwargs

    #     def _apply_undersampling(self, X, y, undersample_ratio=5.0, random_state=None):
    #         """
    #         Apply undersampling to the dataset, preserving all cases (y==1) and
    #         undersampling controls (y==0) based on the specified ratio.

    #         Parameters:
    #         -----------
    #         X : array-like, shape (n_samples, n_features)
    #             Training input samples.
    #         y : array-like, shape (n_samples,)
    #             Target labels.
    #         undersample_ratio : float
    #             Ratio of controls to cases for training.
    #         random_state : int, optional
    #             Random state for reproducible sampling.

    #         Returns:
    #         --------
    #         X_resampled : array-like
    #             Resampled input samples.
    #         y_resampled : array-like
    #             Resampled target labels.
    #         """
    #         if random_state is None:

    #                         random_state = self.random_state
    #         # Separate cases and controls
    #         case_mask = (y == 1)
    #         control_mask = (y == 0)

    #         X_cases = X[case_mask]
    #         y_cases = y[case_mask]
    #         X_controls = X[control_mask]
    #         y_controls = y[control_mask]

    #         n_cases = len(y_cases)
    #         n_controls = len(y_controls)

    #         print(f"Original dataset: {n_cases} cases, {n_controls} controls")

    #         if n_cases == 0:
    #             raise ValueError("No cases (y==1) found in the dataset")

    #         if n_controls == 0:
    #             raise ValueError("No controls (y==0) found in the dataset")

    #         # Calculate target number of controls based on ratio
    #         target_n_controls = int(n_cases * undersample_ratio)

    #         if target_n_controls >= n_controls:
    #             print(f"Warning: Requested {target_n_controls} controls but only {n_controls} available. Using all controls.")
    #             X_controls_sampled = X_controls
    #             y_controls_sampled = y_controls
    #         else:
    #             # Undersample controls
    #             X_controls_sampled, y_controls_sampled = resample(
    #                 X_controls, y_controls,
    #                 n_samples=target_n_controls,
    #                 random_state=random_state,
    #                 replace=False
    #             )
    #             print(f"Undersampled to: {n_cases} cases, {target_n_controls} controls (ratio: {undersample_ratio:.2f})")

    #         # Combine cases and undersampled controls
    #         X_resampled = np.vstack([X_cases, X_controls_sampled])
    #         y_resampled = np.hstack([y_cases, y_controls_sampled])

    #         # Shuffle the combined dataset
    #         shuffle_idx = np.random.RandomState(random_state).permutation(len(y_resampled))
    #         X_resampled = X_resampled[shuffle_idx]
    #         y_resampled = y_resampled[shuffle_idx]

    #         return X_resampled, y_resampled

    #     def fit(self, X, y, undersample_model=None, undersample_ratio=1.0, random_state=None):
    #         """
    #         Fit the TabPFN classifier with optional undersampling.

    #         Parameters:
    #         -----------
    #         X : array-like, shape (n_samples, n_features)
    #             Training input samples.
    #         y : array-like, shape (n_samples,)
    #             Target labels.
    #         undersample_model : bool, optional (default=None)
    #             Whether to apply undersampling before fitting. If None, no undersampling is applied.
    #         undersample_ratio : float, optional (default=1.0)
    #             Ratio of controls to cases for training.
    #             - 1.0 means equal number of controls and cases
    #             - 2.0 means twice as many controls as cases
    #             - 0.5 means half as many controls as cases
    #         random_state : int, optional (default=None)
    #             Random state for reproducible undersampling. If None, uses self.random_state.

    #         Returns:
    #         --------
    #         self : object
    #             Fitted estimator.
    #         """
    #         X = np.array(X)
    #         y = np.array(y)

    #         # Apply undersampling if requested
    #         if undersample_model:
    #             X_resampled, y_resampled = self._apply_undersampling(
    #                 X, y, undersample_ratio, random_state
    #             )
    #         else:
    #             X_resampled, y_resampled = X, y

    #         # Fit the model
    #         self.model_ = TabPFNClassifier(
    #             device=self.device,
    #             random_state=self.random_state,
    #             ignore_pretraining_limits=True,
    #             **self.kwargs
    #         )
    #         self.model_.fit(X_resampled, y_resampled)
    #         return self

    #     @property
    #     def classes_(self):
    #         if hasattr(self, "model_") and self.model_ is not None:
    #             return self.model_.classes_
    #         else:
    #             return None
    #     def predict(self, X):
    #         """
    #         Predict class labels for samples in X.

    #         Parameters:
    #         -----------
    #         X : array-like, shape (n_samples, n_features)

    #         Returns:
    #         --------
    #         y_pred : array, shape (n_samples,)
    #             Predicted class labels.
    #         """
    #         return self.model_.predict(np.array(X))

    #     def predict_proba(self, X):
    #         """
    #         Predict class probabilities for samples in X.

    #         Parameters:
    #         -----------
    #         X : array-like, shape (n_samples, n_features)

    #         Returns:
    #         --------
    #         y_proba : array, shape (n_samples, n_classes)
    #             Predicted class probabilities.
    #         """
    #         return self.model_.predict_proba(np.array(X))

    #     def score(self, X, y):
    #         """
    #         Return the mean accuracy on the given test data and labels.

    #         Parameters:
    #         -----------
    #         X : array-like, shape (n_samples, n_features)
    #         y : array-like, shape (n_samples,)

    #         Returns:
    #         --------
    #         score : float
    #         """
    #         return self.model_.score(np.array(X), np.array(y))

    #     def get_params(self, deep=True):
    #         """
    #         Get parameters for this estimator.
    #         """
    #         return {"device": self.device}

    #     def set_params(self, **params):
    #         """
    #         Set the parameters of this estimator.
    #         """
    #         for key, value in params.items():
    #             setattr(self, key, value)
    #         return self""


    class SurvivalEstimatorWrapper(rsf): # change to rsf, gbs, cox
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        # def fit(self, X, y, var_of_interest="status", **kwargs):
        #     # Ensure `y` is structured properly
        #     if isinstance(y, pd.DataFrame):
        #         self.y_array_fit = np.array(y[var_of_interest], dtype=[("Status", "bool"), ("Survival", "float64")])
        #     else:
        #         self.y_array_fit = y

        #     # Call the estimator's fit method
        #     self.estimator.fit(X=X, y=self.y_array_fit, **kwargs)
        #     return self

        # def predict(self, X):
        #     # Delegate to the base estimator
        #     return self.estimator.predict(X)

        def predict_proba(self, X):
            prediction = self.predict(X)
            # Convert predictions to a DataFrame with required format
            results = pd.DataFrame(prediction, columns=[1])
            results[0] = np.nan
            results = results.loc[:, [0, 1]]
            return results

        def permutate_feature_imp(self,X,y,n_repeats=30,random_state=42,n_jobs=-1):
            self._feature_importances_mean=permutation_importance(self, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs).get('importances_mean')
            self._feature_importances_std=permutation_importance(self, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs).get('importances_std')

        @property
        def feature_importances_(self):
            return self._feature_importances_mean



        # def score(self, X, y, var_of_interest="status"):
        #     # Ensure `y` is structured properly
        #     if isinstance(y, pd.DataFrame):
        #         y_array = np.array(y[var_of_interest], dtype=[("Status", "bool"), ("Survival", "float64")])
        #     else:
        #         y_array = y
        #     return self.estimator.score(X, y=y_array)

        def _get_param_names(self):
            return sorted(list(vars(self)))

        def get_params(self, deep=True):
            """
            Get parameters for this estimator.

            Parameters
            ----------
            deep : bool, default=True
                If True, will return the parameters for this estimator and
                contained subobjects that are estimators.

            Returns
            -------
            params : dict
                Parameter names mapped to their values.
            """
            out = dict()
            for key in self._get_param_names():
                if key not in ["estimator", "estimator_params", "class_weight"]:
                    value = getattr(self, key)
                    if deep and hasattr(value, "get_params") and not isinstance(value, type):
                        deep_items = value.get_params().items()
                        out.update((key + "__" + k, val) for k, val in deep_items)
                    out[key] = value
            return out

        def set_params(self, **params):
            """Set the parameters of this estimator.

            The method works on simple estimators as well as on nested objects
            (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
            parameters of the form ``<component>__<parameter>`` so that it's
            possible to update each component of a nested object.

            Parameters
            ----------
            **params : dict
                Estimator parameters.

            Returns
            -------
            self : estimator instance
                Estimator instance.
            """
            if not params:
                # Simple optimization to gain speed (inspect is slow)
                return self
            valid_params = self.get_params(deep=True)

            nested_params = defaultdict(dict)  # grouped by prefix
            for key, value in params.items():
                key, delim, sub_key = key.partition("__")
                if key not in valid_params:
                    local_valid_params = self._get_param_names()
                    raise ValueError(
                        f"Invalid parameter {key!r} for estimator {self}. "
                        f"Valid parameters are: {local_valid_params!r}."
                    )

                if delim:
                    nested_params[key][sub_key] = value
                else:
                    setattr(self, key, value)
                    valid_params[key] = value

            for key, sub_params in nested_params.items():
                valid_params[key].set_params(**sub_params)

            return self

    # Create an ensemble of classifiers



    ##### here you can edit the models to be called in the pipeline: dict of key -> that you have to give in the pipeline and the model you want to use
    models = {
        "RFC": RandomForestClassifier, #pass all fixed hyperparameters as separate kwargs to the model
        "XGB": GradientBoostingClassifier,
        "Log_l1": LogisticRegression,
        "neuronMLP": neural_network.MLPClassifier,
        "survival_rsf": SurvivalEstimatorWrapper,
        "CatBoost": CatBoostClassifier,
        #"TabPFN": TabPFNClassifierWrapper



        #"survival_cox": SurvivalEstimatorWrapper(cox),


    #TODO Ensemble classifier revivial @Paul
    # # Instantiate individual classifiers (these ones are for ensemble classifier)
    # mlp_classifier = neural_network.MLPClassifier(
    #     hidden_layer_sizes=(10, 10), activation="relu", solver="adam", random_state=42
    # )
    # rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    # ensemble_classifier = VotingClassifier(
    #     estimators=[("mlp", mlp_classifier), ("rf", rf_classifier)],
    #     voting="soft",)
    }

    return models[label](**fixed_params)


def save_model(model, ohe, base_path, postfix_model):

    # Get the current date and construct the full path
    current_date = datetime.now().strftime("%Y_%m_%d")
    full_path = os.path.join(base_path, current_date + "_model_exported")
    # Create the subfolder if it doesn't exist
    os.makedirs(full_path, exist_ok=True)

    # Save the model to the constructed path
    model_filename = os.path.join(full_path, f"model_{postfix_model}.joblib")
    dump(model, model_filename)
    ohe_filename = os.path.join(full_path, f"ohe.joblib")
    dump(ohe, ohe_filename)
    print(f"Model saved to: {model_filename}")


def load_master_model(master_path="."):
    """Func not needed anymore

    Args:
        master_path (str, optional): provide str. path to the Master_model you want to save. Defaults to '.'.

    Returns:
        models: list of the models loaded -> can be used to init the master class
        ohe: corresponding ohe
    """
    # Construct the filename for the model

    # Check if the model file exists
    if os.path.exists(master_path):
        # Load the model from the constructed path
        models = []
        for item in os.listdir(master_path):
            if item.startswith("ohe"):
                ohe = load(os.path.join(master_path, item))
            elif item.endswith("model_exported"):
                a = 0
                for model in os.listdir(os.path.join(master_path, item)):
                    models.append(load(os.path.join(master_path, item, model)))
                    a = a + 1
    else:
        print(f"Model file not found at: {master_path}")

    # Overwrite current model
    return models, ohe


class master_model_RFC:
    """creates a majority voting model out of 5 random forrest classifiers"""

    def __init__(self, list_of_models, loading_path=None, ohe=None):
        """use the models of the k-fold training to constuct a majority-vote-model or import a previously saved model
        Args:
            list_of_models (list): list/array with the models from the kfold cross training instance
        """
        if type(loading_path) is str:
            """load a previously run model"""
            self.models, self.ohe = load_master_model(loading_path)
            print("Loaded the master model from the given path")
        if type(list_of_models) == dict:
            self.models = [i.get("model") for i in list_of_models.values()]
            self.models_with_eids_of_datasets = list_of_models
        else:
            self.models_with_eids_of_datasets = None
            self.models = list_of_models
        if ohe != None:
            self.ohe = ohe

        print(f"Imported {len(list_of_models)} models for the mean voting")

    def predict_proba(self, X):
        """Get a mean probability prediction for the given DATA
        Args:
            X (array): ohe-encoded table X
        """
        predictions = []
        for model in self.models:
            pred_model = model.predict_proba(X)[:, 1]
            predictions.append(pred_model)
        return pd.DataFrame(data=predictions).transpose().mean(axis=1)

    def get_best_params(self):
        """returns a df of the parameters found for the models best fit

        Returns:
            df: with the k models and their choosen parameters
        """
        best_params = []
        for modeli in self.models:
            best_params.append(modeli.best_params_)
        return pd.DataFrame(best_params)

    def save(self, path, ohe):
        """Saving the models for majority vote model under the given path

        Args:
            path (path): path to save the models to
        """
        from joblib import dump
        from datetime import datetime

        def save_model(model, base_path, postfix_model=""):
            """Save a fitted model and the ohe

            Args:
                model (_type_): _description_
                ohe (_type_): _description_
                base_path (_type_): _description_
                postfix_model (_type_): _description_
            """
            from joblib import dump
            from datetime import datetime

            # Get the current date and construct the full path
            current_date = datetime.now().strftime("%Y_%m_%d")
            full_path = os.path.join(base_path, current_date + "_model_exported")

            # Create the subfolder if it doesn't exist
            os.makedirs(full_path, exist_ok=True)

            # Save the model to the constructed path
            model_filename = os.path.join(full_path, f"model_{postfix_model}.joblib")
            dump(model, model_filename)

        # save the ohe in the base_path
        current_date = datetime.now().strftime("%Y_%m_%d")

        ohe_filename = os.path.join(path, f"ohe_{current_date}.joblib")
        dump(ohe, ohe_filename)

        if self.models_with_eids_of_datasets != None:
            dict_filename = os.path.join(path, f"dict_{current_date}.joblib")
            dump(self.models_with_eids_of_datasets, dict_filename)

        for model, index in zip(self.models, np.arange(len(self.models))):
            save_model(model=model, base_path=path, postfix_model=str(index))

    def feature_importances_(self):
        """get the mean(feature importance) of the best estimator as a pd.Series"""
        export = pd.DataFrame()
        for model, name in zip(self.models, np.arange(len(self.models))):
            name = f"model_{str(name)}"
            feature_imp = model.best_estimator_.feature_importances_
            export[name] = feature_imp
        export["mean_feature_imp"] = export.mean(axis=1)
        export.set_axis(labels=self.ohe.get_feature_names_out().tolist())  # type: ignore
        return export

    def plot_rocs_test_train(self):
        print()
