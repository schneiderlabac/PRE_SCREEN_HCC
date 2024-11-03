import pandas as pd
import numpy as np
import os
from joblib import load
from joblib import dump
from datetime import datetime

# For RandomForestClassifier
from sklearn import neural_network
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression


def get_estimator(label, random_state=1):
    # Instantiate individual classifiers
    mlp_classifier = neural_network.MLPClassifier(
        hidden_layer_sizes=(10, 10), activation="relu", solver="adam", random_state=42
    )
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Create an ensemble of classifiers
    ensemble_classifier = VotingClassifier(
        estimators=[("mlp", mlp_classifier), ("rf", rf_classifier)],
        voting="soft",
    )

    ##### here you can edit the models to be called in the pipeline: dict of key -> that you have to give in the pipeline and the model you want to use
    models = {
        "RFC": RandomForestClassifier(class_weight="balanced_subsample", random_state=random_state),
        "XGB": GradientBoostingClassifier(learning_rate=0.1, random_state=random_state),
        "Log_l1": LogisticRegression(penalty="l1", solver="liblinear", random_state=random_state),
        "neuron": neural_network.MLPClassifier(
            hidden_layer_sizes=(
                100,
                50,
            ),
            activation="relu",
            solver="adam",
            random_state=random_state,
        ),
        "ensemble": ensemble_classifier,
    }
    return models[label]


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

        print(f"Imported {len(list_of_models)} models for the majority voting")

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
