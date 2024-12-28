# Preprocessing
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from copy import deepcopy

# For OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

## basic plots
import seaborn as sns
import matplotlib.pyplot as plt

# TODO: adjust the adjust_ohe_mapping() function; now the df_snips is hardcoded! better solution!


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


def userpath(user: str, project: str = "cca"):
    """get the userpath from the user json

    Args:
        user (str): user key from the json file
        project (str, optional): what project dir you want to refer to. Defaults to "cca".

    Returns:
        str: path you want to have
    """
    print(f"Debug: userpath called with user={user}, project={project}")  # Debug print

    # Path to the current file
    current_file = Path(__file__).resolve()
    # Path to the JSON file
    home = current_file.parent
    try:
        with open(home / "user_settings.json") as file:
            data = json.load(file)
        print(f"Debug: Loaded user_settings.json: {data}")  # Debug print
        if user in data and project in data[user]:
            output = data[user][project]
            print(f"Debug: Found path for user {user} and project {project}: {output}")  # Debug print
        else:
            raise KeyError(f"User '{user}' or project '{project}' not found in user_settings.json")
    except Exception as e:
        print(f"Error loading user settings: {e}")
        output = str(e)
    return output


def load_data(subset, test_cohort = "val"):
    """
    Load the specified dataset separated into different DataFrames.
    Checks for both CSV and Excel files.

    Args:
        subset (str): String specifying the subsetting
        test_cohort: Defines whether you will evaluate on the test or on an (external) val dataset. For test, no _outer dfs are needed

    Returns:
        X_inner, y_inner, X_outer, y_outer: as pandas.DataFrames
    """
    filepaths = {
        "inner": {"X": f"X_inner_{subset}", "y": f"y_inner_{subset}"},
        "outer": {"X": f"X_outer_{subset}", "y": f"y_outer_{subset}"},
    }

    def load_file(base_name):
        csv_path = f"{base_name}.csv"
        excel_path = f"{base_name}.xlsx"

        if os.path.exists(csv_path):
            return pd.read_csv(csv_path, index_col=0)
        elif os.path.exists(excel_path):
            return pd.read_excel(excel_path, index_col=0)
        else:
            raise FileNotFoundError(f"Neither CSV nor Excel file found for {base_name}")

    X_inner = load_file(filepaths["inner"]["X"])
    y_inner = load_file(filepaths["inner"]["y"])
    X_inner = ensure_split_int(X_inner)

    if test_cohort == "test":
        return X_inner, y_inner, None, None
    elif test_cohort == "val":
        X_outer = load_file(filepaths["outer"]["X"])
        y_outer = load_file(filepaths["outer"]["y"])
        return X_inner, y_inner, X_outer, y_outer
    else:
        raise ValueError("test_cohort must be either 'test' or 'val'")


def ensure_split_int(X_inner, seed=42):
    """
    Check if 'split_int' column is in X_inner. If not, add it with values from 1 to 5.

    Args:
        X_inner (pd.DataFrame): The input dataframe
        seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: X_inner with 'split_int' column assured
    """
    if 'split_int' not in X_inner.columns:
        np.random.seed(seed)
        X_inner['split_int'] = np.random.randint(1, 6, size=len(X_inner))
        print("'split_int' column added to X_inner with random values from 1 to 5.")
    else:
        print("'split_int' column already exists in X_inner.")

    return X_inner

def summarize_df(X, y, X_val, y_val, col_subset, row_subset, DOI):
    """
    Print summary statistics of the provided dataframes and arrays.

    Parameters:
    - X: DataFrame or array for training features.
    - y: Series or array for training labels.
    - X_val: DataFrame or array for validation features.
    - y_val: Series or array for validation labels.
    """
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    print(f"Count of {DOI} in y:", (y == 1).sum())


    if X_val is not None and y_val is not None:
        print("Shape of X_val:", X_val.shape)
        print("Shape of y_val:", y_val.shape)
        print(f"Count of {DOI} in y_val:", (y_val == 1).sum())
    else:
        print("No validation data provided.")
    print(f"Active Model: {col_subset}; Cohort: {row_subset}")


def fit_ohe(pip_data):
    column_names = pip_data.X.columns
    column_classes = pip_data.X.dtypes

    variables = pd.DataFrame({"Column Name": column_names, "Data Class": column_classes})
    object_list = variables[variables["Data Class"] == "object"]["Column Name"].tolist()

    # One Hot Encoder encodes categorical variables into separate columns
    ct = ColumnTransformer(
        [
            (
                "one_hot_encoder",
                OneHotEncoder(categories="auto"),
                object_list,
            )
        ],  # Telling One Hot Encoder which columns to encode (all with categorical variables)
        remainder="passthrough",  # Leave the rest of the columns untouched
    )

    return ct.fit(pip_data.X)


def adjust_ohe_mapping(
    X_ohe: pd.DataFrame, columngroups_df: pd.DataFrame, prefix_ohe: str, prefix_remain: str
) -> pd.DataFrame:
    """
    adjust the mapping df for X for the one hot encoded X with the adjusted column titles

    Args:
        object_cols (list): list of columns that were passed into ohe
        X_ohe (df): oh encoded X array as pd.DataFrame
        columngroups_df (df): columngroups for X without ohe -> col titles are "column name" and "source_df"
        prefix_ohe = (str): str used as prefix for ohe features
        prefix_remain = (str): str used as prefix for the passed features

        CAVE: an encoded column starting with 'rs' (as normally the SNIPs) will be labeled as df_snp
    Returns:
    adjusted column mapper as df with 'column_name' as index and 'source' and 'name_print' (lstriped the prefixes) as columns
    """
    columngroups_X = dict(zip(columngroups_df["column_name"], columngroups_df["source_df"]))
    column_name = []
    source_df = []
    # construct a new mapping dataframe for the cols of X_ohe columns with the sources

    for el in X_ohe.columns.tolist():
        if el.startswith(prefix_ohe):
            if el.startswith(prefix_ohe + "rs"):
                source = "df_snp"
            else:
                source = columngroups_X.get(el.split("__")[1].split("_")[0])
        else:
            source = columngroups_X.get(el.split("__")[1])
        source_df.append(source)
        column_name.append(el)
    col_mapper_ohe = pd.DataFrame(data=[column_name, source_df], index=["column_name", "source"]).transpose()
    col_mapper_ohe["name_print"] = col_mapper_ohe.column_name.str.replace(prefix_ohe, "").str.replace(prefix_remain, "")
    col_mapper_ohe.set_index("column_name", inplace=True)
    return col_mapper_ohe


def correlation_matrix_plot(
    pip_data,
    subsets_startwith=[
        "all",
        "one_hot_encoder__",
        "remainder__",
    ],
    subset_isin=[],
    print_columns=False,
):
    """_summary_

    Args:
        pip_data (_type_): _description_
        subsets_startwith (list, optional): _description_. Defaults to [ "all", "one_hot_encoder__", "remainder__", ] set to [None] to use the isin.
        subset_isin (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """
    axes = []
    for (
        subseting
    ) in subsets_startwith:  #  'one_hot_encoder__','remainder__' one_hot_encoder__ or remainder__ or all to use all
        df = deepcopy(pip_data.X_ohe_df)
        if print_columns:
            return df.columns
        if subseting != "all":
            if subseting is None:
                df = df.loc[:, df.columns.isin(subset_isin)]
            else:
                df = df.loc[:, list(df.columns.str.startswith(subseting))]

        df.columns = df.columns.str.removeprefix("remainder__")
        df.columns = df.columns.str.removeprefix("one_hot_encoder__")

        sns.set_theme(style="dark")
        d = df

        # Compute the correlation matrix
        corr = d.corr()

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(20, 20))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        ax = sns.heatmap(
            corr, mask=mask, cmap=cmap, vmax=0.3, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}
        )
        axes.append(ax)
    return axes


class pp_user_input:
    def __init__(self, project_vars: dict):
        self.project = project_vars.get('project', 'cca')
        self.path = userpath(os.environ.get("USER", os.environ.get("USERNAME")), project=self.project)  # type: ignore
        self.data_path = (
            f"{self.path}/data/"
            + project_vars[
                "export_date_data"
            ]  ########## Change this date depending on the data version you want to use
        )
        self.fig_path = f"{self.path}/visuals"
        self.model_path = f"{self.path}/Models"
        os.chdir(self.data_path)

        # update the user_input variables stored with the objekt
        for key, value in project_vars.items():
            setattr(self, key, value)

        ##################################################################################################################################
        # Define Description variables based on
        if self.col_subset in [  # type: ignore
            "Model_A",
            "Model_B",
            "Model_C",
            "Model_Csmall",
            "Model_AMAP-RFC",
            "Model_D",
            "Model_Demographics",
            "Model_Diagnosis",
            "Model_Blood",
            "Model_SNP",
            "Model_TOP75",
            "Model_TOP30",
            "Model_TOP15",
            "Model_FACS"

        ]:  # Change import_col_subset according to chosen model
            self.import_col_subset = "basic"
        elif self.col_subset in ["Model_E", "Model_Metabolomics"]:  # type: ignore
            self.import_col_subset = "met"
        else:
            self.import_col_subset = None
            print("Unexpected col_subset value.")

        self.subset = f"{self.import_col_subset}_{self.row_subset}"  # Combining the subsets# type: ignore

        if self.row_subset == "par":  # type: ignore  # Change this for exporting a "nicer description" in visuals later on
            self.row_subset_long = "Patients at risk"
        elif self.row_subset == "all":  # type: ignore
            self.row_subset_long = "All"
        elif self.row_subset == "par_Cirrhosis":  # type: ignore
            self.row_subset_long = "Liver cirrhosis"
        else:
            self.row_subset_long = "Unknown"
        ##################################################################################################################################



class data:

    def __init__(
        self,
        user_input,
        col_subsets_options={
            "Model_A": ["df_metadata", "df_covariates"],
            "Model_B": ["df_metadata", "df_covariates", "df_diagnosis"],
            "Model_C": ["df_metadata", "df_covariates", "df_diagnosis", "df_blood",],
            "Model_D": ["df_metadata", "df_covariates", "df_diagnosis", "df_blood", "df_snp"],
            "Model_E": ["df_metadata", "df_covariates", "df_diagnosis", "df_blood", "df_snp", "df_metabolomics"],
            "Model_Demographics": ["df_metadata", "df_covariates"],
            "Model_Diagnosis": ["df_metadata", "df_diagnosis"],
            "Model_Blood": ["df_metadata", "df_blood"],
            "Model_SNP": ["df_metadata", "df_snp"],
            "Model_Metabolomics": ["df_metadata", "df_metabolomics"],
            "Model_TOP75": ["df_metadata", "df_covariates", "df_diagnosis", "df_blood",],
            "Model_TOP30": ["df_metadata", "df_covariates", "df_diagnosis", "df_blood",],
            "Model_TOP15": ["df_metadata", "df_covariates", "df_diagnosis", "df_blood",],
            "Model_AMAP-RFC": ["df_metadata", "df_covariates", "df_diagnosis", "df_blood",],
            "Model_FACS": ["df_metadata", "df_blood", "df_condensed"],
        },
        dict_reduce_columns = {
            "Model_TOP15": "top15",
            "Model_TOP30": "top30",
            "Model_TOP75": "top75",
            'Model_AMAP-RFC': "keep_amap"
        },
        columngroups_path="columngroups",
        group_to_predict="Hepatobiliary cancer",
    ):

        default_columns_to_remove = ["date_of_diag", "assessment", "difftime", "status_cancerreg"]
        default_columns_to_transfer = ["Date of assessment", "Ethnicity", "Handgripstrength", "UKB assessment centre",
            "BMI_cat", "AGE_cat" #"PC1", "PC2", "PC3", "PC4", "PC5",
        ]
        self.user_input = user_input
        self.col_subsets_options = col_subsets_options
        self.dict_reduce_columns = dict_reduce_columns
        self.columngroups_path = columngroups_path
        self.group_to_predict = group_to_predict
        self.columns_to_remove = getattr(user_input, "columns_to_remove", default_columns_to_remove)
        self.columns_to_transfer = getattr(user_input, "columns_to_transfer", default_columns_to_transfer)

        if self.user_input.col_subset in self.dict_reduce_columns:
            self.user_input.reduce_columns = self.dict_reduce_columns[self.user_input.col_subset]
        else:
            self.user_input.reduce_columns = None


        X, y, X_val, y_val = load_data(user_input.subset, test_cohort=user_input.test_cohort) # Loads csv files into dataframes

        # Stores files in pl.data class
        self.y_orig=y.copy()
        self.y_val_orig=y_val.copy()

        self.columns_to_remove = [col for col in self.columns_to_remove if col in X.columns]  #adapt vector to present columns
        self.columns_to_transfer = [col for col in self.columns_to_transfer if col in X.columns]

        # Select for y only the column predefined with "target" in user_input
        if self.user_input.target in y.columns:
            y = y[[self.user_input.target]]
        else:
            raise ValueError(f"Target column '{self.user_input.target}' not found in y DataFrame")

        # For y_val (if it exists)
        if y_val is not None:
            if self.user_input.target in y_val.columns:
                y_val = y_val[[self.user_input.target]]
            else:
                raise ValueError(f"Target column '{self.user_input.target}' not found in y_val DataFrame")

        # Optionally, you might want to rename the column to a standard name like 'target' or 'label'
        y = y.rename(columns={self.user_input.target: 'status'})
        if y_val is not None:
            y_val = y_val.rename(columns={self.user_input.target: 'status'})

        # Store the processed dataframes
        self.y = y
        if y_val is not None:
            self.y_val = y_val

        if X_val is not None:
            X_val["split_int"] = np.zeros(len(X_val)).tolist()
        summarize_df(
            X, y, X_val, y_val, col_subset=user_input.col_subset, row_subset=user_input.row_subset, DOI=user_input.DOI
        )  # print summary values (make sure to check these!)



        # Subset dataframes according to the meta-info in columngroups_df
        def load_columngroups(file_path):
            """
            Load the columngroups data from either a CSV or Excel file.

            Args:
                file_path (str): Path to the columngroups file, without file extension

            Returns:
                pandas.DataFrame: The loaded columngroups data
            """
            csv_path = f"{file_path}.csv"
            excel_path = f"{file_path}.xlsx"

            print(f"Attempting to load columngroups from:")
            print(f"CSV path: {os.path.abspath(csv_path)}")
            print(f"Excel path: {os.path.abspath(excel_path)}")

            if os.path.exists(csv_path):
                print(f"Loading CSV file from {csv_path}")
                return pd.read_csv(csv_path)
            elif os.path.exists(excel_path):
                print(f"Loading Excel file from {excel_path}")
                return pd.read_excel(excel_path)
            else:
                print(f"Current working directory: {os.getcwd()}")
                print(f"Files in current directory: {os.listdir('.')}")
                raise FileNotFoundError(f"Neither CSV nor Excel file found for columngroups at {file_path}")


        columngroups_df = load_columngroups(columngroups_path)
        columngroups_df = columngroups_df[columngroups_df["column_name"] != group_to_predict]  # Remove group that should be predicted
        relevant_data_sources = col_subsets_options[user_input.col_subset]
        columngroups_df = columngroups_df[columngroups_df["source_df"].isin(relevant_data_sources)]
        relevant_columns = columngroups_df["column_name"].tolist()  # Extract the relevant columns based on the filtered columngroups_df
        relevant_columns_X = [col for col in relevant_columns if col != "split_ext"]  # remove split_ext for the training data
        X = X[[col for col in relevant_columns_X if col in X.columns]]

        if X_val is not None:
            X_val = X_val[[col for col in relevant_columns if col in X_val.columns]]

        # Drop columns that are duplicated or rows that are NAN
        X = X.loc[:, ~X.columns.duplicated()]
        y = y.loc[(pd.isna(X).sum(axis=1) == 0), :]
        X.dropna(axis=0, how="any", inplace=True)

        ## create a table Z with additional information, which includes further info like date_of_diag that needs to be taken along, but not as input for the model
        # 1st take all info from y
        z = self.y_orig.copy()
        if self.y_val_orig is not None:
            z_val = self.y_val_orig.copy()

        # now all info from X
        z[self.columns_to_transfer] = X[self.columns_to_transfer]
        if X_val is not None:
            z_val[self.columns_to_transfer] = X_val[self.columns_to_transfer]



        # Remove all unncessary columns from y. y shall only contain eids and your target as boolean
        y = y.drop(columns=[col for col in self.columns_to_remove if col in y.columns])
        if y_val is not None:
            y_val = y_val.drop(columns=[col for col in self.columns_to_remove if col in y_val.columns])


        # Remove predefined columns from X
        X.drop(columns=self.columns_to_transfer, inplace=True)
        if X_val is not None:
            X_val.drop(columns=self.columns_to_transfer, inplace=True)



        ######################################################
        ##### Reduction of dataframes for ablation study #####
        #load excel with options on how to reduce columns for dataframe
        self.reduce_df = self.load_reduce_columns()

        # Apply column reduction if specified
        if user_input.reduce_columns is not None:
            X, X_val = self.reduce_columns(X, X_val)


        def update_dataframe(df):
            # Check if "SEX" is in the DataFrame and convert
            if "SEX" in df.columns:
                df["SEX"] = df["SEX"].replace({"Female": 0, "Male": 1}).astype(int)

            # Convert "Path_Alk" and "High_Alk" to string, handling if they do not exist
            for column in ["Path_Alk", "High_Alk", "Ever smoked"]:
                if column in df.columns:
                    df[column] = df[column].astype(str)
                    print(f'Column "{column}" adapted to string')
                else:
                    print(f'Column "{column}" not in imported columns')

            return df

        def print_mixed_types(df):
            """
            Print columns with mixed types in a DataFrame.

            Args:
            - df (DataFrame): The DataFrame to check for mixed data types.
            """
            mixed_types = {}

            # Iterate through columns and collect all unique types
            for col in df.columns:
                types = df[col].apply(lambda x: type(x).__name__).unique()  # Get the type names for uniqueness
                if len(types) > 1:
                    mixed_types[col] = types  # Store in dict if more than one type is detected

            # Print the columns with mixed data types
            for col, types in mixed_types.items():
                print(f"{col}: {types}")

        print("\n")
        print_mixed_types(X)
        if X_val is not None:
            print_mixed_types(X_val)

        X = update_dataframe(X)
        if X_val is not None:
            X_val = update_dataframe(X_val)

        # set Alkohol columns to object type
        # try:
        #    X_val.loc[:, ["Path_Alk", "High_Alk"]] = X_val.loc[:, ["Path_Alk", "High_Alk"]].astype(str)
        #    X.loc[:, ["Path_Alk", "High_Alk"]] = X.loc[:, ["Path_Alk", "High_Alk"]].astype(str)
        # except Exception as e:
        #    print(f'no ["Path_Alk", "High_Alk"] in the inported columns: {e}')

        print("\n\nSummary after subsetting:")
        summarize_df(
            X, y, X_val, y_val, col_subset=user_input.col_subset, row_subset=user_input.row_subset, DOI=user_input.DOI
        )  # print summary values (make sure to check these!)

        ## Save the arrays to the object
        self.X = X
        self.y = y
        self.z = z
        self.columngroups_df = columngroups_df
        if X_val is not None:
            self.X_val = X_val
            self.y_val = y_val
            self.z_val = z_val

    def load_reduce_columns(self):
            """
            Loads the df that defines for your specific model which of the available features to use.
            E.g. if you want to train only on the TOP15 predefined features, select those in advance in the excel file (1=yes, 0=no).
            Subsetting happens in "reduce_columns".
            """
            reduce_columns_path = os.path.join(self.user_input.path, 'data', 'reduce_columns.xlsx')
            if os.path.exists(reduce_columns_path):
                return pd.read_excel(reduce_columns_path)
            else:
                print(f"Warning: reduce_columns.xlsx not found at {reduce_columns_path}")
                return None


    def reduce_columns(self, X, X_val):
        if self.reduce_df is not None and self.user_input.reduce_columns in self.reduce_df.columns:
            keep_column = self.user_input.reduce_columns
            if keep_column in self.reduce_df.columns:
                features_to_keep = self.reduce_df[self.reduce_df[keep_column] == 1]['Feature'].tolist()
                X = X[X.columns.intersection(features_to_keep + ['split_int'])]
                X_val = X_val[X_val.columns.intersection(features_to_keep + ['split_int', 'split_ext', "Date of assessment"])]
                print(f"Columns reduced based on '{keep_column}' in reduce_columns.xlsx")
            else:
                print(f"Warning: Column '{keep_column}' not found in reduce_columns.xlsx. No columns were reduced.")
        else:
            print(f"Warning: No columns were reduced.")

        return X, X_val



    def andjust_enc_X_and_map(self, ohe):
        # construct a df for the OH encoded features -> with interpretable columnnames
        try:
            X_ohe_df = pd.DataFrame(np.array(ohe.transform(self.X)), columns=ohe.get_feature_names_out())
            self.X_ohe_df = X_ohe_df
        except Exception as e:
            print(f"No ohe found: {e}")

        try:
            X_ohe_map = adjust_ohe_mapping(
                X_ohe=X_ohe_df,
                columngroups_df=self.columngroups_df,
                prefix_ohe="one_hot_encoder__",
                prefix_remain="remainder__",
            )
            self.X_ohe_map = X_ohe_map
        except Exception as e:
            print(f"The adjustment of the mapping did not work: {e}")


class mapper:

    def __init__(self, pip_self):
        def hex_to_rgb(hex_color):
            # Remove '#' if present
            if hex_color.startswith("#"):
                hex_color = hex_color[1:]

            # Convert hex to RGB
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)

            return (r / 255, g / 255, b / 255)

        self.color_groups = pip_self.user_input.color_groups_all
        if list(self.color_groups.values())[0].startswith("#"):
            self.color_groups_rgb = {key: hex_to_rgb(value) for key, value in self.color_groups.items()}
        self.color_groups_violin = pip_self.user_input.color_groups_violin
