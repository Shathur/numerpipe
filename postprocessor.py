import os
from pathlib import Path
import numerapi
import pandas as pd
from tqdm import tqdm
from preprocessor import Preprocessor
from predictions import get_predictions_per_era
from utils import get_era_idx
from feature_neutralization import neutralize, neutralize_short
from setup_env_variables import setup


class PostProcessor:
    def __init__(self, preprosessor: Preprocessor, models_path: str, prefix: str):
        self.preprocessor = preprosessor
        self.models_path = models_path
        self.prefix = prefix
        self.predictions_gathered_df = pd.DataFrame()
        self.napi = self.get_napi()

    def neutralize(
        self,
        df,
        columns,
        proportion=1,
        normalize=True,
        era_col="era",
        extra_neutralizers=[],
    ):
        """
        Calls either the neutralize or the neutralize_short from utils
        :param df: pd.DataFrame
        :param columns: list of features
        :param proportion: strength of neutralization
        :param normalize: this is needed if we have values in our columns that are negative
            the SVD will not converge in this situation
        :param era_col: name of each time period to group on when 'per_era' is used
        :param extra_neutralizers: a list of other vectors to neutralize against
        """
        return neutralize(
            df=df,
            columns=columns,
            proportion=proportion,
            normalize=normalize,
            era_col=era_col,
            extra_neutralizers=extra_neutralizers,
        )

    def get_keys(self):
        """get keys from global variables"""
        # setup env variables
        setup()
        # load keys from global environment
        public_id = os.getenv("PUBLIC_ID")
        secret_key = os.getenv("SECRET_KEY")
        return public_id, secret_key

    def get_napi(self):
        """get numerapi instance with specific keys"""
        public_id, secret_key = self.get_keys()
        return numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)

    def get_predictions_per_era(self, df, model_path, model_type):
        """Return predictions per era for our df"""
        return get_predictions_per_era(
            df=df[self.preprocessor.feature_cols],
            num_models=len(os.listdir(model_path)),
            prefix=self.prefix,
            folder_name=model_path,
            era_idx=get_era_idx(df),
            model_type=model_type,
            rank_average=False,
            verbose=False,
        )

    def add_predictions_per_era(
        self, df: pd.DataFrame(), model_name: str, model_type: str
    ):
        """predict and add prediction to dataframe"""
        predictions = self.get_predictions_per_era(
            df, os.path.join(self.models_path, model_name), model_type
        )
        self.predictions_gathered_df[model_name] = predictions

    def get_id_column(self, df):
        """return index id as a column"""
        if df.index.name == "id":
            # id_series = df.reset_index()['id']
            id_series = df.index  # much smaller memory blueprint
        else:
            raise ValueError("df should be a DataFrame with an index named 'id'")
        return id_series

    def save_predictions(self, path):
        """save predictions df"""
        if os.path.exists(path):
            self.predictions_gathered_df.to_csv(
                os.path.join(path, "predictions_gathered_df.csv")
            )
        else:
            raise FileNotFoundError(f"{path} is not a valid path")

    def submit(self, df, submission_name, submit_type, keep_diagnostics=True):
        """
        Submits either live either diagnostics
        The passed df is used only to decide on which indexes to keep.
        The predictions that are sent for validation come from
        self.predictions_gathered_df. PostProcessor.validation_df instances
        are used just for indices
        If diagnostics are submitted we assume that we have trained
        on eras before the tournament provided validation_df, and the df
        passed is the validation_df
        important to note that we have to pass a data file (*.csv) to
        napi.upload_predictions.
        """
        # get_keys
        public_key, secret_key = self.get_keys()
        # keep our model names as list to iterate on
        model_names = self.predictions_gathered_df.columns.tolist()
        # keep id as a column
        predictions_df = self.get_id_column(df).to_frame()
        # need to also reset the index
        predictions_df = predictions_df.reset_index(drop=True)
        if submit_type == "prediction":
            for name in tqdm(model_names):
                predictions_df["prediction"] = self.predictions_gathered_df[name]
                model_id = self.napi.get_models()[name]
                # Upload predictions
                predictions_df.to_csv(f"{submission_name}.csv", index=False)
                submission_id = self.napi.upload_predictions(
                    file_path=f"{submission_name}.csv", model_id=model_id
                )
                if not keep_diagnostics:
                    submission_name_path = Path(f"{submission_name}.csv")
                    submission_name_path.unlink()
        elif submit_type == "diagnostics":
            for name in tqdm(model_names):
                predictions_df["prediction"] = self.predictions_gathered_df[name]
                model_id = self.napi.get_models()[name]
                # Upload predictions
                predictions_df.to_csv(f"{submission_name}.csv", index=False)
                submission_id = self.napi.upload_diagnostics(
                    file_path=f"{submission_name}.csv", model_id=model_id
                )
                if not keep_diagnostics:
                    submission_name_path = Path(f"{submission_name}.csv")
                    submission_name_path.unlink()
        else:
            raise ValueError(
                "submit_type needs to be either 'prediction' or 'diagnostics'"
            )
