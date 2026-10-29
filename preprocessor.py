import sys
import os
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import numerapi
import pandas as pd
import json
import gc
from cv_splits import (
    cv_split_creator,
    cross_validate_train,
    TimeSeriesSplitGroupsPurged,
)


class Preprocessor:

    def __init__(self, datapath, n_splits, target):
        self.napi = numerapi.NumerAPI()
        self.datapath = datapath
        self.train_df = pd.DataFrame()
        self.validation_df = (pd.DataFrame(),)
        self.test_df = pd.DataFrame()
        self.train_neutralizer_df = pd.DataFrame()
        self.validation_neutralizer_df = pd.DataFrame()
        self.live_neutralizer_df = pd.DataFrame()
        self.feature_cols = []
        self.target = target
        self.n_splits = n_splits
        self.cv_split_data: []

    def download_data(
        self,
        data_version="quantum",
        train=False,
        validation=False,
        live=False,
        live_example_preds=False,
        validation_example_preds=False,
        live_example_preds_csv=False,
        validation_example_preds_csv=False,
        features=False,
        # meta_model=False,
        train_benchmark_models=False,
        validation_benchmark_models=False,
        live_benchmark_models=False,
        train_neutralizer=False,
        validation_neutralizer=False,
        live_neutralizer=False,
    ):
        """
        Updated for 6.0 data. For more info for the various downloads here visit
        https://numer.ai/data
        NOTE: v5.0-v5.3 train/validation files are retired on Numerai (live only
        until 2026-11-06). v6.0 ("unity") changes the schema: the
        era column is "date" and the row identifier is "numerai_ticker".
        get_data() normalizes these back to "era"/"id" for the rest of the pipeline.
        Targets are the Jupiter family (target aliases target_jupiter_60);
        the v5 target_ender_60 etc no longer exist.
        """
        dv_list = [
            "atlas",
            "5.0",
            "faith",
            "5.1",
            "faith2",
            "5.2",
            "quantum",
            "5.3",
            "unity",
            "6.0",
        ]
        if data_version == "atlas" or data_version == "5.0":
            dv = "5.0"
        elif data_version == "faith" or data_version == "5.1":
            dv = "5.1"
        elif data_version == "faith2" or data_version == "5.2":
            dv = "5.2"
        elif data_version == "quantum" or data_version == "5.3":
            dv = "5.3"
        elif data_version == "unity" or data_version == "6.0":
            dv = "6.0"
        if data_version not in dv_list:
            raise ValueError(f"data_version should have one of the values: {dv_list}")
        needs_neutralizer = any(
            [train_neutralizer, validation_neutralizer, live_neutralizer]
        )
        if needs_neutralizer and dv != "6.0":
            raise ValueError(
                "neutralizer files only exist for data_version 6.0 (unity) and later"
            )
        if train:
            self.napi.download_dataset(
                f"v{dv}/train.parquet",
                os.path.join(self.datapath, f"train.parquet"),
            )
        if validation:
            self.napi.download_dataset(
                f"v{dv}/validation.parquet",
                os.path.join(self.datapath, f"validation.parquet"),
            )
        if live:
            self.napi.download_dataset(
                f"v{dv}/live.parquet",
                os.path.join(self.datapath, f"live.parquet"),
            )
        if validation_example_preds:
            self.napi.download_dataset(
                f"v{dv}/validation_example_preds.parquet",
                os.path.join(self.datapath, "validation_example_preds.parquet"),
            )
        if validation_example_preds_csv:
            self.napi.download_dataset(
                f"v{dv}/validation_example_preds.csv",
                os.path.join(self.datapath, "validation_example_preds.csv"),
            )
        if live_example_preds:
            self.napi.download_dataset(
                f"v{dv}/live_example_preds.parquet",
                os.path.join(self.datapath, "live_example_preds.parquet"),
            )
        if live_example_preds_csv:
            self.napi.download_dataset(
                f"v{dv}/live_example_preds.csv",
                os.path.join(self.datapath, "live_example_preds.csv"),
            )
        if features:
            self.napi.download_dataset(
                f"v{dv}/features.json", os.path.join(self.datapath, "features.json")
            )
        # if meta_model:
        #     self.napi.download_dataset(
        #         f"v{dv}/meta_model.parquet",
        #         os.path.join(self.datapath, "meta_model.parquet"),
        #     )
        if live_benchmark_models:
            self.napi.download_dataset(
                f"v{dv}/live_benchmark_models.parquet",
                os.path.join(self.datapath, "live_benchmark_models.parquet"),
            )
        if validation_benchmark_models:
            self.napi.download_dataset(
                f"v{dv}/validation_benchmark_models.parquet",
                os.path.join(self.datapath, "validation_benchmark_models.parquet"),
            )
        if train_benchmark_models:
            self.napi.download_dataset(
                f"v{dv}/train_benchmark_models.parquet",
                os.path.join(self.datapath, "train_benchmark_models.parquet"),
            )
        if train_neutralizer:
            self.napi.download_dataset(
                f"v{dv}/train_neutralizer.parquet",
                os.path.join(self.datapath, "train_neutralizer.parquet"),
            )
        if validation_neutralizer:
            self.napi.download_dataset(
                f"v{dv}/validation_neutralizer.parquet",
                os.path.join(self.datapath, "validation_neutralizer.parquet"),
            )
        if live_neutralizer:
            self.napi.download_dataset(
                f"v{dv}/live_neutralizer.parquet",
                os.path.join(self.datapath, "live_neutralizer.parquet"),
            )

    @staticmethod
    def _normalize_unity_columns(df):
        """
        v6.0 (unity) schema uses "date" for the era column and
        "numerai_ticker" as the row identifier. The rest of numerpipe
        (cv_splits, utils, PostProcessor, models_code scripts) expects
        "era"/"id", so rename them back here.
        """
        if df is None:
            return df
        rename_map = {}
        if "date" in df.columns and "era" not in df.columns:
            rename_map["date"] = "era"
        if "numerai_ticker" in df.columns and "id" not in df.columns:
            rename_map["numerai_ticker"] = "id"
        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    def get_data(
        self,
        train=False,
        validation=False,
        live=False,
        merge=False,
        train_neutralizer=False,
        validation_neutralizer=False,
        live_neutralizer=False,
    ):
        if (os.path.exists(os.path.join(self.datapath, f"train.parquet"))) and train:
            self.train_df = pd.read_parquet(
                os.path.join(self.datapath, f"train.parquet")
            )
            self.train_df = self._normalize_unity_columns(self.train_df)
        if (
            os.path.exists(os.path.join(self.datapath, f"validation.parquet"))
        ) and validation:
            self.validation_df = pd.read_parquet(
                os.path.join(self.datapath, f"validation.parquet")
            )
            self.validation_df = self._normalize_unity_columns(self.validation_df)
        if (os.path.exists(os.path.join(self.datapath, f"live.parquet"))) and live:
            self.live_df = pd.read_parquet(os.path.join(self.datapath, f"live.parquet"))
            self.live_df = self._normalize_unity_columns(self.live_df)
        if merge:
            self.train_df = pd.concat([self.train_df, self.validation_df])
        if (
            os.path.exists(os.path.join(self.datapath, "train_neutralizer.parquet"))
        ) and train_neutralizer:
            self.train_neutralizer_df = pd.read_parquet(
                os.path.join(self.datapath, "train_neutralizer.parquet")
            )
            self.train_neutralizer_df = self._normalize_unity_columns(
                self.train_neutralizer_df
            )
        if (
            os.path.exists(
                os.path.join(self.datapath, "validation_neutralizer.parquet")
            )
        ) and validation_neutralizer:
            self.validation_neutralizer_df = pd.read_parquet(
                os.path.join(self.datapath, "validation_neutralizer.parquet")
            )
            self.validation_neutralizer_df = self._normalize_unity_columns(
                self.validation_neutralizer_df
            )
        if (
            os.path.exists(os.path.join(self.datapath, "live_neutralizer.parquet"))
        ) and live_neutralizer:
            self.live_neutralizer_df = pd.read_parquet(
                os.path.join(self.datapath, "live_neutralizer.parquet")
            )
            self.live_neutralizer_df = self._normalize_unity_columns(
                self.live_neutralizer_df
            )

    def get_features(self, feature_group, include_country=False):
        """
        features_json contains:
            feature_sets
                small (101), medium (351), all (1040), signals (23)
            targets
                the v6 Jupiter family; 'target' is an alias of target_jupiter_60
        NOTE for v6.0 (unity):
            - sets like v2_equivalent_features, fncv3_features, sunshine, rain
              and the per-attribute sets no longer exist
            - feature_stats is no longer published
            - every set includes "feature_country" which is TEXT (country
              codes); excluded by default since numeric models (xgb/lgb)
              would break on it - pass include_country=True to keep it
            - the 1017 new v6 features use bins 0-4 (NaN -> 2); the 22
              restored signals numeric features keep their values and can
              hold NaNs
        """
        possible_feature_list = [
            "small",
            "medium",
            "all",
            "v2_equivalent_features",
            "v3_equivalent_features",
            "fncv3_features",
            "intelligence",
            "charisma",
            "strength",
            "dexterity",
            "constitution",
            "wisdom",
            "agility",
            "serenity",
            "sunshine",
            "rain",
            "signals",
        ]
        assert os.path.exists(
            f"{self.datapath}/features.json"
        ), "features_json does not exist, need to download it first"
        assert (
            feature_group in possible_feature_list
        ), f"unavailable feature_set name -- use one of the following {possible_feature_list}"

        f = open(f"{self.datapath}/features.json")
        features_json = json.load(f)
        self.feature_cols = features_json["feature_sets"][feature_group]

        # v6.0: feature_country is a text (country code) column - numeric
        # models (xgb/lgb) break on it, so drop it unless explicitly kept
        if not include_country and "feature_country" in self.feature_cols:
            self.feature_cols = [
                c for c in self.feature_cols if c != "feature_country"
            ]

    def per_era_correlations(self, df, features, era_col, target_col):
        """Get the correlation of each era with the designated target"""
        all_feature_corrs = df.groupby(era_col).apply(
            lambda era: era[features].corrwith(era[target_col])
        )
        return all_feature_corrs

    def get_riskiest_features(self, per_era_corrs, num_of_features):
        """per_era_correlations function is designed to return the input for here"""
        sorted_eras = per_era_corrs.index.sort_values()
        first_half = per_era_corrs[: len(per_era_corrs) // 2]
        second_half = per_era_corrs[len(per_era_corrs) // 2 :]
        first_half_means = per_era_corrs.loc[first_half.index, :].mean()
        second_half_means = per_era_corrs.loc[second_half.index, :].mean()
        per_era_corrs_diff = first_half_means - second_half_means
        sorted_diffs = per_era_corrs_diff.abs().sort_values(ascending=False)
        worst_n = sorted_diffs.head(num_of_features).index.tolist()
        return sorted_diffs, worst_n

    def get_cv(self, n_splits=None, embg_grp_num=12):
        """
        gets cv_split_data object. Defaut behaviour is passing the n_splits
        in the class Constructor. We may override it here if we want
        """
        if n_splits == None:
            n_splits = self.n_splits
        self.cv_split_data = cv_split_creator(
            df=self.train_df,
            col="era",
            cv_scheme=TimeSeriesSplitGroupsPurged,
            n_splits=n_splits,
            extra_constructor_params={"embg_grp_num": embg_grp_num},
        )

    def get_test_data(self, num_tour_eras):
        unique_train_eras = self.train_df.columns.tolist()
        unique_test_eras = unique_train_eras[-num_tour_eras:]
        self.test_df = self.train_df[self.train_df.isin(unique_test_eras)]
        self.train_df = self.train_df[~self.train_df.isin(unique_test_eras)]

    def save_custom_era_split(
        self,
        filename="full_data.parquet",
        erasplit=[0, -500, -300, -1],
        save_filenames=[
            "new_train.parquet",
            "new_test.parquet",
            "new_validation.parquet",
        ],
        drop_nan_target_eras=True,
    ):
        """
        loads the dataframe of the provided filename and then splits it into train and
        test data.
        erasplit : [train_start, train_end, test_start, test_end, validation_start, validation_end]
        save_filenames : [train_path, test_path, validation_path]
        drop_nan_target_eras : eras where the designated target (self.target) has
            NaN values (tail eras whose target horizon is not yet filled) are
            removed from the TRAIN split before saving, so new_train_df can
            never contain eras with a non-existent target. Test and
            validation splits are left untouched - NaN targets there are
            acceptable. The split indices themselves are still resolved on
            the full era list, so the erasplit boundaries keep their
            meaning (only trailing/uncovered eras get dropped, which is
            exactly the NaN-target tail).

        Memory blueprint: the full dataframe is never loaded. Only the
        "era" + target columns are read to detect the NaN-target eras, and
        every output file is written by streaming the parquet through a
        pyarrow dataset filter, so only the matching rows of one split are
        ever resident in memory (no boolean-mask copies of the whole frame).
        """
        src_path = os.path.join(self.datapath, filename)

        # resolve target column(s); self.target may be a string or a list
        targets = (
            [t for t in self.target if t is not None]
            if isinstance(self.target, list)
            else [self.target]
            if self.target is not None
            else []
        )

        schema_names = pq.ParquetFile(src_path).schema_arrow.names
        target_cols = [t for t in targets if t in schema_names]
        for t in targets:
            if t not in schema_names:
                print(
                    f"save_custom_era_split: WARNING target column '{t}' not "
                    f"found in {filename}, skipping NaN check for it"
                )

        # ------------------------------------------------------------------
        # 1. read ONLY the era column to resolve the split era lists
        # ------------------------------------------------------------------
        era_tbl = pq.read_table(src_path, columns=["era"])
        unique_eras = pd.unique(era_tbl.column("era").to_pandas()).tolist()
        del era_tbl
        gc.collect()

        train_eras = unique_eras[erasplit[0] : erasplit[1]]
        test_eras = unique_eras[erasplit[2] : erasplit[3]]
        validation_eras = unique_eras[erasplit[4] : erasplit[5]]
        split_eras = [train_eras, test_eras, validation_eras]

        # ------------------------------------------------------------------
        # 2. read ONLY era + target columns, find eras with NaN target
        # ------------------------------------------------------------------
        if drop_nan_target_eras and target_cols:
            tgt_tbl = pq.read_table(src_path, columns=["era"] + target_cols)
            era_np = tgt_tbl.column("era").to_numpy(zero_copy_only=False)
            bad_mask = np.zeros(len(tgt_tbl), dtype=bool)
            for t in target_cols:
                t_np = tgt_tbl.column(t).to_numpy(zero_copy_only=False)
                if t_np.dtype != np.float64:
                    t_np = t_np.astype(np.float64)
                n_nan = int(np.isnan(t_np).sum())
                if n_nan:
                    print(
                        f"save_custom_era_split: '{t}' has {n_nan} NaN "
                        f"values in {filename}"
                    )
                bad_mask |= np.isnan(t_np)
                del t_np
            bad_eras = set(pd.unique(era_np[bad_mask]).tolist())
            del tgt_tbl, era_np, bad_mask
            gc.collect()
            if bad_eras:
                print(
                    f"save_custom_era_split: dropping {len(bad_eras)} eras "
                    f"where the designated target has NaN values (train only)"
                )
                kept_before = len(train_eras)
                train_eras[:] = [e for e in train_eras if e not in bad_eras]
                print(
                    f"  -> {save_filenames[0]}: kept {len(train_eras)}/"
                    f"{kept_before} train eras (dropped "
                    f"{kept_before - len(train_eras)} NaN-target eras)"
                )
                print(
                    "  -> test and validation splits are left untouched; "
                    "NaN targets there are acceptable"
                )

        # ------------------------------------------------------------------
        # 3. never write an empty split (especially the train file)
        # ------------------------------------------------------------------
        for name, eras in zip(save_filenames, split_eras):
            if len(eras) == 0:
                raise ValueError(
                    f"save_custom_era_split: split '{name}' would contain no "
                    f"eras after the NaN-target filter - refusing to write "
                    f"an empty file. Check erasplit/drop_nan_target_eras."
                )

        # ------------------------------------------------------------------
        # 4. write each split with a pyarrow filter (rows streamed, only the
        #    matching split is ever materialised - full df never loaded)
        # ------------------------------------------------------------------
        dataset = ds.dataset(src_path, format="parquet")
        for name, eras in zip(save_filenames, split_eras):
            out_path = os.path.join(self.datapath, name)
            table = dataset.to_table(filter=ds.field("era").isin(eras))
            pq.write_table(
                table,
                out_path,
            )
            print(
                f"save_custom_era_split: wrote {out_path} "
                f"({len(eras)} eras, {table.num_rows} rows)"
            )
            del table
            gc.collect()

    def train(
        self,
        type_of_model,
        model_params,
        fit_params,
        save_to_drive,
        save_folder,
        calculate_metrics,
        plot_metrics,
        legacy_save=False,
        iteration=0,
    ):
        """
        wrapper around cross_validate_train
        iteration is here to choose a target in case we have a list of targets
        """
        if type(self.target) != list:
            target = self.target
        else:
            target = self.target[iteration]

        cross_validate_train(
            feature_names=self.feature_cols,
            cv_split_data=self.cv_split_data,
            target_name=target,
            train_df=self.train_df,
            tour_df=self.test_df,
            type_of_model=type_of_model,
            model_params=model_params,
            fit_params=fit_params,
            save_to_drive=save_to_drive,
            save_folder=save_folder,
            legacy_save=legacy_save,
            calculate_metrics=calculate_metrics,
            plot_metrics=plot_metrics,
        )
