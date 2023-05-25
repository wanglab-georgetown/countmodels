import pandas as pd
import numpy as np
from scipy import stats
from models.nb import NB
from models.zinb import ZINB
from models.nbh import NBH

from utils import remove_list, find_independent_columns, gen_feature_df


class LRTest:
    def __init__(
        self,
        model_class,
        df_data,
        df_feature,
        conditions,
        nb_features,
        scaler_col=None,
        infl_features=None,
        add_intercept=True,
        model_path="./models",
    ):
        self.model_class = model_class
        self.df_data = df_data
        self.df_feature = df_feature
        self.conditions = list(conditions)
        self.nb_features = list(set(nb_features) - set(conditions))
        self.num_sample, self.num_out = np.shape(self.df_data)

        if infl_features is not None:
            self.infl_features = list(set(infl_features) - set(conditions))
            self._no_infl = False
        else:
            self.infl_features = None
            self._no_infl = True

        self.scaler_col = scaler_col
        self.is_mast = False
        if model_class.__name__.lower() == "mast" and self.scaler_col is not None:
            self.is_mast = True

        self.add_intercept = add_intercept
        self.model_path = model_path

        self._gen_feature_dfs()

        self.res0 = None
        self.res1 = None
        self.df_result = None
        self.start_params1 = None

    def _get_feature_idx_map(self, Xs):
        df_maps = []
        for X in Xs:
            df_f = pd.DataFrame(X.columns, columns=["feature"])
            df_f["idx"] = range(len(df_f))
            df_maps.append(df_f)
        dft = pd.merge(df_maps[1], df_maps[0], on="feature")
        idx_map = [dft.idx_x.values, dft.idx_y.values]
        return idx_map

    def _gen_feature_dfs(self):
        fs = [self.nb_features, self.nb_features + self.conditions]
        dfs = []
        for feature in fs:
            df = gen_feature_df(
                self.df_feature, feature, add_intercept=self.add_intercept
            )
            dfs.append(df)
        dfs = self._find_full_rank_dfs(dfs)
        Xs = []
        for df in dfs:
            Xs.append(df.values)
        self.dfs = dfs
        self.Xs = Xs
        self.X_idx_map = self._get_feature_idx_map(dfs)

        if not self._no_infl:
            fs = [self.infl_features, self.infl_features + self.conditions]
            dfs = []
            for feature in fs:
                df = gen_feature_df(
                    self.df_feature, feature, add_intercept=self.add_intercept
                )
                dfs.append(df)
            dfs = self._find_full_rank_dfs(dfs)
            X_infls = []
            for df in dfs:
                X_infls.append(df.values)
            self.dfs_infl = dfs
            self.X_infls = X_infls
            self.X_infl_idx_map = self._get_feature_idx_map(dfs)
        else:
            self.X_infls = [None, None]
            self.X_infl_idx_map = None

    # note that even though df_feature may be independent across all clusters
    # some features may be dependent in certain clusters
    def _find_full_rank_dfs(self, dfs):
        # always keep the condition columns
        keep_cols = remove_list(list(dfs[1].columns), list(dfs[0].columns))
        # find independent columns with condition columns
        ind_cols = find_independent_columns(dfs[1], keep_cols=keep_cols)
        return [dfs[0][ind_cols], dfs[1][ind_cols + keep_cols]]

    def _gen_condition_init(self, res0):
        n_params0 = np.shape(self.X_infls[0])[1] + np.shape(self.Xs[0])[1] + 1
        for r in res0:
            if "params" in r:
                n_params0 = len(r["params"])
                break

        params0 = []
        for r in res0:
            if "params" in r:
                params0.append(r["params"])
            else:
                params0.append(np.nan * np.ones(n_params0))
        params0 = np.array(params0)

        params1 = []
        idx_base0 = 0
        if not self._no_infl:
            gamma = np.zeros((self.num_out, np.shape(self.X_infls[1])[1]))
            X_infl_idx_map = self.X_infl_idx_map
            gamma[:, X_infl_idx_map[1]] = params0[:, X_infl_idx_map[0] + idx_base0]
            idx_base0 = idx_base0 + np.shape(self.X_infls[0])[1]
            params1.append(gamma)

        beta = np.zeros((self.num_out, np.shape(self.Xs[1])[1]))
        X_idx_map = self.X_idx_map
        beta[:, X_idx_map[1]] = params0[:, X_idx_map[0] + idx_base0]
        idx_base0 = idx_base0 + np.shape(self.Xs[0])[1]
        params1.append(beta)

        # these 3 have an additional dispersion
        if self.model_class in [NB, ZINB, NBH]:
            params1.append(params0[:, -1:])

        return np.concatenate(params1, axis=1)

    def run_condition(self, method, condition=True, start_params=None):
        Y = self.df_data.values
        condition_idx = 1 if condition else 0

        if self.is_mast:
            scaler = self.df_feature[self.scaler_col].values

            mod = self.model_class(
                Y,
                self.Xs[condition_idx],
                exog_infl=self.X_infls[condition_idx],
                model_path=self.model_path,
                scaler=scaler,
            )
        else:
            mod = self.model_class(
                Y,
                self.Xs[condition_idx],
                exog_infl=self.X_infls[condition_idx],
                model_path=self.model_path,
            )

        res = mod.fit(method=method, start_params=start_params)
        return res

    def run(self, method="stan", start_params=None, **kwargs):

        Y = self.df_data.values

        if self.is_mast:
            scaler = self.df_feature[self.scaler_col].values

            mod0 = self.model_class(
                Y,
                self.Xs[0],
                exog_infl=self.X_infls[0],
                model_path=self.model_path,
                scaler=scaler,
            )
        else:
            mod0 = self.model_class(
                Y, self.Xs[0], exog_infl=self.X_infls[0], model_path=self.model_path
            )

        res0 = mod0.fit(method=method, start_params=start_params, **kwargs)
        self.res0 = res0
        self._no_infl = mod0._no_exog_infl

        # initialize mod1 with mod0 results
        start_params1 = self._gen_condition_init(res0)
        self.start_params1 = start_params1

        if self.is_mast:
            mod1 = self.model_class(
                Y,
                self.Xs[1],
                exog_infl=self.X_infls[1],
                model_path=self.model_path,
                scaler=scaler,
            )
        else:
            mod1 = self.model_class(
                Y, self.Xs[1], exog_infl=self.X_infls[1], model_path=self.model_path
            )

        res1 = mod1.fit(method=method, start_params=start_params1, **kwargs)
        self.res1 = res1

        res = []
        for i in range(len(self.df_data.columns)):
            r0 = res0[i]
            r1 = res1[i]
            r = {}
            if "llf" in r0:
                r["llf0"] = r0["llf"]
                r["aic0"] = r0["aic"]
                r["df0"] = r0["df"]
                r["cpu_time0"] = r0["cpu_time"]

            if "llf" in r1:
                r["llf1"] = r1["llf"]
                r["aic1"] = r1["aic"]
                r["df1"] = r1["df"]
                r["cpu_time1"] = r1["cpu_time"]

            if "llf" in r0 and "llf" in r1:
                r["llfd"] = r["llf1"] - r["llf0"]
                r["aicd"] = r["aic1"] - r["aic0"]
                dfd = r["df1"] - r["df0"]
                r["pvalue"] = 1 - stats.chi2.cdf(2 * r["llfd"], dfd)

            r["model"] = r0["model"]
            r["method"] = r0["method"]

            res.append(r)
        dfr = pd.DataFrame(res)
        dfr["subject"] = self.df_data.columns

        self.df_result = dfr

        return dfr
