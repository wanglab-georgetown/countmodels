import os
import pickle
import warnings
import time
import pystan
import numpy as np
import statsmodels.api as sm

from .suppress_stdout_stderr import suppress_stdout_stderr


class BaseModel:
    def __init__(self, endog, exog, exog_infl=None, model_path="./models"):

        if len(endog.shape) == 1:
            self.endog = endog.reshape((-1, 1))
        else:
            self.endog = endog

        self.num_sample, self.num_out = np.shape(self.endog)

        df_model = 0
        self.exog = exog
        df_model = np.linalg.matrix_rank(exog)
        if len(exog.shape) == 1:
            self.k_exog = 1
        else:
            self.k_exog = exog.shape[1]
        idx = np.where(np.ptp(self.exog, axis=0) <= 1e-5)[0].flatten()
        self.exog_const_idx = None
        if len(idx) > 0:
            # take the first constant column. feature processing should already
            # remove multiple constant columns
            self.exog_const_idx = idx[0]

        self.exog_infl = exog_infl
        if exog_infl is None:
            self.k_exog_infl = 0
            self._no_exog_infl = True
            self.exog_infl_const_idx = None
        else:
            if len(exog_infl.shape) == 1:
                self.k_exog_infl = 1
            else:
                self.k_exog_infl = exog_infl.shape[1]
            self._no_exog_infl = False
            if self.k_exog_infl > 0:
                df_model = df_model + np.linalg.matrix_rank(exog_infl)
            idx = np.where(np.ptp(self.exog_infl, axis=0) <= 1e-5)[0].flatten()
            self.exog_infl_const_idx = None
            if len(idx) > 0:
                self.exog_infl_const_idx = idx[0]

        self.df_model = df_model
        self.model_path = model_path

        # class/static variable
        if not hasattr(self.__class__, "model_name"):
            self.__class__.model_name = self.__class__.__name__.lower()

        if not hasattr(self.__class__, "stan_model") and self._stan_code() is not None:
            self.__class__.stan_model = self.get_stan_model()

    def _stan_code(self):
        return None

    def _stan_model_file(self):
        # think about path here
        return os.path.join(self.model_path, "{}.model".format(self.model_name))

    def save_stan_model(self):
        with suppress_stdout_stderr():
            model = pystan.StanModel(model_code=self._stan_code())
        file = self._stan_model_file()
        with open(file, "wb") as f:
            pickle.dump(model, f)
        return model

    def get_stan_model(self):
        file = self._stan_model_file()
        if not os.path.exists(file):
            model = self.save_stan_model()
            return model

        with open(file, "rb") as f:
            model = pickle.load(f)
        return model

    # https://github.com/statsmodels/statsmodels/blob/main/statsmodels/discrete/discrete_model.py#L3691
    def _estimate_dispersion(self, mu, resid, df_resid=None, loglike_method="nb2"):
        if df_resid is None:
            df_resid = resid.shape[0]
        if loglike_method == "nb2":
            a = ((resid**2 / mu - 1) / mu).sum() / df_resid
        else:  # self.loglike_method == 'nb1':
            a = (resid**2 / mu - 1).sum() / df_resid
        return a

    def _compute_pi_init(self, nz_prob, p_nonzero, infl_prob_max=0.99):
        ww = 1 - min(nz_prob / p_nonzero, infl_prob_max)
        return -np.log(1 / ww - 1)

    def _logit_fit(self, Y_logit, exog, start_params=None, disp=False):
        nonzero = np.sum(Y_logit)
        eps = 1e-3
        if nonzero > len(Y_logit) - eps or nonzero < eps:
            res = {}
            res["params"] = np.zeros(exog.shape[1])
            res["llf"] = 0.0
            res["cpu_time"] = 0.0
            return res

        methods = ["newton", "bfgs", "nm"]
        maxiters = [35, 1000, 35]
        logit_mod = None
        find_sol = False
        for ii in range(len(methods)):
            method = methods[ii]
            maxiter = maxiters[ii]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    start_time = time.time()
                    logit_mod = sm.Logit(Y_logit, exog).fit(
                        method=method,
                        maxiter=maxiter,
                        disp=disp,
                        start_params=start_params,
                    )
                    cpu_time = time.time() - start_time
                    if (
                        np.isnan(logit_mod.params).any()
                        or not logit_mod.mle_retvals["converged"]
                    ):
                        continue
                    find_sol = True
                    break
                except Exception as e:
                    continue

        res = {}
        res["params"] = logit_mod.params
        res["llf"] = logit_mod.llf
        res["cpu_time"] = cpu_time
        res["converged"] = find_sol

        return res

    def _parse_statsmodel_res(self, mod, cpu_time):
        res = {}
        res["params"] = mod.params
        res["llf"] = mod.llf
        res["df"] = self.df_model  # statsmodel.df_model does not count intercept
        res["aic"] = mod.aic
        res["cpu_time"] = cpu_time
        res["model"] = self.model_name
        res["method"] = "statsmodels"

        return res

    def _statsmodel_fit(
        self,
        endog1,
        maxiter=100,
        disp=False,
        warn_convergence=False,
        start_params=None,
        **kwargs,
    ):
        raise Exception(
            "statsmodel is not implemented for {}".format(self.__class__.__name__)
        )

    def _stan_model_optimizing(self, data, niter=1, init=0, seed=None):
        for i in range(niter):
            try:
                r = dict(self.stan_model.optimizing(data=data, init=init, seed=seed))
                return r
            except Exception as e:
                continue
        return {}

    def _stan_fit(self, endog1, start_params=None, **kwargs):
        raise Exception(
            "stan is not implemented for {}".format(self.__class__.__name__)
        )

    # start_params order: params_infl + params + dispersion
    def fit(self, method="stan", start_params=None, **kwargs):
        single_init = False
        if (
            start_params is None
            or len(np.shape(start_params)) == 1
            or np.shape(start_params)[0] == 1
        ):
            single_init = True
            if start_params is not None:
                start_params = np.reshape(start_params, (-1,))
        else:
            start_params = np.array(start_params)
        res = []
        for i in range(self.num_out):
            endog1 = self.endog[:, i]
            if single_init:
                _start_params = start_params
            else:
                if np.isnan(start_params[i, 0]):
                    _start_params = None
                else:
                    _start_params = start_params[i, :]
            r = {}
            if method == "stan":
                r = self._stan_fit(endog1, start_params=_start_params, **kwargs)
            if method == "statsmodels":
                r = self._statsmodel_fit(endog1, start_params=_start_params, **kwargs)
            res.append(r)
        return res
