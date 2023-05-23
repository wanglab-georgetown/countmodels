import warnings
import time
import numpy as np
import statsmodels.api as sm
from .basemodel import BaseModel

from .suppress_stdout_stderr import suppress_stdout_stderr


class Poi(BaseModel):
    def __init__(self, endog, exog, exog_infl=None, model_path="./models"):
        super(Poi, self).__init__(endog, exog, exog_infl=None, model_path=model_path)

    def _stan_code(self):
        stan_code = ""
        return stan_code

    def _statsmodel_fit(
        self,
        endog1,
        maxiter=100,
        disp=False,
        warn_convergence=False,
        start_params=None,
        **kwargs
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                start_time = time.time()
                mod = sm.Poisson(endog1, self.exog).fit(
                    maxiter=maxiter,
                    disp=disp,
                    warn_convergence=warn_convergence,
                    start_params=start_params,
                )
                cpu_time = time.time() - start_time
                if np.isnan(mod.params).any():
                    return {}
            except Exception as e:
                print("_statsmodel_fit Poisson exception {}".format(e))
                return {}

        res = self._parse_statsmodel_res(mod, cpu_time)

        if "return_resid" in kwargs and kwargs["return_resid"]:
            res["mu"] = mod.predict()
            res["resid"] = mod.resid
            res["df_resid"] = mod.df_resid
        return res

    def _get_stan_init(self, endog1, start_params=None):
        if start_params is not None and len(start_params) == self.k_exog:
            return {"beta": np.array(start_params)}

        start_params = np.zeros(self.k_exog) + 0.001
        if self.exog_const_idx is not None:
            m = self.exog[0, self.exog_const_idx]
            start_params[self.exog_const_idx] = np.log(np.mean(endog1)) / m
        return {"beta": start_params}

    def _stan_fit(self, endog1, start_params=None, **kwargs):
        data = {}
        data["N"] = self.num_sample
        data["X"] = self.exog
        data["Y"] = endog1
        data["K"] = self.k_exog

        start_time = time.time()
        init = self._get_stan_init(endog1, start_params=start_params)

        with suppress_stdout_stderr():
            r = self._stan_model_optimizing(data, niter=1, init=init)
            if len(r) == 0:
                r = self._stan_model_optimizing(data, niter=1, init=0)
                if len(r) == 0:
                    r = self._stan_model_optimizing(data, niter=10, init="random")
        cpu_time = time.time() - start_time

        if len(r) == 0:
            raise Exception("Poisson stan cannot find solution")
            return {"model": self.model_name, "method": "stan"}

        res = {}
        res["params"] = r["beta"].reshape((-1,))
        res["llf"] = r["log_lik"].flatten()[0]
        res["df"] = self.df_model
        res["aic"] = 2 * (self.df_model - res["llf"])
        res["cpu_time"] = cpu_time
        res["model"] = self.model_name
        res["method"] = "stan"

        if "return_resid" in kwargs and kwargs["return_resid"]:
            res["mu"] = r["mu"]
            res["resid"] = endog1 - r["mu"]
            res["df_resid"] = self.num_sample - self.df_model

        return res
