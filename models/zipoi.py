import warnings
import time
import numpy as np
from statsmodels.discrete.count_model import ZeroInflatedPoisson
from .basemodel import BaseModel
from .poi import Poi

from .suppress_stdout_stderr import suppress_stdout_stderr


class ZIPoi(BaseModel):
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
        **kwargs,
    ):
        if self._no_exog_infl:
            return {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                start_time = time.time()
                mod = ZeroInflatedPoisson(
                    endog1, self.exog, exog_infl=self.exog_infl
                ).fit(
                    start_params=start_params,
                    maxiter=maxiter,
                    disp=disp,
                    warn_convergence=warn_convergence,
                )
                cpu_time = time.time() - start_time
                if np.isnan(mod.params).any():
                    return {}
            except Exception as e:
                print("_statsmodel_fit ZeroInflatedPoisson exception {}".format(e))
                return {}

        res = self._parse_statsmodel_res(mod, cpu_time)
        return res

    def _get_stan_init(self, Y, X, X_infl=None, start_params=None, infl_prob_max=0.99):
        if (
            start_params is not None
            and len(start_params) == self.k_exog + self.k_exog_infl
        ):
            start_params = np.array(start_params)
            init = {
                "gamma": np.array(start_params[: self.k_exog_infl]),
                "beta": np.array(start_params[self.k_exog_infl :]),
            }
            return init

        if X_infl is None:
            return {}

        model = Poi(Y, X, model_path=self.model_path)
        res = model.fit(method="stan", return_resid=True)[0]
        if "params" in res:
            mu = res["mu"]
            beta = res["params"]
        else:
            beta = np.zeros(self.k_exog) + 0.001
            mu = np.mean(Y)
            if self.exog_const_idx is not None:
                m = X[0, self.exog_const_idx]
                beta[self.exog_const_idx] = np.log(mu) / m

        gamma = np.zeros(self.k_exog_infl) + 0.001
        if self.exog_infl_const_idx is not None:
            p_nonzero = 1 - np.mean(np.exp(-mu))
            nz_prob = np.mean(Y > 0)
            w_pi = self._compute_pi_init(
                nz_prob, p_nonzero, infl_prob_max=infl_prob_max
            )
            m = X_infl[0, self.exog_infl_const_idx]
            gamma[self.exog_infl_const_idx] = w_pi / m

        init = {
            "beta": beta,
            "gamma": gamma,
        }

        return init

    def _stan_fit(self, endog1, start_params=None, **kwargs):
        if self._no_exog_infl:
            return {}

        data = {}
        data["N"] = self.num_sample
        data["Xb"] = self.exog_infl
        data["Kb"] = self.k_exog_infl
        data["Xc"] = self.exog
        data["Kc"] = self.k_exog
        data["Y"] = endog1

        start_time = time.time()
        init = self._get_stan_init(
            endog1, self.exog, X_infl=self.exog_infl, start_params=start_params
        )

        with suppress_stdout_stderr():
            start_time = time.time()
            r = self._stan_model_optimizing(data, niter=1, init=init)
            if len(r) == 0:
                r = self._stan_model_optimizing(data, niter=1, init=0)
                if len(r) == 0:
                    r = self._stan_model_optimizing(data, niter=5, init="random")
        cpu_time = time.time() - start_time

        if len(r) == 0:
            return {"model": self.model_name, "method": "stan"}

        res = {}
        res["params"] = np.concatenate(
            (r["gamma"].reshape((self.k_exog_infl,)), r["beta"].reshape((self.k_exog,)))
        )
        res["llf"] = r["log_lik"].flatten()[0]
        res["df"] = self.df_model
        res["aic"] = 2 * (self.df_model - res["llf"])
        res["cpu_time"] = cpu_time
        res["model"] = self.model_name
        res["method"] = "stan"

        return res
