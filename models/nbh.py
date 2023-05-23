import warnings
import time
import numpy as np
from statsmodels.discrete.truncated_model import TruncatedLFNegativeBinomialP
from .basemodel import BaseModel
from .nb import NB
from .poih import PoiH

from .suppress_stdout_stderr import suppress_stdout_stderr


class NBH(NB):
    def __init__(self, endog, exog, exog_infl=None, model_path="./models"):
        BaseModel.__init__(
            self, endog, exog, exog_infl=exog_infl, model_path=model_path
        )

        # dispersion parameter
        self.df_model = self.df_model + 1

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
        eps=1e-8,
        **kwargs,
    ):
        # as of statsmodel 0.14.0 HurdleCountModel does not support a logit modeling
        # on the hurdle part so do logit and truncated negative binomial separately
        # https://www.statsmodels.org/dev/examples/notebooks/generated/count_hurdle.html

        if self._no_exog_infl:
            return {}

        start_params_logit = None
        start_params_pos = None
        if (
            start_params is not None
            and len(start_params) == self.k_exog_infl + self.k_exog + 1
        ):
            start_params = np.array(start_params)
            start_params_logit = start_params[: self.k_exog_infl]
            start_params_pos = start_params[self.k_exog_infl :]

        nonzero_index = np.where(endog1 > eps)[0]

        # logit
        Y_logit = np.zeros(self.num_sample)
        Y_logit[nonzero_index] = 1
        res_logit = self._logit_fit(
            Y_logit, self.exog_infl, start_params=start_params_logit
        )

        if len(res_logit) == 0:
            return {}

        # positive negative binomial
        X = self.exog[nonzero_index, :]
        Y = endog1[nonzero_index].astype(int)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                start_time = time.time()
                mod = TruncatedLFNegativeBinomialP(Y, X).fit(
                    start_params=start_params_pos,
                    maxiter=maxiter,
                    disp=disp,
                    warn_convergence=warn_convergence,
                )
                cpu_time = time.time() - start_time
                if np.isnan(mod.params).any():
                    return {}
            except Exception as e:
                print("_statsmodel_fit TruncatedLFNegativeBinomialP exception {}".format(e))
                return {}

        res_poi = self._parse_statsmodel_res(mod, cpu_time)

        res = {}
        res["params"] = np.concatenate((res_logit["params"], res_poi["params"]))
        res["llf_logit"] = res_logit["llf"]
        res["llf_nb"] = res_poi["llf"]
        res["llf"] = res_logit["llf"] + res_poi["llf"]
        res["df"] = self.df_model
        res["aic"] = 2 * (self.df_model - res["llf"])
        res["cpu_time"] = res_logit["cpu_time"] + res_poi["cpu_time"]
        res["model"] = self.model_name
        res["method"] = "statsmodels"

        return res

    def _stan_hurdle_fit(self, Y, X, start_params=None):
        data = {}
        data["N"] = len(Y)
        data["X"] = X
        data["Y"] = Y
        data["K"] = self.k_exog

        start_time = time.time()
        init = self._get_stan_init(Y, X, start_params=start_params)

        with suppress_stdout_stderr():
            start_time = time.time()
            r = self._stan_model_optimizing(data, niter=1, init=init)
            if len(r) == 0:
                r = self._stan_model_optimizing(data, niter=1, init=0)
                if len(r) == 0:
                    r = self._stan_model_optimizing(data, niter=5, init="random")
        cpu_time = time.time() - start_time

        if len(r) == 0:
            return {}

        res = {}
        res["params"] = np.concatenate(
            (r["beta"].reshape((-1,)), r["log_alpha"].reshape((1,)))
        )
        res["llf"] = r["log_lik"].flatten()[0]
        res["cpu_time"] = cpu_time
        return res

    # NBH can be considered as a combination of PoiH and NB
    # stan fit is the same as PoiH has two parts but the hurdle part is NB
    def _stan_fit(self, endog1, start_params=None, eps=1e-8, **kwargs):
        return PoiH._stan_fit(
            self, endog1, start_params=start_params, eps=eps, **kwargs
        )

    def fit(self, method="stan", start_params=None, **kwargs):
        if method == "tensorflow":
            raise Exception(
                "tensorflow method is not supported for {}".format(self.model_name)
            )

        return super(NBH, self).fit(method=method, start_params=start_params, **kwargs)

    def fit_fast(self, start_params=None, eps=1e-8):
        res = []
        for i in range(self.num_out):
            endog1 = self.endog[:, i]
            r = self._statsmodel_fit(endog1, start_params=start_params)
            if np.isnan(r["llf_nb"]):
                r = self._stan_fit(endog1, start_params=start_params)
            res.append(r)
        return res
