import warnings
import time
import numpy as np
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from .basemodel import BaseModel
from .nb import NB

from .suppress_stdout_stderr import suppress_stdout_stderr


class ZINB(NB):
    def __init__(self, endog, exog, exog_infl=None, model_path="./models"):
        # inherit from NB but NB has exog_infl==None and cannot call NB init
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
        **kwargs
    ):
        if self._no_exog_infl:
            return {}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                start_time = time.time()
                mod = ZeroInflatedNegativeBinomialP(
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
                print(
                    "_statsmodel_fit ZeroInflatedNegativeBinomialP exception {}".format(
                        e
                    )
                )
                return {}

        res = self._parse_statsmodel_res(mod, cpu_time)
        return res

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
            (
                r["gamma"].reshape((self.k_exog_infl,)),
                r["beta"].reshape((self.k_exog,)),
                r["log_alpha"].reshape((1,)),
            )
        )
        res["llf"] = r["log_lik"].flatten()[0]
        res["df"] = self.df_model
        res["aic"] = 2 * (self.df_model - res["llf"])
        res["cpu_time"] = cpu_time
        res["model"] = self.model_name
        res["method"] = "stan"

        return res

    def _tensorflow_fit(self, start_params=None, **kwargs):
        try:
            from tensorzinb.tensorzinb import TensorZINB
        except ModuleNotFoundError:
            raise Exception("TensorZINB is not installed")

        mod = TensorZINB(self.endog, self.exog, exog_infl=self.exog_infl)

        init_weights = self._get_tensorflow_init(start_params)

        res_tf = mod.fit(init_weights=init_weights, reset_keras_session=True)

        res = []
        cpu_time = res_tf["cpu_time"] / self.num_out
        for i in range(self.num_out):
            r = {}
            r["params"] = np.concatenate(
                (
                    res_tf["weights"]["x_pi"][:, i],
                    res_tf["weights"]["x_mu"][:, i],
                    res_tf["weights"]["theta"][:, i],
                )
            )
            r["llf"] = res_tf["llfs"][i]
            r["aic"] = res_tf["aics"][i]
            r["df"] = res_tf["df"]
            r["cpu_time"] = cpu_time
            r["model"] = self.model_name
            r["method"] = "tensorflow"
            res.append(r)
        return res
