import warnings
import time
import numpy as np
from statsmodels.discrete.truncated_model import TruncatedLFPoisson
from .basemodel import BaseModel
from .poi import Poi

from .suppress_stdout_stderr import suppress_stdout_stderr


class PoiH(BaseModel):
    def __init__(self, endog, exog, exog_infl=None, model_path="./models"):
        super(PoiH, self).__init__(
            endog, exog, exog_infl=exog_infl, model_path=model_path
        )

    def _stan_code(self):
        stan_code = """
            data{
                int N;
                int K;
                matrix[N, K] X;
                int Y[N];
            }
            parameters{
                vector[K] beta;
            }
            transformed parameters{
                vector[N] mu;
                mu = exp(X * beta);
            }
            model{
                for (i in 1:N) Y[i] ~ poisson(mu[i]) T[1,];
            }
            generated quantities {
              real log_lik;
              log_lik = 0;
              for (i in 1:N){
                log_lik += -lgamma(Y[i]+1)+Y[i]*log(mu[i])-mu[i]-log1m_exp(-mu[i]);
              }
            }
            """

        # the above code is a robust implementation of the following line
        # log_lik += poisson_lpmf(Y[i] | mu[i]) - log(1-exp(poisson_lpmf(0 | mu[i])));
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
        # on the hurdle part so do logit and truncated poisson separately
        # https://www.statsmodels.org/dev/examples/notebooks/generated/count_hurdle.html

        if self._no_exog_infl:
            return {}

        start_params_logit = None
        start_params_pos = None
        if (
            start_params is not None
            and len(start_params) == self.k_exog_infl + self.k_exog
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

        # positive Poisson
        X = self.exog[nonzero_index, :]
        Y = endog1[nonzero_index].astype(int)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                start_time = time.time()
                mod = TruncatedLFPoisson(Y, X).fit(
                    start_params=start_params_pos,
                    maxiter=maxiter,
                    disp=disp,
                    warn_convergence=warn_convergence,
                )
                cpu_time = time.time() - start_time
                if np.isnan(mod.params).any():
                    return {}
            except Exception as e:
                print("_statsmodel_fit TruncatedLFPoisson exception {}".format(e))
                return {}

        res_poi = self._parse_statsmodel_res(mod, cpu_time)

        res = {}
        res["params"] = np.concatenate((res_logit["params"], res_poi["params"]))
        res["llf_logit"] = res_logit["llf"]
        res["llf_poi"] = res_poi["llf"]
        res["llf"] = res_logit["llf"] + res_poi["llf"]
        res["df"] = self.df_model
        res["aic"] = 2 * (self.df_model - res["llf"])
        res["cpu_time"] = res_logit["cpu_time"] + res_poi["cpu_time"]
        res["model"] = self.model_name
        res["method"] = "statsmodels"

        return res

    def _get_stan_init(self, Y, X, start_params=None):
        if start_params is not None and len(start_params) == self.k_exog:
            return {"beta": np.array(start_params)}

        model = Poi(Y, X, model_path=self.model_path)
        res = model.fit(method="stan")[0]
        if "params" in res:
            return {"beta": res["params"]}

        start_params = np.zeros(self.k_exog) + 0.001
        if self.exog_const_idx is not None:
            m = np.mean(X[:, self.exog_const_idx])
            start_params[self.exog_const_idx] = np.log(np.mean(Y)) / m
        return {"beta": np.array(start_params)}

    def _stan_hurdle_fit(self, Y, X, start_params=None):
        data = {}
        data["N"] = len(Y)
        data["X"] = X
        data["Y"] = Y
        data["K"] = self.k_exog

        start_time = time.time()
        init = self._get_stan_init(Y, X, start_params=start_params)

        with suppress_stdout_stderr():
            r = self._stan_model_optimizing(data, niter=1, init=init)
            if len(r) == 0:
                r = self._stan_model_optimizing(data, niter=1, init=0)
                if len(r) == 0:
                    r = self._stan_model_optimizing(data, niter=5, init="random")
        cpu_time = time.time() - start_time

        if len(r) == 0:
            return {}

        res = {}
        res["params"] = r["beta"].reshape((-1,))
        res["llf"] = r["log_lik"].flatten()[0]
        res["cpu_time"] = cpu_time
        return res

    def _stan_fit(self, endog1, start_params=None, eps=1e-8, **kwargs):
        if self._no_exog_infl:
            return {}

        start_params_logit = None
        start_params_pos = None
        if start_params is not None:
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

        # positive Poisson
        X = self.exog[nonzero_index, :]
        Y = endog1[nonzero_index].astype(int)
        res_poi = self._stan_hurdle_fit(Y, X, start_params=start_params_pos)

        if len(res_poi) == 0:
            return {"model": self.model_name, "method": "stan"}

        res = {}
        res["params"] = np.concatenate((res_logit["params"], res_poi["params"]))
        res["llf_logit"] = res_logit["llf"]
        res["llf_poi"] = res_poi["llf"]
        res["llf"] = res_logit["llf"] + res_poi["llf"]
        res["df"] = self.df_model
        res["aic"] = 2 * (self.df_model - res["llf"])
        res["cpu_time"] = res_logit["cpu_time"] + res_poi["cpu_time"]
        res["model"] = self.model_name
        res["method"] = "stan"

        return res
