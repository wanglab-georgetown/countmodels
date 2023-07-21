import warnings
import time
import numpy as np
import statsmodels.api as sm
from .basemodel import BaseModel
from .poi import Poi

from .suppress_stdout_stderr import suppress_stdout_stderr


class NB(BaseModel):
    def __init__(self, endog, exog, exog_infl=None, model_path="./models"):
        super(NB, self).__init__(endog, exog, exog_infl=None, model_path=model_path)

        # dispersion parameter
        self.df_model = self.df_model + 1

    def _stan_code(self):
        stan_code = """
            data{
                int N;
                int K;
                matrix[N,K] X;
                int Y[N];
            }
            parameters{
                vector[K] beta;
                real<lower=-30, upper=30> log_alpha;
            }
            transformed parameters{
                vector[N] mu;
                real alpha;
            
                mu = exp(X * beta);
                alpha = exp(log_alpha);
            }
            model{ 
                Y ~ neg_binomial_2(mu, alpha);
            }
            generated quantities {
                real log_lik;
                log_lik = 0;
                for (i in 1:N) {
                    log_lik += neg_binomial_2_lpmf(Y[i]| mu[i], alpha);
                }
            }
            """
        return stan_code

    def _statsmodel_fit(
        self,
        endog1,
        maxiter=1000,
        disp=False,
        warn_convergence=False,
        start_params=None,
        **kwargs,
    ):

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                start_time = time.time()
                mod = sm.NegativeBinomial(endog1, self.exog).fit(
                    maxiter=maxiter,
                    disp=disp,
                    warn_convergence=warn_convergence,
                    start_params=start_params,
                )
                cpu_time = time.time() - start_time
                if np.isnan(mod.params).any():
                    return {}
            except Exception as e:
                print("_statsmodel_fit NegativeBinomial exception {}".format(e))
                return {}

        res = self._parse_statsmodel_res(mod, cpu_time)
        return res

    def _get_stan_init(
        self, Y, X, X_infl=None, start_params=None, theta_lb=0.05, infl_prob_max=0.99
    ):
        nb_only = X_infl is None
        if start_params is not None:
            start_params = np.array(start_params)
            if nb_only and len(start_params) == self.k_exog + 1:
                init = {
                    "beta": np.array(start_params[: self.k_exog]),
                    "log_alpha": start_params[-1],
                }
                return init

            if not nb_only and len(start_params) == self.k_exog + self.k_exog_infl + 1:
                init = {
                    "gamma": np.array(start_params[: self.k_exog_infl]),
                    "beta": np.array(start_params[self.k_exog_infl : -1]),
                    "log_alpha": start_params[-1],
                }
                return init

        model = Poi(Y, X, model_path=self.model_path)
        res = model.fit(method="stan", return_resid=True)[0]
        if "params" in res:
            theta = self._estimate_dispersion(
                res["mu"], res["resid"], df_resid=res["df_resid"]
            )
            mu = res["mu"]
            beta = res["params"]
        else:
            beta = np.zeros(self.k_exog) + 0.001
            mu = np.mean(Y)
            if self.exog_const_idx is not None:
                m = X[0, self.exog_const_idx]
                beta[self.exog_const_idx] = np.log(mu) / m
            resid = Y - mu
            theta = self._estimate_dispersion(mu, resid)

        if np.isnan(theta) or np.isinf(theta):
            theta = theta_lb
        theta = max(theta, theta_lb)

        init = {
            "beta": beta,
            "log_alpha": -np.log(theta),
        }

        if not nb_only:
            gamma = np.zeros(self.k_exog_infl) + 0.001
            if self.exog_infl_const_idx is not None:
                # 1/theta is the dispersion in nb2 model
                p_nonzero = 1 - np.mean(np.power(1 / (1 + mu * theta), 1 / theta))
                nz_prob = np.mean(Y > 0)
                w_pi = self._compute_pi_init(
                    nz_prob, p_nonzero, infl_prob_max=infl_prob_max
                )
                m = X_infl[0, self.exog_infl_const_idx]
                gamma[self.exog_infl_const_idx] = w_pi / m
            init["gamma"] = gamma

        return init

    def _stan_fit(self, endog1, start_params=None, **kwargs):
        data = {}
        data["N"] = self.num_sample
        data["X"] = self.exog
        data["Y"] = endog1
        data["K"] = self.k_exog

        start_time = time.time()
        init = self._get_stan_init(endog1, self.exog, start_params=start_params)

        with suppress_stdout_stderr():
            start_time = time.time()
            r = self._stan_model_optimizing(data, niter=1, init=init)
            if len(r) == 0:
                r = self._stan_model_optimizing(data, niter=1, init=0)
                if len(r) == 0:
                    r = self._stan_model_optimizing(data, niter=10, init="random")
        cpu_time = time.time() - start_time

        if len(r) == 0:
            return {"model": self.model_name, "method": "stan"}

        res = {}
        res["params"] = np.concatenate(
            (r["beta"].reshape((-1,)), r["log_alpha"].reshape((1,)))
        )
        res["llf"] = r["log_lik"].flatten()[0]
        res["df"] = self.df_model
        res["aic"] = 2 * (self.df_model - res["llf"])
        res["cpu_time"] = cpu_time
        res["model"] = self.model_name
        res["method"] = "stan"

        return res

    def _get_tensorflow_init(self, start_params=None):
        if start_params is None:
            return {}

        weights = []
        if len(np.shape(start_params)) == 1:
            weights = [start_params for i in range(self.num_out)]
        elif np.shape(start_params)[0] == 1:
            start_params = np.reshape(start_params, (-1,))
            weights = [start_params for i in range(self.num_out)]
        else:
            weights = start_params
        weights = np.array(weights).T

        if self._no_exog_infl:
            init = {
                "x_mu": weights[:-1, :],
                "theta": weights[-1:, :],
            }
        else:
            init = {
                "x_pi": weights[: self.k_exog_infl, :],
                "x_mu": weights[self.k_exog_infl : -1, :],
                "theta": weights[-1:, :],
            }
        return init

    def _tensorflow_fit(self, start_params=None, **kwargs):
        try:
            from tensorzinb.tensorzinb import TensorZINB
        except ModuleNotFoundError:
            raise Exception("TensorZINB is not installed")

        mod = TensorZINB(self.endog, self.exog, nb_only=True)

        init_weights = self._get_tensorflow_init(start_params)

        res_tf = mod.fit(init_weights=init_weights, reset_keras_session=True)

        res = []
        cpu_time = res_tf["cpu_time"] / self.num_out
        for i in range(self.num_out):
            r = {}
            r["params"] = np.concatenate(
                (
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

    def fit(self, method="stan", start_params=None, **kwargs):
        if method == "tensorflow":
            return self._tensorflow_fit(start_params=start_params, **kwargs)

        return super(NB, self).fit(method=method, start_params=start_params, **kwargs)
