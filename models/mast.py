import warnings
import time
import numpy as np
import statsmodels.api as sm
from .basemodel import BaseModel


class MAST(BaseModel):
    def __init__(self, endog, exog, exog_infl=None, model_path="./models", scaler=None):
        super(MAST, self).__init__(
            endog, exog, exog_infl=exog_infl, model_path=model_path
        )

        # assume y_r = np.log2(1 + scaler*y)
        if scaler is None:
            self.scaler = np.ones(self.num_sample)
        else:
            self.scaler = scaler

    def llf_each(self, x, b, m, ss):
        tt = x * b + 1.0
        y = np.log2(tt)
        return np.exp(-0.5 * np.power(y - m, 2) / ss / ss) * b / tt

    # this assumes Y_r=np.log2(1+Y*beta)
    def _mast_llf_corrected(self, Y, beta, linear_mod):
        nY = len(Y)
        mu = linear_mod.predict()
        ss = np.sqrt(linear_mod.ssr / nY)

        ll0 = -0.5 * nY + np.sum(np.log(beta) - np.log(Y * beta + 1.0))

        rr = 0
        for i in range(nY):
            x = int(Y[i])
            b = beta[i]
            m = mu[i]
            xs = np.array(range(1, max(60, x * 20)))
            rr = rr + np.log(np.sum(self.llf_each(xs, b, m, ss)))

        return ll0 - rr

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

        # normal
        X = self.exog[nonzero_index, :]
        Y = endog1[nonzero_index]
        Y_r = np.log2(1.0 + endog1[nonzero_index] * self.scaler[nonzero_index])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                start_time = time.time()
                mod = sm.OLS(Y_r, X).fit(
                    maxiter=maxiter, disp=disp, warn_convergence=warn_convergence
                )
                cpu_time = time.time() - start_time
                if np.isnan(mod.params).any():
                    return {}
            except Exception as e:
                print("_statsmodel_fit OLS exception {}".format(e))
                return {"model": self.model_name, "method": "statsmodels"}

        res_normal = self._parse_statsmodel_res(mod, cpu_time)

        beta = self.scaler[nonzero_index].flatten()
        llf = self._mast_llf_corrected(Y.flatten(), beta, mod)

        res = {}
        res["params"] = np.concatenate((res_logit["params"], res_normal["params"]))
        res["llf_logit"] = res_logit["llf"]
        res["llf_normal_o"] = res_normal["llf"]
        res["llf_normal"] = llf
        res["llf"] = res["llf_logit"] + res["llf_normal"]
        res["df"] = self.df_model
        res["aic"] = 2 * (self.df_model - res["llf"])
        res["cpu_time"] = res_logit["cpu_time"] + res_normal["cpu_time"]
        res["model"] = self.model_name
        res["method"] = "statsmodels"

        return res
