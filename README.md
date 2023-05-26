# Compare Hurdle and Zero-inflated Count Models

## Overview
The purpose of this repo is

- to provide numerically stable implementations of hurdle and zero-inflated count models in stan (`Pystan`) and TensorFlow (`TensorZINB`).
- to provide a framework to compare different hurdle and zero-inflated count models.
- to provide a framework for scRNA-seq DEG analysis using different count models.
- to provide a framework for model selection and feature selection for count models.

This can be used as an alternative to `statsmodels` for Hurdle and Zero-inflated count models.

The following 7 models are supported, along with supported methods for each model given in `[]`. `stan` denotes `PyStan`, `statsmodels` denotes `statsmodels`, and `tensorflow` denotes [`TensorZINB`](https://github.com/wanglab-georgetown/tensorzinb)
:

``` r
    Poisson                         (Poi):   ['stan','statsmodels'],
    Negative Binomial               (NB):    ['stan','statsmodels','tensorflow'],
    Poisson Hurdle                  (PoiH):  ['stan','statsmodels'],
    Negative Binomial Hurdle        (NBH):   ['stan','statsmodels'],
    Zero-inflated Poisson           (ZIPoi): ['stan','statsmodels'],
    Zero-inflated Negative Binomial (ZINB):  ['stan','statsmodels','tensorflow'],
    Normal Hurdle                   (MAST):  ['statsmodels']
``` 

We provide protocols for performing  **model selection**, **feature selection** and **DEG analysis** for single-cell RNA sequencing (scRNA-seq) analysis using the 7 models. 

## Background
For a counter distribution on non-negative integers with probability mass function (PMF) $f(y)$, e.g., Poisson, negative binomial, its **hurdle model** can be expressed as 
$$Pr(Y=0)=\pi, Pr(Y=y)=(1-\pi)\frac{f(y)}{1-f(0)},y>0.$$

Its **zero-inflated model** can be written as 
$$Pr(Y=0)=\pi+(1-\pi)f(0),Pr(Y=y)=(1-\pi)f(y),y>0.$$


## Installation

No installation is required to use this repo. It is recommended to run `pip install -r requirements.txt` to install the required packages. To use the `tensorflow` method, you need to install [`TensorZINB`](https://github.com/wanglab-georgetown/tensorzinb). For Apple silicon (M1, M2, etc), it is recommended to install `tensorflow` by following the command in the Troubleshooting section below.

## Model Specifications
### Model Class

You can import each model using `from models.{model_name} import {ModelName}`.

- `model_name`: `poi`, `nb`, `poih`, `nbh`, `zipoi`, `zinb`, `mast`
- `ModelName`: `Poi`, `NB`, `PoiH`, `NBH`, `ZIPoi`, `ZINB`, `MAST`

The API and results format follows those in `statsmodels`.

``` r
model = ModelName(
    endog,                     # counts data: number of samples x number of subjects
    exog,                      # observed variables for the non logit part
    exog_infl=None,            # observed variables for the logit part (hurdle and zero-inflated models)
    model_path='./models',     # the location of stan model file
    scaler=None,               # scaler for MAST in log2 transformed count: log_2(1+scaler*count)
)        
```

### Model fit

``` r
model.fit(
    method='stan',            # method for solving model: stan, statsmodels, tensorflow
    start_params=None,        # start params for model
)        
```

### Model results

`model.fit` returns a list of dictionaries, where each dictionary corresponds to a subject in `endog`. Each dictionary has the following format:
``` r
{
    "params":                 # params of model
    "llf":                    # log likelihood
    "aic":                    # AIC
    "df":                     # degree of freedom
    "cpu_time":               # computing time in seconds
    "model":                  # name of the model: `poi`, `nb`, `poih`, `nbh`, `zipoi`, `zinb`, `mast`
    "method":                 # method for solving the model: 'stan','statsmodels','tensorflow'
}     
```

### Likelihood-ratio test (LRT)

`LRTest` provides a wrapper for LRT. It runs the likelihood ratio test by computing the log likelihood difference with and without using conditions in the given model. It automatically generates a feature matrix and removes dependent feature columns.

To import this class, run `from lrtest import LRTest` (see `DEG_analysis.ipynb` for an example). To construct a `LRTest` object, run
``` r
lrtest = LRTest(
    model_class,             # model class from `Poi`, `NB`, `PoiH`, `NBH`, `ZIPoi`, `ZINB`, `MAST`
    df_data,                 # count data frame. columns: subjects (genes), rows: samples
    df_feature,              # feature data frame. columns: features, rows: samples
    conditions,              # list of features to test DEG, e.g., diagnosis
    nb_features,             # list of features for the non logit part
    infl_features=None,      # list of features for the logit part (hurdle and zero-inflated models)
    add_intercept=True,      # whether add intercept. False if df_feature already contains intercept
    scaler_col=None,         # scaler column name for MAST in log2 transformed count: log_2(1+scaler*count)
    model_path='./models',   # the location of stan model file
)        
```

We then call `lrtest.run` to run the likelihood ratio test.
``` r
lrtest.run(
    method,                  # method for solving the model
)        
```

`lrtest.run` returns a result dataframe `dfr` with columns:
``` r
[
	"llf0":                  # log likelihood without conditions
	"aic0":                  # AIC without conditions
	"df0":                   # degree of freedom without conditions
	"cpu_time0":             # computing time for each subject without conditions
	"llf1":                  # log likelihood without conditions
	"aic1":                  # AIC with conditions
	"df1":                   # degree of freedom with conditions
	"cpu_time1":             # computing time for each subject with conditions
	"llfd":                  # ll1 - ll0
	"aicd":                  # aic1 - aic0
	"pvalue":                # p-value: 1 - stats.chi2.cdf(2 * lld, df1 - df0)
]
```

We can further correct the pvalues for multiple testing by calling `utils.correct_pvalues_for_multiple_testing(dfr['pvalue'])`.

We also call `lrtest.run_condition` to get results from with conditions only (useful for feature selection. see `feature_selection.ipynb` for an example).
``` r
lrtest.run_condition(
    method,                  # method for solving the model
    condition=True,          # whether include condition
)        
```
 
## scRNA-seq Analysis

We provide notebooks on how to perform **model selection**, **feature selection** and **DEG analysis** using the 7 models. `data` folder contains a sample scRNA-seq dataset that contains 17 clusters, and each cluster contains 20 genes from https://cells.ucsc.edu/?ds=autism.
``` r
 `model_sel_genes.csv`:      list of genes and clusters  
 `model_sel_count.zip`:      raw counts for the selected genes  
 `meta.zip`:                 cell meta annotations (features)  
```

### Model selection

`model_selection.ipynb` runs all 7 models across all supported methods for each model on the sample dataset. log likelihood (llf), Akaike information criterion (AIC) and average computing time are compared.

### Feature selection

`feature_selection.ipynb` runs the `topdown` or `bottomup` algorithm to successively remove or add a feature based on AIC. We also show how to remove redundant features before performing feature selection.

### DEG analysis

`DEG_analysis.ipynb` performs DEG analysis through `LRTest` using one of the 7 supported models.

### R code

For comparison, the following two notebooks in `\R` folder solve the ZINB model using R packages:

`R/ZINB_WaVE_R.ipynb`: ZINB_WaVE solver  
`R/VGAM_R.ipynb`: VGAM solver

## Tests

In `./tests`, we provide tests for each of the 7 models: 

- cross validate the supported methods `stan`, `statsmodels`, or `tensorflow` returning similar results on simulation data. 
- show simulation examples where `statsmodels` incurs numerical errors.
- show `MAST` implemented in Python returns identical results as the original `R` implementation.


## Troubleshooting

To run TensorFlow on Apple silicon (M1, M2, etc), install TensorFlow using the following:

`conda install -c apple tensorflow-deps`

`python -m pip install tensorflow-macos==2.9.2`

`python -m pip install tensorflow-metal==0.5.1`


## References
Cui, T., Wang, T. A Comprehensive Assessment of Hurdle and Zero-inflated Models for Single Cell RNA-sequencing Analysis (2023).

## Support and Contribution
For technical issues particular to this repo, please report the issue on this GitHub repository.


<!-- ``` r
Poi:   Poisson
NB:    Negative Binomial
PoiH:  Poisson Hurdle
NBH:   Negative Binomial Hurdle
ZIPoi: Zero-inflated Poisson
ZINB:  Zero-inflated Negative Binomial 
MAST:  Normal Hurdle
```

``` r
methods={
    Poi: ['stan','statsmodels'],
    NB: ['stan','statsmodels','tensorflow'],
    PoiH: ['stan','statsmodels'],
    NBH: ['stan','statsmodels'],
    ZIPoi: ['stan','statsmodels'],
    ZINB: ['stan','statsmodels','tensorflow'],
    MAST: ['statsmodels'],
}
```  -->