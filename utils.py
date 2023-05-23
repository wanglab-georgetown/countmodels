import numpy as np
from numpy.linalg import matrix_rank
from patsy import dmatrices


def remove_list(A, B):
    return [e for e in list(A) if e not in list(B)]


def find_independent_matrix(df, keep_cols=[]):
    original_rank = matrix_rank(df)
    N = np.shape(df)[1]
    cur_cols = remove_list(list(range(N)), list(keep_cols))
    while len(cur_cols) > 0:
        df_sel = df[:, cur_cols + keep_cols]
        if matrix_rank(df_sel) == np.shape(df_sel)[1]:
            return df_sel
        for c in cur_cols:
            df_sel = df[:, list(set(cur_cols + keep_cols) - set([c]))]
            if matrix_rank(df_sel) == original_rank:
                cur_cols.remove(c)
                break
    return []


def find_independent_columns(df, keep_cols=[]):
    original_rank = matrix_rank(df)
    cur_cols = remove_list(list(df.columns), list(keep_cols))
    while len(cur_cols) > 0:
        df_sel = df[cur_cols + keep_cols]
        if matrix_rank(df_sel) == np.shape(df_sel)[1]:
            return cur_cols
        for c in cur_cols:
            df_sel = df[list(set(cur_cols + keep_cols) - set([c]))]
            if matrix_rank(df_sel) == original_rank:
                cur_cols.remove(c)
                break
    return []


def gen_feature_df(df_feature, features, add_intercept=True):
    if add_intercept:
        if len(features) == 0:
            formula = "{} ~ {}".format("1", "1")
        else:
            formula = "{} ~ {}".format("1", " + ".join(features))
    else:
        if len(features) == 0:
            return None
        formula = "{} ~ {} - 1".format("1", " + ".join(features))

    _, predictors = dmatrices(formula, df_feature, return_type="dataframe")
    return predictors


def find_independent_features(df_feature, features, keep_cols=[], add_intercept=True):
    df = gen_feature_df(df_feature, features, add_intercept=add_intercept)
    original_rank = matrix_rank(df)
    cur_cols = remove_list(list(features), list(keep_cols))
    while len(cur_cols) > 0:
        df_sel = gen_feature_df(
            df_feature, cur_cols + keep_cols, add_intercept=add_intercept
        )
        if matrix_rank(df_sel) == np.shape(df_sel)[1]:
            return cur_cols
        for c in cur_cols:
            df_sel = gen_feature_df(
                df_feature,
                list(set(cur_cols + keep_cols) - set([c])),
                add_intercept=add_intercept,
            )
            if matrix_rank(df_sel) == original_rank:
                cur_cols.remove(c)
                break
    return []
