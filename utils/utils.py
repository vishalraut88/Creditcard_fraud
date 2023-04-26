from functools import partial
from typing import Dict

import pandas as pd
import numpy as np

from dython.nominal import associations
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
import warnings
import lightgbm as lgb
# from typing import Dict
# from dython.nominal import associations
# from hyperopt import hp, tpe, Trials

class feature_importances:

    def __init__(self):
        pass

    def filter_out_features_based_on_statistical_approach(self,
            X_train_feature_selection,
            disallowed_columns,
            target_column = 'Class'
    ):
    #     X_train_feature_selection = x_train_tmp.copy(deep=True)
        for col in X_train_feature_selection.columns:
            if X_train_feature_selection[col].isna().sum() /\
                    len(X_train_feature_selection) > 0.5:
                if col not in disallowed_columns:
                    disallowed_columns.append(col)
        print("Removing columns: [" + ", ".join(disallowed_columns) + "]")
        for column in disallowed_columns:
            if column in X_train_feature_selection.columns:
                X_train_feature_selection.drop(columns=[column], inplace=True)
        assocs = associations(X_train_feature_selection, compute_only=True)
        assoc_result = pd.DataFrame(assocs['corr'])
        for col in assoc_result.columns:
            assoc_result[col] = assoc_result[col].astype(float)
            assoc_result[col] = round(assoc_result[col], 2)
        assoc_result[target_column] = abs(assoc_result[target_column])
        accepted_columns = []
        midpoint = assoc_result[target_column].quantile(0.75)
        for row in zip(X_train_feature_selection.columns, assoc_result[target_column]):
            if row[1] > midpoint and row[0] not in disallowed_columns:
                accepted_columns.append(row[0])
        X_train_feature_selection = X_train_feature_selection[accepted_columns]
        X_train_feature_selection = X_train_feature_selection.drop(
            columns=[target_column]
        )
        return X_train_feature_selection, accepted_columns,assoc_result,disallowed_columns
    
    

def feature_imp_lgbm(Xtrain,y_train,X_val,y_val,params_scope):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        initial_params = initial_hyperparam_search(Xtrain, y_train, X_val[Xtrain.columns], y_val, params_scope)
    
    params = cast_params_to_proper_types(initial_params)
    model = lgb.LGBMClassifier(
            random_state=42,
            verbose=-1,
            objective='binary'
            ,**params
    )   
    
    model.fit(
    Xtrain,
    y_train,
    eval_set=[(X_val[Xtrain.columns], y_val)],
    verbose=0,
    eval_metric=lgb_f1_score
)
    return model,initial_params


def cast_params_to_proper_types(params):

    return {
        'learning_rate': float(params['learning_rate']),
        'num_leaves': int(params['num_leaves']),
        'n_estimators': int(params['n_estimators']),
        'scale_pos_weight': int(params['scale_pos_weight']),
        'max_bin': int(params['max_bin'])
    }


def objective(params, x_tmp_train, y_tmp_train):
    params = cast_params_to_proper_types(params)
    kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    model = lgb.LGBMClassifier(
            random_state=42,
            verbose=-1,
            objective='binary',
            **params
        )
    score = cross_val_score(
        model,
        x_tmp_train,
        y_tmp_train,
        scoring='f1',
        cv=kfold,
        n_jobs=-1
    ).mean()
    return 1 - score


def objective_under_validation_set(
    params,
    x_tmp_train,
    y_tmp_train,
    x_tmp_validation,
    y_tmp_validation
):
    params = cast_params_to_proper_types(params)
    model = lgb.LGBMClassifier(
            random_state=42,
            verbose=-1,
            objective='binary',
            **params
    )
    model.fit(
        x_tmp_train,
        y_tmp_train,
        eval_set=[(x_tmp_validation, y_tmp_validation)],
        verbose=0,
        eval_metric=lgb_f1_score
    )
    return 1 - f1_score(
        y_tmp_validation,
        model.predict(x_tmp_validation)
        )


def lgb_f1_score(y_true, y_proba):
    y_hat = np.where(y_proba < 0.5, 0, 1)  
    return 'f1', f1_score(y_true, y_hat), True


def initial_hyperparam_search(
x_train_tmp,
y_train_tmp,
x_val_tmp,
y_val_tmp,
params_scope):

    fmin_objective = partial(
        objective_under_validation_set,
        x_tmp_train=x_train_tmp,
        y_tmp_train=y_train_tmp,
        x_tmp_validation=x_val_tmp,
        y_tmp_validation=y_val_tmp
    )
    best = fmin(
        fn=fmin_objective,
        space=params_scope,
        algo=tpe.suggest,
        max_evals=250,
    )
    return best


def perform_rfe(
    x_train_tmp,
    y_train_tmp,
    x_val_tmp,
    y_val_tmp,
    initial_params: Dict,
    minimal_importance: float = 0.01,
    minimal_amount_of_features_to_keep: int = 10,
    n_repeats: int = 10
):

    n_features = []
    scores_per_features = []
    while True:
        feature_importance = pd.DataFrame(
            {'column': x_train_tmp.columns}
        )
        split = 0
        n_features.append(len(x_train_tmp.columns))
        current_scores = []
        for ix in range(0, n_repeats):
            model = lgb.LGBMClassifier(
                verbose=-1,
                objective='binary',
                **initial_params
            )
            model.fit(
                x_train_tmp,
                y_train_tmp,
                eval_set=[(x_val_tmp, y_val_tmp)],
                eval_metric=lgb_f1_score,
                verbose=0
            )
            current_scores.append(f1_score(y_val_tmp, model.predict(x_val_tmp)))
            feature_importance[f'fi_{split}'] = model.feature_importances_
            split += 1
        feature_importance['mean'] = feature_importance[
            feature_importance.columns[1:]
        ].mean(axis=1)
        #
        fi_min = min(feature_importance['mean'])
        fi_max = max(feature_importance['mean'])
        fi_norm = (feature_importance['mean'] - fi_min) / (fi_max - fi_min)
        feature_importance['mean'] = fi_norm
        feature_importance['mean'] = round(feature_importance['mean'], 2)
        #
        idx_to_remove = feature_importance['mean'].idxmin()
        scores_per_features.append(np.mean(current_scores))
        print(feature_importance['mean'].min(), feature_importance.loc[
            idx_to_remove, 'column']
            )
        # print(feature_importance)
        if feature_importance['mean'].min() >= minimal_importance:
            break
        if len(feature_importance) <= minimal_amount_of_features_to_keep:
            break
        x_train_tmp = x_train_tmp.drop(
            columns=[feature_importance.loc[idx_to_remove, 'column']]
            )
        x_val_tmp = x_val_tmp.drop(
            columns=[feature_importance.loc[idx_to_remove, 'column']]
            )
    to_keep = list(x_train_tmp.columns)
    return to_keep, n_features, scores_per_features
