from .supervised_learning import function

import numpy as np
np.random.seed(seed=32)

# 表示オプションの変更
import pandas as pd
pd.options.display.max_columns = 100
pd.set_option('display.max_rows', 500)

evaluation_list = {'AUC':'roc_auc',
                   'F1':'f1',
                   'Recall':'recall',
                   'Precision':'precision',
                   'Accuracy':'accuracy'}
options_algorithm = ['lightgbm', 'knn', 'ols', 'ridge', 'tree', 'rf', 'gbr1', 'gbr2', 'xgboost', 'catboost']

feature_importances_algorithm_list = ['tree', 'rf', 'gbr1', 'gbr2', 'lightgbm', 'xgboost', 'catboost']

exception_algorithm_list = ['lightgbm', 'tree', 'gbr1', 'gbr2']
exception_algorithm_list = ['tree', 'gbr1', 'gbr2', 'knn', 'rf', 'ols', 'ridge', 'xgboost']
exception_algorithm_list = ['tree', 'gbr1', 'gbr2', 'ols', 'ridge', 'rf']
exception_algorithm_list = ['tree', 'gbr1', 'gbr2', 'ols', 'ridge', 'rf', 'knn', 'ols', 'xgboost']

def setup_algorithm(pipelines = {}, is_predict_proba = False):
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC, LinearSVC
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import RFE
    lgbm_params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'boosting_type': 'gbdt', 'feature_pre_filter': False, 'lambda_l1': 9.65103630752439, 'lambda_l2': 4.015485886216817, 'num_leaves': 5, 'feature_fraction': 0.4,
              'bagging_fraction': 0.8776114298029899, 'bagging_freq': 3,
               'min_child_samples': 20,
               'random_state' : 1
    }
    lgbm_params = {
        'random_state' : 1,
        #'objective' : 'binary',
        #'metric' : 'binary_logloss',
        #'device' : 'gpu'
    }
    #lgbm_params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'boosting_type': 'gbdt', 'feature_pre_filter': False, 'lambda_l1': 7.448574459066696e-08, 'lambda_l2': 1.3771631987966848e-05, 'num_leaves': 3, 'feature_fraction': 0.516, 'bagging_fraction': 0.8471270267389193, 'bagging_freq': 5, 'min_child_samples': 20}
    catboost_params = {
        #'task_type' : 'GPU',
        'random_state' : 1,
        #'eval_metric' : 'Logloss',
        #'num_boost_round' : 10000
    }
    xgboost_params = {
        'objective' : 'binary:hinge',
        'tree_method' : 'gpu_hist',
        'random_state' : 1,
    }
    pipelines = {
        'lightgbm':
            Pipeline([
                    ('pca', PCA(random_state=1)),
                    #('est', lgb.LGBMClassifier(random_state=1, objective='binary', metric='binary_logloss', device='gpu'))]),
                    ('est', lgb.LGBMClassifier(random_state=1, objective='binary', metric='binary_logloss'))]),
        'xgboost':
            Pipeline([
                    ('pca', PCA(random_state=1)),
                    #('est', xgb.XGBClassifier(random_state=1, objective='binary:hinge', tree_method='gpu_hist'))]),
                    ('est', xgb.XGBClassifier(random_state=1, objective='binary:hinge'))]),
        'catboost':
            Pipeline([('pca', PCA(random_state=1)),
                    #('est', CatBoostClassifier(random_state=1, eval_metric='Logloss', task_type='GPU', num_boost_round=10000))]),
                    ('est', CatBoostClassifier(random_state=1, eval_metric='Logloss', num_boost_round=10000))]),
        'knn':
            Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(random_state=1)),
                    ('est', KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski'))]),
        'logistic':
            Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(random_state=1)),
                    ('est', LogisticRegression(random_state=1, max_iter=1000))]),
        'rsvc':
            Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(random_state=1)),
                    ('est', SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=is_predict_proba, random_state=1))]),
        'tree':
            Pipeline([('pca', PCA(random_state=1)),
                    ('est', DecisionTreeClassifier(random_state=1))]),
        'rf':
            Pipeline([
                    ('pca', PCA(random_state=1)),
                    ('est', RandomForestClassifier(random_state=1))]),
        'gb':
            Pipeline([('pca', PCA(random_state=1)),
                    ('est', GradientBoostingClassifier(random_state=1))]),
        'mlp':
            Pipeline([
                    ('scl', StandardScaler()),
                    ('pca', PCA(random_state=1)),
                    ('est', MLPClassifier(hidden_layer_sizes=(3,3), max_iter=10000, random_state=1))])
    }
    return pipelines

tuning_prarameter_list = []
# パラメータグリッドの設定
tuning_prarameter = {
    'lightgbm':{
        'est__learning_rate': [0.1,0.05,0.01],
        'est__n_estimators':[1000,2000],
        'est__num_leaves':[31,15,7,3],
        'est__max_depth':[4,8,16]
    },
    'tree':{
        "est__min_samples_split": [10, 20, 40],
        "est__max_depth": [2, 6, 8],
        "est__min_samples_leaf": [20, 40, 100],
        "est__max_leaf_nodes": [5, 20, 100],
    },
    'rf':{
        'est__n_estimators':[5,10,20,50,100],
        'est__max_depth':[1,2,3,4,5],
    },
    'knn':{
        'est__n_neighbors':[1,2,3,4,5,],
        'est__weights':['uniform','distance'],
        'est__algorithm':['auto','ball_tree','kd_tree','brute'],
        'est__leaf_size':[1,10,20,30,40,50],
        'est__p':[1,2]
    },
    'logistic':{
        'pca__n_components':[5,7,9],
        'est__C':[0.1,1.0,10.0,100.0],
        'est__penalty':['l2'],
        'est__max_iter':[1000],
        'est__intercept_scaling':[1]
    },
    'gb':{
        'est__loss':['deviance','exponential'],
        'est__n_estimators':[5,10,50,100,500],
    }
}

def evaluation(output_data_dir, scores, pipelines, X_train, y_train, phase, evaluation_function_list,
              input_evaluation, is_optuna = False, is_sklearn = False, is_predict_proba = False):
    print(input_evaluation.value)
    for pipe_name, _ in pipelines.items():
        scores[(pipe_name, phase)] = evaluation_function_list[input_evaluation.value](y_train[pipe_name],
                                        scoring(output_data_dir, pipe_name, X_train[pipe_name], input_evaluation,
                                        is_optuna, is_sklearn, is_predict_proba))

def scoring(out_put_data_dir, algorithm_name :str, X, input_evaluation, is_optuna = False, is_sklearn = False, is_predict_proba = False):
    from joblib import load
    clf = load(out_put_data_dir + 'model/' + algorithm_name + '_classiffier.joblib')
    print(algorithm_name)
    if algorithm_name in 'lightgbm' and (is_optuna or not is_sklearn):
        if input_evaluation.value == 'Accuracy':
            return clf.predict(X).round()
        return clf.predict(X)
    print(algorithm_name)
    if is_predict_proba and is_sklearn:
        print('predict_proba')
        return clf.predict_proba(X)[:, 1]
    print('predict')
    return clf.predict(X)

def cross_validatior(kf, scorer, output_data_dir, n_features_to_select, pipelines, input_evaluation, X_train, y_train):
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import StratifiedKFold
    str_all_print = 'n_features_to_select:' + str(n_features_to_select) + '\n'
    print('評価指標:' + input_evaluation.value)
    str_print = ''
    for pipe_name, est in pipelines.items():
        cv_results = cross_val_score(est,
                                    X_train, y_train,
                                    cv=kf,
                                    scoring=scorer)  
        str_print = '----------' + '\n' + 'algorithm:' + str(pipe_name) + '\n' + 'cv_results:' + str(cv_results) + '\n' + 'avg +- std_dev ' + str(cv_results.mean()) + '+-' + str(cv_results.std()) + '\n'
        print(str_print)
        str_all_print += str_print
    import datetime
    with open(output_data_dir + 'cv_results' + '_' + str(n_features_to_select) + "_" +  datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.txt', mode='w') as f:
        f.write(str_all_print)

def decide_evaluation(input_evaluation):
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import log_loss
    function_evaluation = mean_squared_error
    if input_evaluation.value == 'LOG_LOSS':
        function_evaluation = log_loss
    if input_evaluation.value == 'RMSE':
        function_evaluation = root_mean_squared_error
    elif input_evaluation.value == 'MAE':
        function_evaluation = mean_absolute_error
    elif input_evaluation.value == 'R2':
        function_evaluation = r2_score
    elif input_evaluation.value == 'RMSLE':
        function_evaluation = root_mean_squared_log_error
    return function_evaluation

def display_evaluation(is_holdout, output_data_dir, pipelines, function_evaluation, input_evaluation,
                       X_train, y_train, X_valid, y_valid, is_optuna, is_sklearn, is_predict_proba):
    scores = {}
    if is_holdout:
        evaluation(output_data_dir, scores, pipelines, X_train, y_train, 'train', function_evaluation, input_evaluation, is_optuna, is_sklearn, is_predict_proba)
        evaluation(output_data_dir, scores, pipelines, X_valid, y_valid, 'valid', function_evaluation, input_evaluation, is_optuna, is_sklearn, is_predict_proba)
    else:
        evaluation(output_data_dir, scores, pipelines, X_train, y_train, 'train', function_evaluation, input_evaluation, is_optuna, is_sklearn, is_predict_proba)
    ascending = True
    print('評価指標:' + input_evaluation.value)
    if is_holdout:
        display(pd.Series(scores).unstack().sort_values(by=['valid'], ascending=[ascending]))
    else:
        display(pd.Series(scores).unstack().sort_values(by=['train'], ascending=[ascending]))

import datetime
def output_file(output_data_dir, n_features_to_select, target_label, df, id_label, y, model_name, extension, header=True):
    #y = [i if i > 0.01 else 0 for i in y]
    publish_file_name = output_data_dir + "submittion_" + model_name + "_" + str(n_features_to_select) + "_" +  datetime.datetime.now().strftime('%Y%m%d%H%M%S') + "." + extension
    file_name = output_data_dir + "submittion_" + model_name + "." + extension
    separator = ','
    if extension == 'tsv':
        separator = '\t'
    if id_label != '':
        df_output = pd.concat([df[id_label], pd.DataFrame(y, columns=target_label)], axis=1)
        df_output.to_csv(publish_file_name, index=False, sep=separator, header=header)
        df_output.to_csv(file_name, index=False, sep=separator, header=header)
        #concated = pd.concat([df, pd.DataFrame(y, columns=[target_label])], axis=1)
        #pd.DataFrame(concated[id_label, target_label]).to_csv(file_name, index=False, sep=separator, header=header)
        #concated = pd.concat([df, pd.DataFrame(y, columns=[target_label])], axis=1)
        #pd.DataFrame(concated[id_label, target_label]).to_csv(file_name, index=False, sep=separator, header=header)
        #concated[id_label, target_label].to_csv(file_name, index=False, sep=separator, header=header)
    else:
        df_output = pd.concat([df, pd.DataFrame(y, columns=target_label)], axis=1)
        df_output.to_csv(publish_file_name, index=False, sep=separator, header=header)
        df_output.to_csv(file_name, index=False, sep=separator, header=header)

# train
def train_model(output_data_dir, pipelines, X_train, X_valid, y_train, y_valid,
                evaluation, is_holdout, is_optuna=False, is_sklearn=True, categorical_feature = [],
                tuning_prarameter_list = [], tuning_prarameter = {}, evaluation_list = {}, y_column=''):
    if is_optuna:
        import optuna.integration.lightgbm as lgb
    elif not is_sklearn:
        import lightgbm as lgb
    from joblib import dump
    from sklearn.model_selection import GridSearchCV
    for pipe_name, pipeline in pipelines.items():
        print(pipe_name)
        if pipe_name in tuning_prarameter_list:
            gs = GridSearchCV(estimator=pipeline,
                        param_grid=tuning_prarameter[pipe_name],
                        scoring=evaluation_list[evaluation],
                        cv=3,
                        return_train_score=False)
            gs.fit(X_train, y_train)
            dump(gs, output_data_dir + pipe_name + '_classiffier.joblib')
            gs.fit(X_valid, y_valid)
            # 探索した結果のベストスコアとパラメータの取得
            print(pipe_name + ' Best Score:', gs.best_score_)
            print(pipe_name + ' Best Params', gs.best_params_)
        else:
            if pipe_name == 'lightgbm' and ((is_optuna) or (not is_sklearn)):
                lgb_train = lgb.Dataset(X_train[pipe_name], y_train[pipe_name],
                            #categorical_feature = categorical_feature
                            )
                lgb_eval = lgb.Dataset(X_valid[pipe_name], y_valid[pipe_name], reference=lgb_train)
                params = {
                    # 二値分類問題
                    'objective': 'binary',
                    # 損失関数は二値のlogloss
                    #'metric': 'auc',
                    #'metric': 'binary_logloss',
                    # 最大イテレーション回数指定
                    'num_iterations' : 1000,
                    # early_stopping 回数指定
                    'early_stopping_rounds' : 100,
                    #'metric': 'binary_logloss',
                    #'verbosity': -1,
                    #'boosting_type': 'gbdt',
                }
                #params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'boosting_type': 'gbdt', 'feature_pre_filter': False, 'lambda_l1': 8.033158760223655, 'lambda_l2': 1.1530347880300857e-07, 'num_leaves': 4, 'feature_fraction': 0.5, 'bagging_fraction': 0.9430649230190336, 'bagging_freq': 1, 'min_child_samples': 20}
                #params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'boosting_type': 'gbdt', 'feature_pre_filter': False, 'lambda_l1': 7.448574459066696e-08, 'lambda_l2': 1.3771631987966848e-05, 'num_leaves': 3, 'feature_fraction': 0.516, 'bagging_fraction': 0.8471270267389193, 'bagging_freq': 5, 'min_child_samples': 20}
                best = lgb.train(params,
                            lgb_train,
                            valid_sets=[lgb_train, lgb_eval],
                            verbose_eval=0,
                            #categorical_feature = categorical_feature
                            )
                dump(best, output_data_dir + 'model/' + pipe_name + '_classiffier.joblib') 
            else:
                print('normal')
                if pipe_name in 'lightgbm':
                    clf = pipeline.fit(X_train[pipe_name], y_train[pipe_name])
                else:
                    #X_train[pipe_name] = function.one_hot_encoding(X_train[pipe_name], categorical_feature)
                    clf = pipeline.fit(X_train[pipe_name], y_train[pipe_name])
                dump(clf, output_data_dir + 'model/' + pipe_name + '_classiffier.joblib')
                if is_holdout:
                    clf = pipeline.fit(X_valid[pipe_name], y_valid[pipe_name])
    return X_train, X_valid, y_train, y_valid
