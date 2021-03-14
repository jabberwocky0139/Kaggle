import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from collections import deque
from sklearn.metrics import roc_auc_score


# -----------------------------------
# 汎用関数の定義
# -----------------------------------

def rf_importance(train_x, train_y):
    ''' RandomForestで重要度を推定'''
    from sklearn.ensemble import RandomForestClassifier
    model_rf = RandomForestClassifier(n_estimators=10, max_features='auto')
    model_rf.fit(train_x, train_y)

    ranking = np.argsort(-model_rf.feature_importances_)
    f, ax = plt.subplots(figsize=(11, 9))
    sns.barplot(x=model_rf.feature_importances_[ranking],
                y=train_x.columns.values[ranking],
                orient='h')
    ax.set_xlabel('feature iportance')
    plt.tight_layout()
    plt.show()


def lgbm_classifier(train_x, train_y, test_x):
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold
    from random import randint

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'force_col_wise': True,
        'metric': {'auc'}
    }
    num_round = 8000

    from sklearn.model_selection import train_test_split
    tr_x, va_x, tr_y, va_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
    lgb_train = lgb.Dataset(tr_x, tr_y)
    lgb_eval = lgb.Dataset(va_x, va_y)

    model = lgb.train(params,
                      train_set=lgb_train,
                      valid_sets=lgb_eval,
                      num_boost_round=num_round,
                      early_stopping_rounds=100,
                      verbose_eval=100)

    pred = model.predict(test_x)

    return pred


def stacking_classifier(train_x, train_y, test_x):
    import lightgbm as lgb
    from rgf.sklearn import RGFClassifier
    from sklearn.ensemble import StackingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.metrics import mean_squared_error

    lgb_params = {
            # 'boosting': 'gbdt',
            'application': 'classifier',
            # 'learning_rate': 0.05,
            # 'min_data_in_leaf': 20,
            # 'feature_fraction': 0.7,
            # 'num_leaves': 41,
            'metric': 'auc'
            # 'drop_rate': 0.15
    }

    et_params = {'n_estimators': 20,
                 'max_features': 0.5,
                 'max_depth': 18,
                 'min_samples_leaf': 4,
                 'n_jobs': -1}

    rf_params = {'n_estimators': 20,
                 'max_features': 0.2,
                 'max_depth': 25,
                 'min_samples_leaf': 4,
                 'n_jobs': -1}

    rgf_params = {'algorithm': 'RGF_Sib',
                  'loss': 'Log'}

    kn_params = {'leaf_size': 10}



    estimators = [
        ('lgb', lgb.LGBMClassifier(**lgb_params)),
        # ('rgf', RGFClassifier(**rgf_params)),
        ('et', ExtraTreesClassifier(**et_params)),
        ('rf', RandomForestClassifier(**rf_params)),
        ('lr', LogisticRegression())
        # ('knn', KNeighborsClassifier(**kn_params))
    ]

    model_stack = StackingClassifier(estimators=estimators,
                                     final_estimator=LogisticRegression(),
                                     verbose=1)
    model_stack.fit(train_x, train_y)

    pred = model_stack.predict(test_x)

    return pred

# -----------------------------------
# 学習データ、テストデータの読み込み
# -----------------------------------

# train.csv - the training data with the target column
# test.csv - the test set; you will be predicting the target for each row in this file (the probability of the binary target)
# sample_submission.csv - a sample submission file in the correct format

# 学習データ、テストデータの読み込み
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
sample_submission = pd.read_csv('./data/sample_submission.csv')


# -----------------------------------
# 特徴量作成
# -----------------------------------

# trainとtestのfloat型を16bitにシュリンク
for name in train.dtypes[train.dtypes=='float64'].index.tolist():
    train[name] = train[name].astype('float16')
    test[name] = test[name].astype('float16')

# shops_nameをLabelEncoding
le = LabelEncoder()
for name in train.dtypes[train.dtypes=='object'].index.tolist():
    train[name] = le.fit_transform(train[name]).astype('int16')
    test[name] = le.fit_transform(test[name]).astype('int16')

# データセットの分割
train_y = train['target']
train_x = train.drop('target', axis=1)
test_x = test.copy()

# RandomForestで重要度を推定
# rf_importance(train_x, train_y)


# -----------------------------------
# モデル作成
# -----------------------------------

# モデルの訓練と推定
# pred = lgbm_classifier(train_x, train_y, test_x)
pred_stack = stacking_classifier(train_x, train_y, test_x)

# 0/1ラベルに変換
# pred_label = np.where(pred > 0.5, 1, 0)
pred_stacking_label = np.where(pred_stack > 0.5, 1, 0)

# 提出用ファイルの作成
submission = pd.DataFrame({'id': test_x.id, 'target': pred_stacking_label})
submission.to_csv('submission_second.csv', index=False)
