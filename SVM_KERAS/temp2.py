from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import seaborn as sn

from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt

# Digits datasetのロード
digits = load_digits()

X = digits.data
y = digits.target

# 学習データと評価データに分割（分割比率8:2）
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)


# デフォルトパラメタ# デフォルト
estimator = SVC()
classifier = OneVsRestClassifier(estimator = estimator)

classifier.fit(X_train, y_train)

# 評価データでconfusion matrixとaccuracy scoreを算出
pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print('Multiclass SVM(default): %.3f' % accuracy)

# Confusion Matrixをheatmapとして表示# Confu
def ConfusionMatrixHeatmap(y_true, y_pred):
    labels = sorted(list(set(y_true)))

    cf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:,np.newaxis]

    plt.imshow(cf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# チューニングモデル(1回目)
model = OneVsRestClassifier(SVC())

C_params = [10,100,1000,10000,100000,1000000]
gamma_params = [0.01,0.001,0.0001,0.00001]

parameters = {
    'estimator__C': C_params,
    'estimator__gamma': gamma_params
}

model_tuning = GridSearchCV(
    estimator = model,
    param_grid = parameters,
    verbose = 3,

)

model_tuning.fit(X_train, y_train)

def PlotGridSearchScores(model_tuning, x_param, line_param):
    x_values = model_tuning.cv_results_['param_' + x_param].data
    x_labels = np.sort(np.unique(x_values))
    x_keys = ['{0:9.2e}'.format(x) for x in x_labels]

    line_values = model_tuning.cv_results_['param_' + line_param].data
    line_labels = np.sort(np.unique(line_values))
    line_keys = ['{0:9.2e}'.format(v) for v in line_labels]

    score = {}

    # (line_key, x_key) -> mean_test_scoreを生成
    for i, test_score in enumerate(model_tuning.cv_results_['mean_test_score']):
        x = x_values[i]
        line_value = line_values[i]

        x_key = '{0:9.2e}'.format(x)
        line_key = '{0:9.2e}'.format(line_value)

        score[line_key, x_key] = test_score

    _, ax = plt.subplots(1,1)

    # 対数軸で表示する
    plt.xscale('log')

    # x_paramをx軸、line_paramを折れ線グラフで表現
    for line_key in line_keys:
        line_score = [score[line_key, x_key] for x_key in x_keys]
        ax.plot(x_labels, line_score, '-o', label=line_param + ': ' + line_key)

    ax.set_title("Grid Search Accuracy Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(x_param, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 0.95, 0.5, .100), fontsize=15)
    ax.grid('on')

def plot_heatmap_from_grid(grid):
    # チューニング対象のパラメータを特定する。
    params = [k for k in grid.cv_results_.keys() if k.startswith('param_')]
    if len(params) != 2: raise Exception('grid has to have exact 2 parameters.')

    # ヒートマップの行、列、値に使うキーを定義する。
    index = params[0]
    columns = params[1]
    values = 'mean_test_score'

    # gridから必要なキーのみを抽出する。
    df_dict = {k: grid.cv_results_[k] for k in grid.cv_results_.keys() & {index, columns, values}}

    # dictをDataFrameに変換してseabornでヒートマップを表示する。
    import pandas as pd
    df = pd.DataFrame(df_dict)
    data = df.pivot(index=index, columns=columns, values=values)
    import seaborn as sns
    sns.heatmap(data, annot=True, fmt='.3f')


# チューニング結果を描画
PlotGridSearchScores(model_tuning, 'estimator__gamma', 'estimator__C')

# Best parameter
model_tuning.best_params_
print('best:',model_tuning.best_params_)


# 評価データでconfusion matrixとaccuracy scoreを算出# 評価データ
classifier_tuned = model_tuning.best_estimator_
pred = classifier_tuned.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print('Multiclass SVM(default): %.3f' % accuracy)



#plot_heatmap_from_grid(model_tuning)
plt.show()