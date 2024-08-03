# Задача по классификации (BANK MODEL)



# Импорт библиотек
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn

# Запуск проекта
df = pd.read_csv("/content/Churn_Modelling.csv")

# ознакомление с дата сетом
df.head()

df.info()

# созлание новых колонок из существующих
df['Geography'].value_counts()
df[['France', 'Germany', 'Spain']] = pd.get_dummies(df['Geography'], dtype=int)
df.drop(columns=['Geography'], inplace=True)
df['Gender'].value_counts()
df[['Male', 'Female']] = pd.get_dummies(df['Gender'], dtype=int)
df.drop(columns=['Gender'], inplace=True)

# Моделирование
# выбор элементов входных данных
df.isnull().sum()
df.dropna(axis=0, inplace=True)
df.columns

X = df[['CreditScore', 'Age', 'Tenure',
       'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
       'EstimatedSalary', 'France', 'Germany', 'Spain', 'Male',
       'Female']]

Y = df['Exited']

# Разделение набора данных на тренировочный и тестовый мини наборы (train, test split)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.10, random_state=6)
X_train.shape, X_test.shape


# Универсальная функция для оценки модели по всем метрикам регрессии
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score
def all_classif_scores(model, name_model, X_test, Test_y):
    ACC = round(accuracy_score(Y_test, model.predict(X_test)), 4)
    Bal_ACC = round(balanced_accuracy_score(Y_test, model.predict(X_test)), 4)
    RECALL = round(recall_score(Y_test, model.predict(X_test), average='micro'), 4)
    PRECISION = round(precision_score(Y_test, model.predict(X_test), average='micro'), 4)
    F1 = round(f1_score(Y_test, model.predict(X_test), average='micro'), 4)
    print(f'{name_model} model: \n', '      Acc: {0}     Bal_Acc: {1}     Recall: {2}     Precision: {3}     F1: {4}'.format(ACC, Bal_ACC, RECALL, PRECISION, F1))

# Logistic regression (LR)
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(max_iter=200)
LR.fit(X_train, Y_train)
LR.score(X_test, Y_test)
y_pred = LR.predict(X_test)
all_classif_scores(LR, 'LR', X_test, Y_test)

# Создание графика
plt.figure(figsize=(17, 3))
plt.plot(y_pred, color='r', label='прогнозные')
plt.plot(Y_test, color='g', label='реальные')
plt.grid()
plt.legend()
plt.figure(figsize=(17, 3))
plt.plot(pd.DataFrame(y_pred).sort_values(0).reset_index().drop('index', axis=1), color='r', label='прогнозные')
plt.plot(pd.DataFrame(Y_test).sort_values(0).reset_index().drop('index', axis=1), color='g', label='реальные')
plt.grid()
plt.legend()


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(criterion='gini',
                             splitter='best',
                             max_depth=9,
                             min_samples_split=2,
                             min_samples_leaf=1,
                             min_weight_fraction_leaf=0.0,
                             max_features=None,
                             random_state=13,
                             max_leaf_nodes=None,
                             min_impurity_decrease=0.0,
                             class_weight=None,
                             ccp_alpha=0.0)
DTC.fit(X_train, Y_train)
DTC.score(X_test, Y_test)
all_classif_scores(DTC, 'DTC', X_test, Y_test)


from sklearn.ensemble import BaggingClassifier
BGC = BaggingClassifier(estimator=LogisticRegression(),
                        n_estimators=10,
                        max_samples=1.0,
                        max_features=1.0,
                        bootstrap=True,
                        bootstrap_features=False,
                        oob_score=False,
                        warm_start=False,
                        n_jobs=None,
                        random_state=13,
                        verbose=0,
                        base_estimator='deprecated')
BGC.fit(X_train, Y_train.ravel())
BGC.score(X_test, Y_test.ravel())
all_classif_scores(BGC, 'BGC', X_test, Y_test)
Pred_BGC = BGC.predict(X_test)

# Bagging Classifier
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100,
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=4,
                            min_weight_fraction_leaf=0.0,
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.0,
                            bootstrap=True,
                            oob_score=False,
                            n_jobs=None,
                            random_state=None,
                            verbose=0,
                            warm_start=False,
                            ccp_alpha=0.0,
                            max_samples=None,)
RFC.fit(X_train, Y_train.ravel())
RFC.score(X_test, Y_test.ravel())
all_classif_scores(RFC, 'RFC', X_test, Y_test)
columns = X.columns


# Random Forest Classifier
from sklearn.ensemble import ExtraTreesClassifier
ExTC = ExtraTreesClassifier(n_estimators=100,
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.0,
                            bootstrap=False,
                            oob_score=False,
                            n_jobs=None,
                            random_state=66,
                            verbose=0,
                            warm_start=False,
                            ccp_alpha=0.0,
                            max_samples=None,)
ExTC.fit(X_train, Y_train.ravel())
ExTC.score(X_test, Y_test.ravel())
all_classif_scores(ExTC, 'ExTC', X_test, Y_test)

# Adaptive Boosting Classifier
from sklearn.ensemble import AdaBoostClassifier
AdBC = AdaBoostClassifier(random_state=0, n_estimators=100)
AdBC.fit(X_train, Y_train.ravel())
AdBC.score(X_test, Y_test.ravel())
all_classif_scores(AdBC, 'AdBC', X_test, Y_test)
from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(
                                learning_rate=0.1,
                                n_estimators=100)
GBC.fit(X_train, Y_train.ravel())
GBC.score(X_test, Y_test.ravel())


# TOTAL Results
all_classif_scores(GBC, 'GBC', X_test, Y_test)
all_classif_scores(LR, 'LC', X_test, Y_test)
all_classif_scores(DTC, 'DTC', X_test, Y_test)
all_classif_scores(BGC, 'BGC', X_test, Y_test)
all_classif_scores(RFC, 'RFC', X_test, Y_test)
all_classif_scores(ExTC, 'ExTC', X_test, Y_test)
all_classif_scores(AdBC, 'AdBC', X_test, Y_test)


df['Exited'].value_counts()
Y_test.sum()
model_pred = ExTC.predict(X_test)
MODEL = ExTC

# Создание модели
import seaborn as sns
from sklearn.metrics import confusion_matrix

dataset_labels = np.array(["0", "1"])   # dataset_labels = np.array(["1_норма", "2_сомнительно", "3_патология"])          #["1_Norma", "2_Somnitelno", "3_Patologya"]
confusion_TEST = confusion_matrix(Y_test, model_pred)
df_TEST = pd.DataFrame(confusion_TEST,
                      dataset_labels,
                      dataset_labels)
sns.set(font_scale=1.2)                    # for label size
plt.figure(figsize=(5, 4))
sns.heatmap(df_TEST,
            annot=True,
            annot_kws={"size": 18},       # font size
            fmt = "d",
            #fmt='.2f',                   # precision (2 digits)
            linewidths=.02,
            cmap="YlGnBu",
            linecolor='b',
            cbar=True)

plt.title(str(f'Матрица невязок (Confusion Matrix) для алгоритма \n{MODEL}'), fontsize = 12)
plt.ylabel("действительные классы", fontsize = 18, color = "blue")
plt.xlabel("распознанные классы", fontsize = 18, color = "blue")
plt.xticks(fontsize = 14, rotation=90)
plt.yticks(fontsize = 14)
#plt.savefig('Confusion Matrix of results.jpg', dpi = 300, bbox_inches='tight', pad_inches=0.03)
plt.show()
