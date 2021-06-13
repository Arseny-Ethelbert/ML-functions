from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.preprocessing import StandardScaler, PowerTransformer
from scipy.stats import kurtosis, skew, shapiro, normaltest
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns
import pandas as pd
import numpy as np


def analyze(data, title):
    plt.style.use('ggplot')
    plt.figure(figsize=(6,4))
    plt.hist(data, bins=60)
    plt.title(title, y=1.01, fontsize=14)
    plt.show()
    print('mean : ', np.mean(data))
    print('var  : ', np.var(data))
    print('skew : ', skew(data))
    print('kurt : ', kurtosis(data))
    print('shapiro : ', shapiro(data))
    print('normaltest : ', normaltest(data))

def value_of_metrics(y_true, y_pred, title):
    sns.set_style('whitegrid')
    print('Accuracy: ', accuracy_score(y_true, y_pred))
    print('Balanced accuracy: ', balanced_accuracy_score(y_true, y_pred))
    print('Recall: ', recall_score(y_true, y_pred))
    print('Precision: ', precision_score(y_true, y_pred))  
    print('F1: ', f1_score(y_true, y_pred))
    print('Roc_AUC: ', roc_auc_score(y_true, y_pred))
    print('Confusion Matrix: ')
    print(pd.DataFrame(confusion_matrix(y_true, y_pred)))
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.title(title, y=1.01, fontsize=14)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right', fontsize=11)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def value_of_metrics_cv(X, y, model, title):
    sns.set_style('whitegrid')
    st_fold = StratifiedKFold(n_splits=7, shuffle=True, random_state=5)
    params = {'estimator': model, 'X': X, 'y': y, 'cv': st_fold}
    scores = cross_validate(**params, scoring=('accuracy', 'balanced_accuracy',
                                               'recall', 'precision', 'f1',
                                               'roc_auc'))
    print('Accuracy: ', scores['test_accuracy'].mean())
    print('Balanced accuracy: ', scores['test_balanced_accuracy'].mean())
    print('Recall: ', scores['test_recall'].mean())
    print('Precision: ', scores['test_precision'].mean())
    print('F1: ', scores['test_f1'].mean())
    print('Roc_AUC: ', scores['test_roc_auc'].mean())
    conf_matrix_list_of_arrays = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    plt.close(fig)
    for i, (train, test) in enumerate(st_fold.split(X, y)):
        model.fit(X[train], y[train])
        viz = plot_roc_curve(model, X[test], y[test],
                                     name='ROC fold {}'.format(i), ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        conf_matrix = confusion_matrix(y[test], model.predict(X[test]))
        conf_matrix_list_of_arrays.append(conf_matrix)
    mean_сonf_matrix = np.round(np.mean(conf_matrix_list_of_arrays, axis=0), 0)
    print('Confusion Matrix:')
    print(pd.DataFrame(mean_сonf_matrix.astype(int)))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    roc_auc = auc(mean_fpr, mean_tpr)
    plt.title(title, y=1.01, fontsize=14)
    plt.plot(mean_fpr, mean_tpr, 'b', label = 'Mean  AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right', fontsize=11)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def value_of_metrics_cv_multiclass(X, y, model, title):
    sns.set_style('whitegrid')
    st_fold = StratifiedKFold(n_splits=7, shuffle=True, random_state=5)
    params = {'estimator': model, 'X': X, 'y': y, 'cv': st_fold}
    scores = cross_validate(**params, scoring=('accuracy', 'balanced_accuracy',
                                               'recall_micro', 'precision_micro',
                                               'f1_micro', 'roc_auc_ovr'))
    print('Accuracy: ', scores['test_accuracy'].mean())
    print('Balanced accuracy: ', scores['test_balanced_accuracy'].mean())
    print('Recall micro: ', scores['test_recall_micro'].mean())
    print('Precision micro: ', scores['test_precision_micro'].mean())
    print('F1 micro: ', scores['test_f1_micro'].mean())
    print('Roc_AUC ovr: ', scores['test_roc_auc_ovr'].mean(), '\n')
    y_pred = cross_val_predict(**params)
    print('Confusion Matrix:')
    print(pd.DataFrame(confusion_matrix(y, y_pred)))
    # binarize the output
    y_bin = label_binarize(y, classes=[x for x in range(y.nunique())])
    n_classes = y_bin.shape[1]
    # plot multiclass roc curve with cross-validation
    y_score = cross_val_predict(**params, method='predict_proba')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['orange', 'yellow', 'green', 'cyan', 'blue', 'magenta'])
    for i, color in zip(range(n_classes), colors):
         plt.plot(fpr[i], tpr[i], color=color,
                  label='Mean AUC of class {0} = {1:0.2f})'
                  ''.format(i, roc_auc[i]))
    plt.title(title, y=1.01, fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


class BestClassifier():

    def __init__(self, X_train, y_train, transformer=None):
        self.X_train = X_train
        self.y_train = y_train
        self.transformer = transformer

    def _make_pipeline(self, clf):
        if self.transformer is None:
            self.pipe = make_pipeline(clf)
        else:
            self.pipe = make_pipeline(self.transformer, clf)

    def _fit_gs_and_print_gs_results(self):
        gs = GridSearchCV(self.pipe, self.grid, cv=5, scoring='accuracy')
        gs.fit(self.X_train, self.y_train)
        self.params = gs.best_params_
        # remove the name of estimator from parameters names
        if '__' in list(self.params.keys())[0]:
            keys_list = [key.split('__')[1] for key in list(self.params.keys())]
            self.params = dict(zip(keys_list, self.params.values()))
        print('Best score (accuracy): {}'.format(gs.best_score_))
        print('Best parameters: {}\n'.format(self.params))
        print('С лучшими параметрами и кросс-валидацией:')

    def knn(self):
        estimator = 'kneighborsclassifier__'
        self.grid = {estimator +'n_neighbors': list(np.arange(5,37,2)),
                     estimator +'weights': ['uniform', 'distance'],
                     estimator +'leaf_size': list(np.arange(10,70,10)),
                     estimator +'p': [1, 2, 3, 4],
                     estimator +'metric': ['minkowski', 'chebyshev']}
        self._make_pipeline(KNeighborsClassifier())
        self._fit_gs_and_print_gs_results()
        self._make_pipeline(KNeighborsClassifier(**self.params))
        value_of_metrics_cv(self.X_train, self.y_train,
                            self.pipe, 'k Nearest Neighbors')

    def gaussian_nb(self):
        estimator = 'gaussiannb__'
        self.grid = {estimator +'var_smoothing': [1e-13, 1e-12, 1e-11, 1e-10,
                                                  1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
        self._make_pipeline(GaussianNB())
        self._fit_gs_and_print_gs_results()
        self._make_pipeline(GaussianNB(**self.params))
        value_of_metrics_cv(self.X_train, self.y_train,
                            self.pipe, 'Gaussian Naive Bayes')

    def bernoulli_nb(self):
        estimator = 'bernoullinb__'
        self.grid = {estimator +'alpha': [0.01, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0],
                     estimator +'binarize': [0.0, 0.2, 0.5, 1.0, 1.5],
                     estimator +'fit_prior': ['True', 'False']}
        self._make_pipeline(BernoulliNB())
        self._fit_gs_and_print_gs_results()
        self._make_pipeline(BernoulliNB(**self.params))
        value_of_metrics_cv(self.X_train, self.y_train,
                            self.pipe, 'Bernoulli Naive Bayes')

    def multinomial_nb(self):
        estimator = 'multinomialnb__'
        self.grid = {estimator +'alpha': [0.01, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0],
                     estimator +'fit_prior': ['True', 'False']}
        self._make_pipeline(MultinomialNB())
        self._fit_gs_and_print_gs_results()
        self._make_pipeline(MultinomialNB(**self.params))
        value_of_metrics_cv(self.X_train, self.y_train,
                            self.pipe, 'Multinomial Naive Bayes')

    def complement_nb(self):
        estimator = 'complementnb__'
        self.grid = {estimator +'alpha': [0.01, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0],
                     estimator +'fit_prior': ['True', 'False'],
                     estimator +'norm': ['True', 'False']}
        self._make_pipeline(ComplementNB())
        self._fit_gs_and_print_gs_results()
        self._make_pipeline(ComplementNB(**self.params))
        value_of_metrics_cv(self.X_train, self.y_train,
                            self.pipe, 'Complement Naive Bayes')

    def svm(self):
        estimator = 'svc__'
        self.grid = {estimator +'kernel': ['poly', 'rbf', 'sigmoid'],
                     estimator +'C': [0.5, 0.6, 0.7, 1, 1.5],
                     estimator +'degree': [1, 2, 3, 4, 5, 6],
                     estimator +'gamma': ['scale', 'auto'],
                     estimator +'class_weight': [None, 'balanced']}
        self._make_pipeline(svm.SVC())
        self._fit_gs_and_print_gs_results()
        self._make_pipeline(svm.SVC(**self.params))
        value_of_metrics_cv(self.X_train, self.y_train,
                            self.pipe, 'Support Vector Machine')

    def logic_regression(self):
        c_values = np.logspace(-2, 3, 500)
        st_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
        self._make_pipeline(LogisticRegressionCV(Cs=c_values, cv=st_fold))
        self.pipe.fit(self.X_train, self.y_train)
        c_result = self.pipe[-1].C_
        estimator = 'logisticregression__'
        self.grid = {estimator +'C': c_result,
                     estimator +'class_weight': [None, 'balanced']}
        self._make_pipeline(LogisticRegression())
        self._fit_gs_and_print_gs_results()
        self._make_pipeline(LogisticRegression(**self.params))
        value_of_metrics_cv(self.X_train, self.y_train,
                            self.pipe, 'Logistic Regression')

    def random_forest(self):
        estimator = 'randomforestclassifier__'
        self.grid = {estimator +'n_estimators': [30, 40, 50, 60, 70],
                     estimator +'criterion': ['gini', 'entropy'],
                     estimator +'max_depth': [None, 5, 6, 7, 8, 9, 10, 11],
                     estimator +'class_weight': [None, 'balanced']}
        self._make_pipeline(RandomForestClassifier())
        self._fit_gs_and_print_gs_results()
        self._make_pipeline(RandomForestClassifier(**self.params))
        value_of_metrics_cv(self.X_train, self.y_train,
                            self.pipe, 'Random Forest')

    def ada_boost(self):
        estimator = 'adaboostclassifier__'
        self.grid = {estimator +'n_estimators': [30, 40, 50, 60, 70],
                     estimator +'learning_rate': [0.5, 0.7, 1.0, 1.3],
                     estimator +'algorithm': ['SAMME', 'SAMME.R']}
        self._make_pipeline(AdaBoostClassifier())
        self._fit_gs_and_print_gs_results()
        self._make_pipeline(AdaBoostClassifier(**self.params))
        value_of_metrics_cv(self.X_train, self.y_train,
                            self.pipe, 'Ada Boost')

    def gb_classifier(self):
        estimator = 'gradientboostingclassifier__'
        self.grid = {estimator +'n_estimators': [20, 30, 40, 50, 60, 70],
                     estimator +'criterion': ['friedman_mse', 'mse'],
                     estimator +'max_depth': [None, 1, 2, 3, 4, 5, 6],
                     estimator +'learning_rate': [0.15, 0.3, 0.5, 0.7]}
        self._make_pipeline(GradientBoostingClassifier())
        self._fit_gs_and_print_gs_results()
        self._make_pipeline(GradientBoostingClassifier(**self.params))
        value_of_metrics_cv(self.X_train, self.y_train,
                            self.pipe, 'Gradient Boosting Classifier')

    def light_gbm(self):
        estimator = 'lgbmclassifier__'
        self.grid = {estimator +'n_estimators': [100, 200],
                     estimator +'max_depth': [6, 10, 20, 30],
                     estimator +'learning_rate': [0.01, 0.05, 0.1],
                     estimator +'num_leaves': [100, 300, 500]}
        self._make_pipeline(LGBMClassifier())
        self._fit_gs_and_print_gs_results()
        self._make_pipeline(LGBMClassifier(**self.params))
        value_of_metrics_cv(self.X_train, self.y_train,
                            self.pipe, 'Light GBM')

    def xg_boost(self):
        estimator = 'xgbclassifier__'
        self.grid = {estimator +'n_estimators': [20, 30, 40, 50, 60],
                     estimator +'max_depth': [2, 3, 4, 5, 6, 7, 8],
                     estimator +'learning_rate': [0.05, 0.1, 0.15, 0.3],
                     estimator +'min_child_weight': [1, 2, 3],
                     estimator +'verbosity': [0]}
        self._make_pipeline(XGBClassifier())
        self._fit_gs_and_print_gs_results()
        self._make_pipeline(XGBClassifier(**self.params))
        value_of_metrics_cv(self.X_train, self.y_train,
                            self.pipe, 'XG Boost')

    def cat_boost(self):
        estimator = 'catboostclassifier__'
        self.grid = {estimator +'iterations': [200, 250, 300],
                     estimator +'depth': [1, 2, 3, 4, 5],
                     estimator +'learning_rate': [None, 0.1, 0.3, 0.5, 0.7],
                     estimator +'auto_class_weights': ['None', 'Balanced'],
                     estimator +'verbose': [False]}
        self._make_pipeline(CatBoostClassifier())
        self._fit_gs_and_print_gs_results()
        self._make_pipeline(CatBoostClassifier(**self.params))
        value_of_metrics_cv(self.X_train, self.y_train,
                            self.pipe, 'Cat Boost')


class BestMultiClassifier():

    def __init__(self, X_train, X_test, transformer=None):
        self.X_train = X_train
        self.y_train = y_train
        self.transformer = transformer
        self.estim = 'onevsrestclassifier__estimator__'

    def _make_pipeline(self, clf):
        if self.transformer is None:
            self.pipe = make_pipeline(clf)
        else:
            self.pipe = make_pipeline(self.transformer, clf)

    def _fit_gs_and_print_gs_results(self):
        gs = GridSearchCV(self.pipe, self.grid, cv=5, scoring='accuracy')
        gs.fit(self.X_train, self.y_train)
        self.params = gs.best_params_
        # remove the name of estimator from parameters names
        keys_list = [key.split('or__')[1] for key in list(self.params.keys())]
        self.params = dict(zip(keys_list, self.params.values()))
        print('Best score (accuracy): {}'.format(gs.best_score_))
        print('Best parameters: {}\n'.format(self.params))
        print('С лучшими параметрами и кросс-валидацией:')

    def random_forest(self):
        self.grid = {self.estim +'n_estimators': [30, 40, 50, 60, 70],
                     self.estim +'criterion': ['gini', 'entropy'],
                     self.estim +'max_depth': [None, 5, 6, 7, 8, 9, 10, 11],
                     self.estim +'class_weight': [None, 'balanced']}
        self._make_pipeline(OneVsRestClassifier(RandomForestClassifier()))
        self._fit_gs_and_print_gs_results()
        self._make_pipeline(OneVsRestClassifier(RandomForestClassifier(**self.params)))
        value_of_metrics_cv_multiclass(self.X_train, self.y_train,
                                       self.pipe, 'Random Forest')

    def gb_classifier(self):
        self.grid = {self.estim +'n_estimators': [20, 30, 40, 50, 60, 70],
                     self.estim +'criterion': ['friedman_mse', 'mse'],
                     self.estim +'max_depth': [None, 1, 2, 3, 4, 5, 6],
                     self.estim +'learning_rate': [0.15, 0.3, 0.5, 0.7]}
        self._make_pipeline(OneVsRestClassifier(GradientBoostingClassifier()))
        self._fit_gs_and_print_gs_results()
        self._make_pipeline(OneVsRestClassifier(GradientBoostingClassifier(**self.params)))
        value_of_metrics_cv_multiclass(self.X_train, self.y_train,
                                       self.pipe, 'Gradient Boosting Classifier')
 
    def light_gbm(self):
        self.grid = {self.estim +'n_estimators': [100, 200],
                     self.estim +'max_depth': [6, 10, 20, 30],
                     self.estim +'learning_rate': [0.01, 0.05, 0.1],
                     self.estim +'num_leaves': [100, 300, 500]}
        self._make_pipeline(OneVsRestClassifier(LGBMClassifier()))
        self._fit_gs_and_print_gs_results()
        self._make_pipeline(OneVsRestClassifier(LGBMClassifier(**self.params)))
        value_of_metrics_cv_multiclass(self.X_train, self.y_train,
                                       self.pipe, 'Light GBM')

    def xg_boost(self):
        self.grid = {self.estim +'n_estimators': [20, 30, 40, 50, 60],
                     self.estim +'max_depth': [2, 3, 4, 5, 6, 7, 8],
                     self.estim +'learning_rate': [0.05, 0.1, 0.15, 0.3],
                     self.estim +'min_child_weight': [1, 2, 3],
                     self.estim +'use_label_encoder': [False],
                     self.estim +'verbosity': [0]}
        self._make_pipeline(OneVsRestClassifier(XGBClassifier()))
        self._fit_gs_and_print_gs_results()
        self._make_pipeline(OneVsRestClassifier(XGBClassifier(**self.params)))
        value_of_metrics_cv_multiclass(self.X_train, self.y_train,
                                       self.pipe, 'XG Boost')


class BestRegression():
    pass
