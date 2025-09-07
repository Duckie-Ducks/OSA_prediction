#sklearn models
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier, KDTree, BallTree
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import PassiveAggressiveRegressor, Ridge, RidgeCV, SGDRegressor
from xgboost.sklearn import XGBRegressor

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel


def get_model(model_name):
    model_names = [
        'MLPClassifier',
        'KNeighborsClassifier', 
        'NearestCentroid',
        'LogisticRegression',
        'SGDClassifier', 
        'Perceptron',
        'DecisionTreeClassifier',
        'GaussianNB', 
        'MultinomialNB',
        'BernoulliNB',
        'SVC', 
        'LinearSVC', 
        'NuSVC',
        'XGBClassifier', 
        'RandomForestClassifier',
        'AdaBoostClassifier',
        'GradientBoostingClassifier', 
        'BaggingClassifier',
        'ExtraTreesClassifier',
        'HistGradientBoostingClassifier', 
        'LinearDiscriminantAnalysis',
        'QuadraticDiscriminantAnalysis', 
        'GaussianProcessClassifier', 
        'RadiusNeighborsClassifier',
        'RidgeClassifierCV', 
        'RidgeClassifier', 
        'PassiveAggressiveClassifier', 
        'LGBMClassifier', 
        'CatBoostClassifier',
        
        'AdaBoostRegressor',
        'BaggingRegressor',
        'CatBoostRegressor',
        'DecisionTreeRegressor',
        'ExtraTreesRegressor',
        'GaussianProcessRegressor',
        'GradientBoostingRegressor',
        'HistGradientBoostingRegressor',
        'KNeighborsRegressor',
        'LGBMRegressor',
        'LinearSVR',
        'MLPRegressor',
        'NuSVR',
        'PassiveAggressiveRegressor',
        'RadiusNeighborsRegressor',
        'RandomForestRegressor',
        'Ridge',
        'RidgeCV',
        'SGDRegressor',
        'SVR',
        'XGBRegressor'
        ]

    models = [MLPClassifier(max_iter=100, random_state=0), KNeighborsClassifier(), NearestCentroid(), LogisticRegression(random_state=0, max_iter=100),
            SGDClassifier(max_iter=100, random_state=0), Perceptron(max_iter=100, random_state=0), DecisionTreeClassifier(random_state=0),
            GaussianNB(), MultinomialNB(), BernoulliNB(),
            SVC(max_iter=100, random_state=0), LinearSVC(max_iter=100, random_state=0), NuSVC(max_iter=100, random_state=0, nu=0.4),
            XGBClassifier(objective="binary:logistic",random_state=0), RandomForestClassifier(random_state=0), AdaBoostClassifier(random_state=0),
            GradientBoostingClassifier(random_state=0), BaggingClassifier(random_state=0), ExtraTreesClassifier(random_state=0),
            HistGradientBoostingClassifier(random_state=0), LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis(),
            GaussianProcessClassifier(random_state=0), RadiusNeighborsClassifier(), RidgeClassifierCV(),
            RidgeClassifier(max_iter=100), PassiveAggressiveClassifier(random_state=0, max_iter=100), LGBMClassifier(random_state=0), CatBoostClassifier(random_state=0),
            
            AdaBoostRegressor(random_state=0), BaggingRegressor(random_state=0), CatBoostRegressor(random_state=0), DecisionTreeRegressor(random_state=0),
            ExtraTreesRegressor(random_state=0), GaussianProcessRegressor(random_state=0), GradientBoostingRegressor(random_state=0),
            HistGradientBoostingRegressor(random_state=0), KNeighborsRegressor(), LGBMRegressor(random_state=0), LinearSVR(max_iter=100), MLPRegressor(random_state=0),
            NuSVR(max_iter=100), PassiveAggressiveRegressor(max_iter=100, random_state=0), RadiusNeighborsRegressor(),
            RandomForestRegressor(random_state=0), Ridge(max_iter=100), RidgeCV(), SGDRegressor(max_iter=100, random_state=0),
            SVR(max_iter=100), XGBRegressor(random_state=0)
            ]
    
    model_dict = {key: value for (key, value) in zip(model_names, models)}

    return model_dict[model_name]
