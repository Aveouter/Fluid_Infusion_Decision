## ML methods
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import xgboost as xgb

multi_task_models = {
    "SVM + SVR": MultiTaskModel(
        classifier=SVC(probability=True, random_state=42),
        regressor=SVR()
    ),
    "Logistic Regression + Linear Regression": MultiTaskModel(
        classifier=LogisticRegression(max_iter=1000, random_state=42),
        regressor=LinearRegression()
    ),
    "Decision Tree": MultiTaskModel(
        classifier=DecisionTreeClassifier(random_state=42),
        regressor=DecisionTreeRegressor(random_state=42)
    ),
    "Random Forest": MultiTaskModel(
        classifier=RandomForestClassifier(n_estimators=100, random_state=42),
        regressor=RandomForestRegressor(n_estimators=100, random_state=42)
    ),
    "GBDT": MultiTaskModel(
        classifier=GradientBoostingClassifier(n_estimators=100, random_state=42),
        regressor=GradientBoostingRegressor(n_estimators=100, random_state=42)
    ),
    "XGBoost": MultiTaskModel(
        classifier=xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42),
        regressor=xgb.XGBRegressor(n_estimators=100, random_state=42)
    )
}

class MultiTaskModel:
    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X, y_class, y_reg):
        self.classifier.fit(X, y_class)
        self.regressor.fit(X, y_reg)

    def predict(self, X):
        class_pred = self.classifier.predict(X)
        reg_pred = self.regressor.predict(X)
        class_proba = self.classifier.predict_proba(X) if hasattr(self.classifier, "predict_proba") else None
        return {
            "class": class_pred,
            "class_proba": class_proba,
            "reg": reg_pred
        }

