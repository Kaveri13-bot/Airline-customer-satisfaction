import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from config import RANDOM_STATE


def build_model(preprocessor):

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            eval_metric="logloss"
        ))
    ])

    return model


def train_model(model, X_train, y_train):

    model.fit(X_train, y_train)

    joblib.dump(model, "model.pkl")

    return model