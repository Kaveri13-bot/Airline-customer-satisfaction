import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE


# ---------------------------------------------------
# 1️⃣ Split Features & Target
# ---------------------------------------------------

def split_features_target(df):
    """
    Split dataframe into features (X) and target (y).
    Drops rows where target is missing.
    """
    df = df.dropna(subset=[TARGET_COLUMN])

    df["Flight Distance"] = pd.to_numeric(
    df["Flight Distance"],
    errors="coerce"
)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


# ---------------------------------------------------
# 2️⃣ Train-Validation Split (Stratified)
# ---------------------------------------------------

def split_train_validation(X, y):
    """
    Split dataset into training and validation sets.
    Uses stratification for balanced classes.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Very important for classification
    )

    return X_train, X_val, y_train, y_val


# ---------------------------------------------------
# 3️⃣ Build Preprocessing Pipeline
# ---------------------------------------------------

def build_preprocessor(X):
    """
    Create preprocessing pipeline for numerical and categorical columns.
    """

    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "category"]).columns

    # Numerical Pipeline
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical Pipeline
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine both
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


# ---------------------------------------------------
# 4️⃣ Prepare Test Features (Safe for Deployment)
# ---------------------------------------------------

def prepare_test_features(df_test):
    """
    Remove target column from test data if present.
    Safe for real-world deployment.
    """
    df = df_test.copy()

    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    return df

if __name__ == "__main__":

    print("Starting preprocessing...")

    df = pd.read_csv(r"C:\data\Airline_info.csv")

    X, y = split_features_target(df)

    X_train, X_val, y_train, y_val = split_train_validation(X, y)

    preprocessor = build_preprocessor(X_train)

    print("X Train Shape:", X_train.shape)
    print("X Validation Shape:", X_val.shape)

    print("Preprocessor created successfully!")