import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    SMOTE = None
    ImbPipeline = None

# Optional XGBoost
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


# ----------------------------
# Utility functions
# ----------------------------
def preprocess_data(df: pd.DataFrame):
    """Apply same preprocessing steps as notebook"""
    df = df.copy()

    # Feature engineering
    if "end_treatment_date" in df.columns and "diagnosis_date" in df.columns:
        df["survival_days"] = (
            pd.to_datetime(df["end_treatment_date"]) - pd.to_datetime(df["diagnosis_date"])
        ).dt.days

    if "family_history" in df.columns:
        df["family_history"] = df["family_history"].map({"Yes": 1, "No": 0})

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    if "cancer_stage" in df.columns:
        mapping = {"I": 1, "II": 2, "III": 3, "IV": 4}
        df["cancer_stage"] = df["cancer_stage"].map(mapping)

    if "country" in df.columns:
        df.drop(columns=["country"], inplace=True)

    return df


def build_pipeline(model_name="rf", use_smote=False):
    """Create ML pipeline with preprocessing + classifier"""
    numeric_features = ["age", "survival_days"]
    categorical_features = ["gender", "family_history", "cancer_stage"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    if model_name == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
    elif model_name == "lr":
        clf = LogisticRegression(max_iter=200, random_state=42)
    elif model_name == "svm":
        clf = SVC(probability=True, random_state=42)
    elif model_name == "knn":
        clf = KNeighborsClassifier()
    elif model_name == "xgb" and XGBClassifier is not None:
        clf = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False
        )
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=42)

    if use_smote and SMOTE is not None and ImbPipeline is not None:
        pipeline = ImbPipeline(steps=[("preprocessor", preprocessor), ("smote", SMOTE()), ("classifier", clf)])
    else:
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

    return pipeline


# ----------------------------
# Streamlit App
# ----------------------------
st.title("🩺 Cancer Survival Prediction App")

st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Data")
    st.write(df.head())

    # Preprocess
    df = preprocess_data(df)

    # Select target column
    target_col = st.sidebar.text_input("Target column (default = 'survived')", value="survived")
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in dataset!")
    else:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Choose model
        model_choice = st.sidebar.selectbox("Choose Model", ["rf", "lr", "svm", "knn", "xgb" if XGBClassifier else "rf"])
        use_smote = st.sidebar.checkbox("Use SMOTE", value=False)
        test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2)

        if st.sidebar.button("🚀 Train Model"):
            pipeline = build_pipeline(model_choice, use_smote)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.success(f"Model trained! Accuracy = {acc:.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
            st.plotly_chart(fig)

            # Save model
            joblib.dump(pipeline, "trained_pipeline.joblib")
            st.sidebar.download_button(
                "⬇️ Download Trained Model",
                data=open("trained_pipeline.joblib", "rb").read(),
                file_name="trained_pipeline.joblib"
            )

            st.session_state["pipeline"] = pipeline

# -------------------------
# Prediction Section
# -------------------------
st.sidebar.header("2. Load Saved Model")
model_file = st.sidebar.file_uploader("Upload a trained pipeline (.joblib)", type=["joblib"])
if model_file:
    pipeline = joblib.load(model_file)
    st.session_state["pipeline"] = pipeline
    st.success("Model loaded successfully!")

# Manual Prediction Form
st.subheader("🔮 Make a Single Prediction")

if "pipeline" in st.session_state:
    pipeline = st.session_state["pipeline"]

    with st.form("prediction_form"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        stage = st.selectbox("Cancer Stage", ["I", "II", "III", "IV"])
        family = st.selectbox("Family History", ["Yes", "No"])
        survival_days = st.number_input("Survival Days (days)", min_value=0, max_value=5000, value=365)
        age = st.number_input("Age", min_value=0, max_value=120, value=50)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Format input
        input_df = pd.DataFrame([{
            "gender": 1 if gender == "Male" else 0,
            "cancer_stage": {"I": 1, "II": 2, "III": 3, "IV": 4}[stage],
            "family_history": 1 if family == "Yes" else 0,
            "survival_days": survival_days,
            "age": age
        }])

        pred = pipeline.predict(input_df)[0]
        st.success(f"✅ Predicted Target = {pred}")

        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(input_df)[0][1]
            st.info(f"Probability of survival = {proba:.2f}")

else:
    st.warning("⚠️ Please train or load a model first before making predictions.")
