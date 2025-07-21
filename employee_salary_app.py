import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc
)

# Streamlit setup
st.set_page_config(layout="centered")
st.title("Employee Salary Prediction")

# Load the pre-uploaded CSV directly
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('adult 3.csv')
    df.replace('?', np.nan, inplace=True)

    cat_cols = df.select_dtypes(include='object').columns
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    for col in cat_cols:
        df_imputed[col] = df_imputed[col].round().astype(int)

    return df_imputed, cat_cols

# Call the function
df, cat_cols = load_and_preprocess_data()
  # Ensure this is in the same directory

st.subheader("Raw Dataset")
st.dataframe(df.head())

# Preprocessing
df.replace('?', np.nan, inplace=True)
cat_cols = df.select_dtypes(include='object').columns
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

for col in cat_cols:
    df_imputed[col] = df_imputed[col].round().astype(int)

# Target selection (assumes 'income' is the label)
target_column = 'income'
feature_columns = [col for col in df_imputed.columns if col != target_column]

# Feature selection via multiselect
st.subheader("Select Features to Train On")
selected_features = st.multiselect(
    "Choose input features:",
    options=feature_columns,
    default=feature_columns  # default selects all
)

if selected_features:
    # Split features and target
    X = df_imputed[selected_features]
    y = df_imputed[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    selected_model_name = st.selectbox("Choose a Model", list(models.keys()))
    model = models[selected_model_name]


    @st.cache_resource
    def train_model(model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

# Train the selected model
    model = train_model(model, X_train, y_train)
    

    y_pred = model.predict(X_test)

    # ROC prediction probabilities
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except:
        y_proba = None

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    st.subheader("Accuracy")
    st.metric(f"{selected_model_name} Accuracy", f"{acc:.4f}")

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # ROC Curve
    if y_proba is not None:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.2f}")
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend(loc="lower right")
        st.pyplot(fig2)

else:
    st.warning("Please select at least one feature to train the model.")
