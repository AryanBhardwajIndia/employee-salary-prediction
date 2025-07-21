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

# Streamlit page setup
st.set_page_config(layout="centered")
st.title("Employee Salary Prediction")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("adult 3.csv")
    df.replace('?', np.nan, inplace=True)
    return df

df = load_data()

st.subheader("Raw Dataset")
st.dataframe(df.head())

# Preprocessing
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

# Feature selection
st.subheader("Select Features to Train the Model")
all_features = [col for col in df_imputed.columns if col != "income"]
selected_features = st.multiselect("Select features", all_features, default=all_features)

if len(selected_features) == 0:
    st.warning("Please select at least one feature.")
    st.stop()

X = df_imputed[selected_features]
y = df_imputed["income"]

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

# Train and Predict
model = train_model(model, X_train, y_train)
y_pred = model.predict(X_test)

try:
    y_proba = model.predict_proba(X_test)[:, 1]
except:
    y_proba = None

# Results
st.subheader("Accuracy")
acc = accuracy_score(y_test, y_pred)
st.metric(f"{selected_model_name} Accuracy", f"{acc:.4f}")

st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

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
