import joblib
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Titanic Analytics",
    page_icon="🚢",
    layout="wide"
)

# LOAD DATA
df = pd.read_csv("Titanic.csv")

# FEATURE ENGINEERING
df['family_size'] = df['sibsp'] + df['parch'] + 1

# HEADER
st.title("🚢 Titanic Data Analytics Dashboard")
st.caption("Explore passenger survival patterns and demographics.")

# SIDEBAR FILTERS
st.sidebar.header("Dashboard Filters")

gender_options = st.sidebar.multiselect(
    "Select Gender",
    options=sorted(df['sex'].dropna().unique()),
    default=sorted(df['sex'].dropna().unique())
)

class_options = st.sidebar.multiselect(
    "Select Passenger Class",
    options=sorted(df['class'].dropna().unique()),
    default=sorted(df['class'].dropna().unique())
)

survival_options = st.sidebar.multiselect(
    "Survival Status",
    options=[0, 1],
    default=[0, 1],
    format_func=lambda x: "Survived" if x == 1 else "Did Not Survive"
)

# APPLY FILTERS
filtered_df = df[
    (df['sex'].isin(gender_options)) &
    (df['class'].isin(class_options)) &
    (df['survived'].isin(survival_options))
]

# KPI CALCULATIONS
total = len(filtered_df)
survived = int(filtered_df['survived'].sum())
not_survived = total - survived

rate = (survived / total * 100) if total else 0

# KPI CARDS
st.subheader("Passenger Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Passengers", total)
col2.metric("Survived", survived)
col3.metric("Did Not Survive", not_survived)
col4.metric("Survival Rate", f"{rate:.2f}%")

st.divider()

# SURVIVAL DISTRIBUTION
st.subheader("Survival Distribution")

if total > 0:
    survival_data = filtered_df['survived'].value_counts().reset_index()
    survival_data.columns = ['survived', 'count']
    survival_data['Status'] = survival_data['survived'].map({
        0: 'Did Not Survive',
        1: 'Survived'
    })

    fig1 = px.pie(
        survival_data,
        names='Status',
        values='count',
        hole=0.5,
        title="Passenger Survival"
    )

    st.plotly_chart(fig1, width="stretch")
else:
    st.warning("No passengers match the selected filters.")

# GENDER AND CLASS ANALYSIS
col1, col2 = st.columns(2)

with col1:
    st.subheader("Survival by Gender")

    gender_data = filtered_df.groupby(
        'sex'
    )['survived'].mean().reset_index()

    gender_data['survived'] *= 100

    fig2 = px.bar(
        gender_data,
        x='sex',
        y='survived',
        color='sex',
        title="Gender-wise Survival Rate",
        labels={
            'sex': 'Gender',
            'survived': 'Survival Rate (%)'
        }
    )

    st.plotly_chart(fig2, width="stretch")

with col2:
    st.subheader("Survival by Passenger Class")

    class_data = filtered_df.groupby(
        'class',
        observed=True
    )['survived'].mean().reset_index()

    class_data['survived'] *= 100

    fig3 = px.bar(
        class_data,
        x='class',
        y='survived',
        color='class',
        title="Class-wise Survival Rate",
        labels={
            'class': 'Passenger Class',
            'survived': 'Survival Rate (%)'
        }
    )

    st.plotly_chart(fig3, width="stretch")

# AGE DISTRIBUTION
st.subheader("Passenger Age Distribution")

fig4 = px.histogram(
    filtered_df,
    x='age',
    color='survived',
    nbins=30,
    barmode='overlay',
    opacity=0.7,
    title="Age Distribution by Survival",
    labels={
        'age': 'Age',
        'survived': 'Survival Status'
    }
)

st.plotly_chart(fig4, width="stretch")

# FAMILY SIZE ANALYSIS
st.subheader("Family Size vs Survival")

family_data = filtered_df.groupby(
    'family_size'
)['survived'].mean().reset_index()

family_data['survived'] *= 100

fig5 = px.bar(
    family_data,
    x='family_size',
    y='survived',
    title="Survival Rate by Family Size",
    labels={
        'family_size': 'Family Size',
        'survived': 'Survival Rate (%)'
    }
)

st.plotly_chart(fig5, width="stretch")

st.success("Titanic Analytics Dashboard loaded successfully!")


# ==================================
# MACHINE LEARNING PREDICTION SYSTEM
# ==================================

st.divider()

st.header("🤖 Passenger Survival Prediction")

st.write(
    "Enter passenger details to predict the probability "
    "of survival using Logistic Regression."
)

# Load trained model
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "Logistic_Regression.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Passenger input form
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        sex = st.selectbox(
            "Gender",
            ["female", "male"]
        )

        age = st.number_input(
            "Age",
            min_value=0.0,
            max_value=100.0,
            value=25.0
        )

        passenger_class = st.selectbox(
            "Passenger Class",
            ["First", "Second", "Third"]
        )

        fare = st.number_input(
            "Ticket Fare",
            min_value=0.0,
            value=30.0
        )

    with col2:
        sibsp = st.number_input(
            "Number of Siblings/Spouses",
            min_value=0,
            max_value=10,
            value=0
        )

        parch = st.number_input(
            "Number of Parents/Children",
            min_value=0,
            max_value=10,
            value=0
        )

        embarked = st.selectbox(
            "Boarding Port",
            ["S", "C", "Q"]
        )

    predict_button = st.form_submit_button(
        "Predict Survival"
    )

# Prediction
if predict_button:

    family_size = sibsp + parch + 1

    is_alone = int(family_size == 1)

    # Exact training feature names
    input_data = pd.DataFrame([{
        'sex': sex,
        'age': age,
        'sibsp': sibsp,
        'parch': parch,
        'fare': fare,
        'embarked': embarked,
        'class': passenger_class,
        'family_size': family_size,
        'is_alone': is_alone
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Estimated Survival Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:

        st.success(
            "Prediction: Passenger is likely to survive."
        )

    else:

        st.error(
            "Prediction: Passenger is unlikely to survive."
        )

    st.caption(
        "This is a statistical prediction from a trained "
        "model, not a guarantee of an actual outcome."
    )



# ==================================
# MODEL PERFORMANCE DASHBOARD
# ==================================

st.divider()

st.header("📊 Machine Learning Model Performance")

# Model comparison data
performance = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [81.56, 75.98, 79.89],
    "Precision": [81.03, 74.07, 85.11],
    "Recall": [68.12, 57.97, 57.97],
    "F1 Score": [0.7402, 0.6504, 0.6897]
})

# Display table
st.subheader("Model Comparison")
st.dataframe(
    performance,
    hide_index=True,
    width="stretch"
)

# Accuracy comparison
fig_accuracy = px.bar(
    performance,
    x="Model",
    y="Accuracy",
    color="Model",
    title="Model Accuracy Comparison",
    text="Accuracy"
)

fig_accuracy.update_traces(texttemplate="%{text:.2f}%")

st.plotly_chart(fig_accuracy, width="stretch")

# F1 score comparison
fig_f1 = px.bar(
    performance,
    x="Model",
    y="F1 Score",
    color="Model",
    title="F1 Score Comparison",
    text="F1 Score"
)

fig_f1.update_traces(texttemplate="%{text:.4f}")

st.plotly_chart(fig_f1, width="stretch")

# Best model
best_model = performance.loc[
    performance["F1 Score"].idxmax()
]

st.success(
 f"Best Model based on F1 Score: {best_model['Model']}"
)

