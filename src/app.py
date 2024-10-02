import streamlit as st
import pandas as pd
from src.data_loader import load_data, preprocess_data
from src.visualizations import (plot_passenger_class_distribution, plot_correlation_heatmap,
                            plot_survival_rate_by_class, plot_age_distribution,
                            plot_feature_importance, plot_model_performance_comparison)
from src.models import create_models, train_and_evaluate_models, predict_survival_probability
from src.utils import filter_dataframe, prepare_input_data
import src.config as config

st.set_page_config(layout="wide")
st.title("Titanic Dataset Analysis")
st.write("Debug: Starting the app")

df = load_data()

if df is not None:
    st.header("Dataset Information")
    st.write(f"Shape: {df.shape}")
    st.write("Columns:")
    st.write(df.columns.tolist())
    st.write("Data Types:")
    st.write(df.dtypes)

    plot_passenger_class_distribution(df)

    st.header("Dataset Preview")
    num_rows = st.slider("Select number of rows to display", 5, 50, 10)
    st.write(df.head(num_rows))

    st.header("Data Filtering")
    selected_columns = st.multiselect("Select columns to display", df.columns.tolist(), default=config.DEFAULT_DISPLAY_COLUMNS)

    pclass_filter = st.multiselect("Filter by Passenger Class", df['Pclass'].unique())
    sex_filter = st.multiselect("Filter by Sex", df['Sex'].unique())
    age_range = st.slider("Filter by Age Range", float(df['Age'].min()), float(df['Age'].max()), (float(df['Age'].min()), float(df['Age'].max())))

    df_filtered = filter_dataframe(df, pclass_filter, sex_filter, age_range)

    st.write(df_filtered[selected_columns])

    st.header("Filtered Data Statistics")
    st.write(df_filtered[selected_columns].describe())

    st.header("Enhanced Correlation Analysis")
    numeric_columns = df_filtered.select_dtypes(include=['int64', 'float64']).columns
    selected_features = st.multiselect("Select features for correlation analysis", numeric_columns, default=numeric_columns[:5])
    plot_correlation_heatmap(df_filtered, selected_features)

    # New advanced analysis: Survival Rate by Age Groups
    st.header("Survival Rate by Age Groups")
    df_filtered['AgeGroup'] = pd.cut(df_filtered['Age'], bins=[0, 18, 30, 50, 100], labels=['0-18', '19-30', '31-50', '51+'])
    survival_rate_by_age = df_filtered.groupby('AgeGroup')['Survived'].mean().reset_index()
    st.bar_chart(survival_rate_by_age.set_index('AgeGroup'))

    plot_survival_rate_by_class(df_filtered)
    plot_age_distribution(df_filtered)

    st.header("Ensemble Methods Comparison")
    st.write("Debug: Entering Ensemble Methods Comparison section")

    X_train, X_test, y_train, y_test, le, le_sex, cat_imputer, num_imputer = preprocess_data(df)

    st.write("Debug: Data preprocessing completed")

    models = create_models()
    results = train_and_evaluate_models(models, X_train, X_test, y_train, y_test)

    st.subheader("Model Performance Comparison")
    results_df = pd.DataFrame(results).T
    st.write(results_df)

    plot_model_performance_comparison(results_df)

    st.write("Debug: Ensemble Methods Comparison section completed")

    plot_feature_importance(config.FEATURE_COLS, models['Random Forest'].feature_importances_)

    st.header("Survival Probability Calculator")
    st.write("Enter passenger details to calculate survival probability:")

    input_pclass = st.selectbox("Passenger Class", [1, 2, 3])
    input_sex = st.selectbox("Sex", ["male", "female"])
    input_age = st.slider("Age", 0, 100, 30)
    input_sibsp = st.slider("Number of Siblings/Spouses Aboard", 0, 8, 0)
    input_parch = st.slider("Number of Parents/Children Aboard", 0, 6, 0)
    input_fare = st.slider("Fare", 0.0, 512.0, 32.0)
    input_embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    if st.button("Calculate Survival Probability"):
        input_data = pd.DataFrame({
            'Pclass': [input_pclass],
            'Sex': [input_sex],
            'Age': [input_age],
            'SibSp': [input_sibsp],
            'Parch': [input_parch],
            'Fare': [input_fare],
            'Embarked': [input_embarked]
        })
        
        input_processed = prepare_input_data(input_data, le_sex, le, cat_imputer, num_imputer, config.CAT_COLS, config.NUM_COLS)
        
        survival_prob = predict_survival_probability(models['Voting Classifier'], input_processed)
        st.write(f"Survival Probability: {survival_prob:.2%}")

    st.sidebar.header("About")
    st.sidebar.info("This Streamlit app analyzes the Titanic dataset. It provides basic information about the dataset, visualizations, filtering options, correlation analysis, feature importance, and a survival probability calculator.")
else:
    st.error("Failed to load the Titanic dataset. Please check your internet connection and try again.")

st.write("Debug: App execution completed")
