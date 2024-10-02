import streamlit as st
import plotly.express as px
import pandas as pd

def plot_passenger_class_distribution(df):
    fig = px.pie(df, names='Pclass', title='Distribution of Passenger Classes')
    st.plotly_chart(fig)

def plot_correlation_heatmap(df_filtered, selected_features):
    if len(selected_features) > 1:
        correlation_matrix = df_filtered[selected_features].corr()
        fig_corr = px.imshow(correlation_matrix, title="Correlation Heatmap", labels=dict(color="Correlation"))
        st.plotly_chart(fig_corr)
    else:
        st.warning("Please select at least two features for correlation analysis.")

def plot_survival_rate_by_class(df_filtered):
    survival_rate = df_filtered.groupby('Pclass')['Survived'].mean()
    fig_survival = px.bar(survival_rate, x=survival_rate.index, y='Survived', title="Survival Rate by Passenger Class")
    st.plotly_chart(fig_survival)

def plot_age_distribution(df_filtered):
    fig_age = px.histogram(df_filtered, x='Age', nbins=20, title="Age Distribution")
    st.plotly_chart(fig_age)

def plot_feature_importance(feature_cols, feature_importances):
    feature_importance = pd.DataFrame({'feature': feature_cols, 'importance': feature_importances})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    fig_importance = px.bar(feature_importance, x='feature', y='importance', title="Feature Importance")
    st.plotly_chart(fig_importance)

def plot_model_performance_comparison(results_df):
    fig = px.bar(results_df, barmode='group', title="Model Performance Comparison")
    st.plotly_chart(fig)
