import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def create_models():
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    ab_model = AdaBoostClassifier(n_estimators=100, random_state=42)

    voting_model = VotingClassifier(
        estimators=[('rf', rf_model), ('gb', gb_model), ('ab', ab_model)],
        voting='soft'
    )

    models = {
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model,
        'AdaBoost': ab_model,
        'Voting Classifier': voting_model
    }

    return models

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred)
    }

def train_and_evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}
    progress_bar = st.progress(0)
    for i, (name, model) in enumerate(models.items()):
        st.write(f"Debug: Training {name} model")
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)
        progress_bar.progress((i + 1) / len(models))
    return results

def predict_survival_probability(model, input_processed):
    return model.predict_proba(input_processed)[0][1]
