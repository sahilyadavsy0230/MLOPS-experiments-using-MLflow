from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow
import mlflow.sklearn

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
rf = RandomForestClassifier(random_state=42)

# Param grid
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

mlflow.set_experiment("Hyperparameter_Tuning_With_MLflow")

with mlflow.start_run():

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # ✅ Log each parameter separately
    for param, value in best_params.items():
        mlflow.log_param(param, value)

    # ✅ Log metric
    mlflow.log_metric("cv_accuracy", best_score)

    # ✅ Log model properly
    mlflow.sklearn.log_model(
        grid_search.best_estimator_,
        artifact_path="random_forest_model"
    )

    train_df = X_train.copy()
    train_df["target"] = y_train

    train_dataset = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_dataset, context="training")

    test_df = X_test.copy()
    test_df["target"] = y_test

    test_dataset = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_dataset, context="testing")


    # Add tag
    mlflow.set_tag("author", "Sahil Yadav")

    print("Best Parameters:", best_params)
    print("Best CV Accuracy:", best_score)
