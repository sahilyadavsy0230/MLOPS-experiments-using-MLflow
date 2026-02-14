import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix

mlflow.set_tracking_uri("http://127.0.0.1:5000")

data=load_wine()
x=data.data
y=data.target

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2)

max_depth=3
n_estimators=5

mlflow.set_experiment('mlflow-exp1')

with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
    rf.fit(X_train,Y_train)

    y_pred=rf.predict(X_test)
    accuracy=accuracy_score(Y_test,y_pred)

    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_param("max_depth",max_depth)
    mlflow.log_param("n_estimators",n_estimators)

    print(accuracy)

    cm = confusion_matrix(Y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig("Confusion-matrix.png")

    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)