import mlflow.sklearn

from train import load_data

X, y = load_data()
sample = X.head(3)

mlflow.set_experiment("heart-disease-risk")
runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=2)

for run_id in runs["run_id"]:
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    print(model.predict_proba(sample)[:, 1].round(3))