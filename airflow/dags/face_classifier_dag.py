"""Weekly retrain DAG for the face classifier."""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def _check_min_classes(**_context) -> str:
    """Confirm every class has at least the min number of validated images."""
    from src.config import CLASS_NAMES, VALIDATED_DIR, get_param
    from src.utils.helpers import list_images

    min_count = get_param("data_collection", "min_per_class", 80)
    deficient = [
        c for c in CLASS_NAMES
        if len(list_images(VALIDATED_DIR / c)) < min_count
    ]
    if deficient:
        raise AssertionError(
            f"{len(deficient)} classes below min={min_count}: {deficient[:5]}..."
        )
    return "ok"


def _register_best_model(**_context) -> str:
    """Promote the best model in MLflow registry."""
    import mlflow
    from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        return "no_experiment"
    runs = client.search_runs(
        [exp.experiment_id],
        order_by=["metrics.test_accuracy DESC"],
        max_results=1,
    )
    if not runs:
        return "no_runs"
    best = runs[0]
    try:
        client.create_registered_model("pak-faces-classifier")
    except Exception:
        pass
    mv = client.create_model_version(
        name="pak-faces-classifier",
        source=f"{best.info.artifact_uri}/model",
        run_id=best.info.run_id,
    )
    return f"registered_version_{mv.version}"


with DAG(
    dag_id="face_classifier_weekly_retrain",
    description="Weekly retrain of face classifier",
    schedule_interval="0 2 * * 0",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "pak-faces"],
) as dag:

    validate_data = BashOperator(
        task_id="validate_data",
        bash_command="cd /opt/pak-faces && python scripts/run_validation.py",
    )

    check_min_classes = PythonOperator(
        task_id="check_min_classes",
        python_callable=_check_min_classes,
    )

    retrain_resnet = BashOperator(
        task_id="retrain_resnet",
        bash_command=(
            "cd /opt/pak-faces && python scripts/run_training.py "
            "--models resnet50"
        ),
    )

    retrain_effnet = BashOperator(
        task_id="retrain_efficientnet",
        bash_command=(
            "cd /opt/pak-faces && python scripts/run_training.py "
            "--models efficientnet_b3"
        ),
    )

    evaluate_resnet = BashOperator(
        task_id="evaluate_resnet",
        bash_command=(
            "cd /opt/pak-faces && python scripts/run_evaluation.py "
            "--model resnet50"
        ),
    )

    evaluate_effnet = BashOperator(
        task_id="evaluate_efficientnet",
        bash_command=(
            "cd /opt/pak-faces && python scripts/run_evaluation.py "
            "--model efficientnet_b3"
        ),
    )

    register_model = PythonOperator(
        task_id="register_best_model",
        python_callable=_register_best_model,
    )

    validate_data >> check_min_classes
    check_min_classes >> [retrain_resnet, retrain_effnet]
    retrain_resnet >> evaluate_resnet
    retrain_effnet >> evaluate_effnet
    [evaluate_resnet, evaluate_effnet] >> register_model
