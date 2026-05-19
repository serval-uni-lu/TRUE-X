import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score

from ml_models.model_registry import ModelRegistry


class Metrics:
    def __init__(self):
        pass

    def get_task_type(self, model):
        """Returns 'Regression' or 'Classification' for the given Model enum."""
        if model in ModelRegistry._registry:
            return ModelRegistry._registry[model].get("task", "Unknown")
        return "Model not found"

    def calculate_metrics(self, model, y_true, y_pred):
        """
        Compute task-appropriate evaluation metrics.

        For regression, returns RMSE, MSE, MAE, and the NASA asymmetric scoring function
        from the PHM challenge, which penalises late predictions more than early ones.
        For classification, returns accuracy, macro precision, and macro recall.
        """
        task_type = self.get_task_type(model)

        if task_type == "Regression":
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            mae  = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mse  = mean_squared_error(y_true, y_pred)

            # Asymmetric score: under-predictions (diff <= 0) are penalised less
            # than over-predictions, reflecting that in RUL estimation it is more
            # dangerous to predict too much remaining life than too little.
            score = 0.0
            for diff in (y_pred - y_true):
                score += np.expm1(-diff / 13) if diff <= 0 else np.expm1(diff / 10)

            return {"rmse": rmse, "mse": mse, "mae": mae, "score": score}

        if task_type == "Classification":
            # Convert probability outputs or one-hot arrays to class indices
            if isinstance(y_true, np.ndarray) and y_true.ndim > 1:
                y_true = np.argmax(y_true, axis=1)
            if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
                y_pred = np.argmax(y_pred, axis=1)
            return {
                "accuracy":  accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="macro"),
                "recall":    recall_score(y_true, y_pred, average="macro"),
            }

        return {"error": f"Unknown task type: {task_type}"}

    def print_metrics(self, metrics):
        for key, value in metrics.items():
            print(f"{key.capitalize()}: {value}")
