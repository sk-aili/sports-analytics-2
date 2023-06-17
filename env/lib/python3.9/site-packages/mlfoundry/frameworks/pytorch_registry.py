import mlflow

from mlfoundry.frameworks.base_registry import BaseRegistry


class PyTorchRegistry(BaseRegistry):
    def log_model(self, model, artifact_path: str, **kwargs):
        """
        Log PyTorch model
        Args:
            model : the model to be registered
            artifact_path (str): artifact_path
        """
        mlflow.pytorch.log_model(model, artifact_path=artifact_path, **kwargs)

    def save_model(self, model, path: str, **kwargs):
        return mlflow.pytorch.save_model(model, path, **kwargs)

    def load_model(self, model_file_path: str, dest_path: str = None, **kwargs):
        """
        Load PyTorch model
        """
        return mlflow.pytorch.load_model(model_file_path, dest_path, **kwargs)
