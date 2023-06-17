from mlflow.exceptions import MlflowException

from mlfoundry import amplitude


class MlFoundryException(MlflowException):
    def __init__(self, message):
        super().__init__(message=message)
        self.message = message
        amplitude.track(
            amplitude.Event.EXCEPTION,
            event_properties={"exception_message": message},
        )
