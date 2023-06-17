import typing
from operator import xor

from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.protos.service_pb2 import LatestRunLog as ProtoLatestRunLog
from mlflow.protos.service_pb2 import RunLog as ProtoRunLog


class RunLog(_MLflowObject):
    """
    Run Log object.
    """

    def __init__(
        self,
        key,
        timestamp,
        step,
        log_type,
        artifact_path=None,
        value=None,
        artifact_signed_uri: typing.Optional[str] = None,
    ):
        if not xor(bool(artifact_path), bool(value)):
            raise MlflowException(
                "either artifact_path or value should be empty. "
                f"artifact_path={artifact_path}, value={value}",
                INVALID_PARAMETER_VALUE,
            )

        self._key = key
        self._value = value
        self._timestamp = timestamp
        self._step = step
        self._log_type = log_type
        self._artifact_path = artifact_path
        self._artifact_signed_uri = None
        self.set_artifact_signed_uri(artifact_signed_uri)

    def set_artifact_signed_uri(self, artifact_signed_uri: str):
        if artifact_signed_uri and not self.artifact_path:
            raise MlflowException("artifact_signed_uri cannot be set if artifact_path is not set")
        self._artifact_signed_uri = artifact_signed_uri

    @property
    def artifact_signed_uri(self) -> typing.Optional[str]:
        return self._artifact_signed_uri

    @property
    def key(self):
        """String key corresponding to the metric name."""
        return self._key

    @property
    def value(self):
        """Float value of the metric."""
        return self._value

    @property
    def timestamp(self):
        """Metric timestamp as an integer (milliseconds since the Unix epoch)."""
        return self._timestamp

    @property
    def step(self):
        """Integer metric step (x-coordinate)."""
        return self._step

    @property
    def log_type(self):
        """type of non scalar metric to log"""
        return self._log_type

    @property
    def artifact_path(self):
        """artifact_path of non scalar metric (can be none)"""
        return self._artifact_path

    def to_proto(self):
        run_log = ProtoRunLog()
        run_log.key = self.key
        run_log.timestamp = self.timestamp
        run_log.step = self.step
        run_log.log_type = self.log_type
        run_log.artifact_path = self.artifact_path or ""
        run_log.value = self.value or ""
        run_log.artifact_signed_uri = self.artifact_signed_uri or ""
        return run_log

    @classmethod
    def from_proto(cls, proto):
        return cls(
            key=proto.key,
            timestamp=proto.timestamp,
            step=proto.step,
            log_type=proto.log_type,
            artifact_path=proto.artifact_path or None,
            value=proto.value or None,
            artifact_signed_uri=proto.artifact_signed_uri or None,
        )


class LatestRunLog:
    def __init__(self, run_log: RunLog, steps: typing.List[int]):
        self.run_log = run_log
        self.steps = steps

    def to_proto(self):
        latest_run_log = ProtoLatestRunLog(run_log=self.run_log.to_proto())
        latest_run_log.steps.extend(self.steps)

        return latest_run_log
