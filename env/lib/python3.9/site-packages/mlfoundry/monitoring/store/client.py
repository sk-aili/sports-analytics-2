import atexit
import json
import os
import queue
import threading
import typing
import uuid
from datetime import datetime

from mlflow.tracking import MlflowClient

from mlfoundry.exceptions import MlFoundryException
from mlfoundry.logger import logger
from mlfoundry.monitoring.entities import (
    Actual,
    ActualPacket,
    BasePacket,
    Prediction,
    PredictionPacket,
)
from mlfoundry.monitoring.store.constants import (
    FLUSH_BATCH_SIZE_ENV_VAR,
    FLUSH_INTERVAL_ENV_VAR,
    MAX_QUEUE_SIZE_ENV_VAR,
)
from mlfoundry.monitoring.store.repositories import get_monitoring_store
from mlfoundry.monitoring.store.worker import MonitoringStoreWorker
from mlfoundry.session import Session


def _check_shutdown(func):
    def wrapper(self, *args, **kwargs):
        if self._is_shutdown:
            raise MlFoundryException("cannot execute after client is shutdown")
        return func(self, *args, **kwargs)

    return wrapper


class MonitoringClient:
    def __init__(self, session: Session):
        self._is_shutdown = False

        self.monitoring_store = get_monitoring_store(session=session)
        max_queue_size = int(os.getenv(MAX_QUEUE_SIZE_ENV_VAR, "10000"))
        flush_interval = float(os.getenv(FLUSH_INTERVAL_ENV_VAR, "5"))
        flush_batch_size = int(os.getenv(FLUSH_BATCH_SIZE_ENV_VAR, "300"))

        if max_queue_size <= 0:
            raise MlFoundryException(
                f"{MAX_QUEUE_SIZE_ENV_VAR} should be a positive number"
            )

        self.task_queue: "queue.Queue[BasePacket]" = queue.Queue(max_queue_size)
        self.worker_terminate_event = threading.Event()
        self.mlflow_client = MlflowClient()

        if flush_interval <= 0:
            raise MlFoundryException(
                f"{FLUSH_INTERVAL_ENV_VAR} should be a positive number"
            )
        if flush_batch_size <= 0 or flush_batch_size > max_queue_size:
            raise MlFoundryException(
                f"{FLUSH_BATCH_SIZE_ENV_VAR} should be positive"
                f" and lower than {MAX_QUEUE_SIZE_ENV_VAR}"
            )

        self.worker: MonitoringStoreWorker = MonitoringStoreWorker(
            task_queue=self.task_queue,
            terminate_event=self.worker_terminate_event,
            monitoring_store=self.monitoring_store,
            flush_interval=flush_interval,
            flush_every_num_message=flush_batch_size,
        )
        atexit.register(self._shutdown)
        self.worker.start()

    def get_inference_dataset(
        self,
        model_fqn: str,
        start_time: datetime,
        end_time: datetime,
        actual_value_required: bool,
    ):
        model = self.mlflow_client.get_model_by_fqn(model_fqn)

        return self.monitoring_store.get_inference_dataset(
            model_id=str(model.id),
            start_time=start_time,
            end_time=end_time,
            actual_value_required=actual_value_required,
        )

    @staticmethod
    def generate_hash_from_data(
        features: typing.Dict, timestamp: typing.Optional[datetime] = None
    ):
        data_str = json.dumps(features, sort_keys=True)
        if timestamp:
            data_str += timestamp.isoformat()
        return uuid.uuid5(uuid.NAMESPACE_X500, data_str).hex

    @staticmethod
    def generate_random_id():
        return uuid.uuid4().hex

    def _generate_prediction_packet_from_data(
        self, model_version_fqn: str, prediction: Prediction
    ) -> PredictionPacket:

        return PredictionPacket(
            model_version_id=self.monitoring_store.enable_monitoring_for_version(
                model_version_fqn
            ).id,
            prediction=prediction,
        )

    def _generate_actual_packet_from_data(self, model_version_fqn: str, actual: Actual):
        return ActualPacket(
            model_version_id=self.monitoring_store.enable_monitoring_for_version(
                model_version_fqn
            ).id,
            actual=actual,
        )

    def _put_in_queue(self, inference_packet: BasePacket):
        try:
            self.task_queue.put_nowait(inference_packet)
        except queue.Full as ex:
            raise MlFoundryException(
                "task queue is full\n"
                f"current task queue length is {self.task_queue.maxsize}\n"
                "consider increasing the task queue length using "
                f"{MAX_QUEUE_SIZE_ENV_VAR} environment variable"
            ) from ex

    @_check_shutdown
    def log_prediction(self, model_version_fqn: str, prediction: Prediction):
        prediction_packet = self._generate_prediction_packet_from_data(
            model_version_fqn=model_version_fqn, prediction=prediction
        )
        self._put_in_queue(prediction_packet)

    @_check_shutdown
    def log_actual(self, model_version_fqn: str, actual: Actual):
        actual_packet = self._generate_actual_packet_from_data(
            model_version_fqn=model_version_fqn, actual=actual
        )
        self._put_in_queue(actual_packet)

    @_check_shutdown
    def log_predictions(
        self, model_version_fqn: str, predictions: typing.List[Prediction]
    ):
        for prediction in predictions:
            self.log_prediction(
                model_version_fqn=model_version_fqn, prediction=prediction
            )

    @_check_shutdown
    def log_actuals(self, model_version_fqn: str, actuals: typing.List[Actual]):
        for actual in actuals:
            self.log_actual(model_version_fqn=model_version_fqn, actual=actual)

    def _flush(self):
        logger.debug(
            f"flushing task queue, {self.task_queue.qsize()} items in the queue"
        )
        self.task_queue.join()
        logger.debug("task queue flushed")

    def _shutdown(self):
        if self._is_shutdown:
            return
        logger.debug("shutting down worker and client")
        self._is_shutdown = True
        # NOTE: We initialize the monitoring store at first in the constructor
        # The task_queue, worker is defined later.
        # There is a chance that monitoring  store initialization will throw error,
        # in that case, shutdown will be called (__del__) but self.task_queue would not have
        # been initialized yet.
        if hasattr(self, "task_queue"):
            self._flush()
        if hasattr(self, "worker_terminate_event"):
            logger.debug("setting worker termination event")
            self.worker_terminate_event.set()
        if hasattr(self, "worker"):
            logger.debug("waiting for worker to terminate")
            self.worker.join()

    def __del__(self):
        self._shutdown()
