import typing
from collections import defaultdict
from datetime import datetime
from functools import lru_cache

from mlflow.utils.rest_utils import MlflowHostCreds, http_request_safe

from mlfoundry.logger import logger
from mlfoundry.monitoring.entities import (
    ActualPacket,
    BasePacket,
    DatasetData,
    ModelVersion,
    PredictionPacket,
)
from mlfoundry.monitoring.store.repositories.dto import (
    ActualData,
    BatchInsertRequest,
    BatchUpdateActualRequest,
    ClassPrediction,
    Data,
    GetDatasetResponse,
    MlModelPrediction,
)


# TODO @nikp1172 add feature of autmatically serializing datetime object in \
#  mlflow.utils.rest_utils.http_request
class RestMonitoringStore:
    def __init__(self, get_host_creds: typing.Callable[[], MlflowHostCreds]):
        self._get_host_creds: typing.Callable[[], MlflowHostCreds] = get_host_creds

    def _insert_prediction(self, prediction_packets: typing.List[PredictionPacket]):
        if not prediction_packets:
            return
        # group packets by model version id
        grouped_packets = defaultdict(list)

        for packet in prediction_packets:
            grouped_packets[packet.model_version_id].append(packet)

        for model_version_id in grouped_packets.keys():
            items = [
                Data(
                    data_id=packet.prediction.data_id,
                    features=packet.prediction.features,
                    prediction=MlModelPrediction(
                        value=packet.prediction.prediction_data.value,
                        probabilities=[
                            ClassPrediction(label=feature_name, score=score)
                            for feature_name, score in packet.prediction.prediction_data.probabilities.items()
                        ],
                        shap_values=packet.prediction.prediction_data.shap_values,
                        occurred_at=packet.prediction.occurred_at,
                    ),
                    actual=packet.prediction.actual_value,
                    raw_data=packet.prediction.raw_data,
                )
                for packet in grouped_packets[model_version_id]
            ]

            batch_insert_request = BatchInsertRequest(
                model_version_id=model_version_id,
                items=items,
            )

            response = http_request_safe(
                host_creds=self._get_host_creds(),
                endpoint="/v1/data/batch-insert",
                method="post",
                json=batch_insert_request.to_json_dict(),
            )

    def _update_actuals(self, actual_packets: typing.List[ActualPacket]):
        if not actual_packets:
            return

        grouped_packets = defaultdict(list)

        for packet in actual_packets:
            grouped_packets[packet.model_version_id].append(packet)

        for model_version_id in grouped_packets.keys():
            batch_update_request = BatchUpdateActualRequest(
                model_version_id=model_version_id,
                items=[
                    ActualData(
                        data_id=packet.actual.data_id, actual=packet.actual.value
                    )
                    for packet in grouped_packets[model_version_id]
                ],
            )
            response = http_request_safe(
                host_creds=self._get_host_creds(),
                endpoint="/v1/data/batch-update-actual",
                method="post",
                json=batch_update_request.dict(),
            )

    def batch_log_inference(self, inference_packets: typing.List[BasePacket]):
        grouped_packets = defaultdict(list)
        for packet in inference_packets:
            grouped_packets[type(packet)].append(packet)

        # log predictions before actuals to avoid actuals getting logged before predictions
        if PredictionPacket in grouped_packets.keys():
            self._insert_prediction(grouped_packets[PredictionPacket])
        if ActualPacket in grouped_packets.keys():
            self._update_actuals(grouped_packets[ActualPacket])

    @lru_cache(maxsize=32)
    def enable_monitoring_for_version(self, model_version_fqn: str):
        response = http_request_safe(
            host_creds=self._get_host_creds(),
            endpoint=f"/v1/model/enable",
            method="post",
            json={"model_version_fqn": model_version_fqn},
        )
        response_json = response.json()
        return ModelVersion(**response_json)

    def _get_inference_data(
        self,
        model_id: str,
        start_time: datetime,
        end_time: datetime,
        actual_value_required: bool,
        offset: int,
        limit: int,
    ) -> GetDatasetResponse:
        response = http_request_safe(
            host_creds=self._get_host_creds(),
            endpoint="/v1/data/get-dataset",
            method="post",
            json={
                "model_id": model_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "actual_value_required": actual_value_required,
                "offset": offset,
                "limit": limit,
            },
        )
        response_json = response.json()
        return GetDatasetResponse(**response_json)

    def get_inference_dataset(
        self,
        model_id: str,
        start_time: datetime,
        end_time: datetime,
        actual_value_required: bool,
    ) -> typing.List[DatasetData]:

        total_results = None
        data = []

        filters = []
        if actual_value_required:
            filters.append({"type": "neq", "dimension": "actual.value", "value": None})

        while total_results is None or len(data) < total_results:
            list_dataset_data_object = self._get_inference_data(
                model_id=model_id,
                start_time=start_time,
                end_time=end_time,
                actual_value_required=actual_value_required,
                offset=len(data),
                limit=5000,
            )
            if not total_results:
                total_results = list_dataset_data_object.total_rows
                logger.info(f"Total Results: {total_results} for model_id: {model_id}")
            data.extend(list_dataset_data_object.data)
            if not len(list_dataset_data_object.data):
                break
            logger.info(
                f"Fetched {len(list_dataset_data_object.data)} rows for {model_id}."
            )
        return data
