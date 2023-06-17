from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel

from mlfoundry.monitoring.entities import DatasetData, EntityValue


class ClassPrediction(BaseModel):
    label: str
    score: float


class MlModelPrediction(BaseModel):
    value: EntityValue
    probabilities: Optional[List[ClassPrediction]]
    shap_values: Optional[Dict[str, float]]
    occurred_at: datetime

    class Config:
        smart_union = True


class Data(BaseModel):
    data_id: str
    features: Dict[str, EntityValue]
    prediction: MlModelPrediction
    actual: Optional[EntityValue]
    raw_data: Dict

    class Config:
        smart_union = True


class ActualData(BaseModel):
    data_id: str
    actual: EntityValue

    class Config:
        smart_union = True


class BatchInsertRequest(BaseModel):
    model_version_id: str
    tag: Optional[str]
    items: List[Data]

    def to_json_dict(self):
        request_dict = self.dict()
        for item in request_dict["items"]:
            item["prediction"]["occurred_at"] = item["prediction"][
                "occurred_at"
            ].isoformat()
        return request_dict


class BatchUpdateActualRequest(BaseModel):
    model_version_id: str
    items: List[ActualData]


class GetDatasetResponse(BaseModel):
    total_rows: int
    data: List[DatasetData]
