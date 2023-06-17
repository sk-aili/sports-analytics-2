from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, constr

# TODO @nikp1172 Add support for np types https://numpy.org/doc/stable/user/basics.types.html
EntityValue = Union[str, int, float]


class PredictionData(BaseModel):
    value: EntityValue
    probabilities: Dict[str, float] = Field(default_factory=dict)
    shap_values: Dict[str, float] = Field(default_factory=dict)

    class Config:
        smart_union = True


class BasePacket(BaseModel):
    class Config:
        smart_union = True


class Prediction(BaseModel):

    data_id: constr(min_length=1, max_length=64)
    features: Dict[str, EntityValue]
    prediction_data: PredictionData
    actual_value: Optional[EntityValue] = None

    raw_data: Optional[Dict] = Field(default_factory=dict)
    occurred_at: datetime = Field(
        default_factory=lambda: datetime.utcnow().replace(tzinfo=None)
    )

    class Config:
        smart_union = True


class PredictionPacket(BasePacket):
    model_version_id: constr(min_length=1, max_length=64)
    prediction: Prediction


class Actual(BaseModel):
    data_id: constr(min_length=1, max_length=64)
    value: EntityValue

    class Config:
        smart_union = True


class ActualPacket(BasePacket):
    model_version_id: constr(min_length=1, max_length=64)
    actual: Actual


class ModelVersion(BaseModel):
    class Config:
        use_enum_values = True
        smart_union = True

    id: str
    fqn: str
    model_id: str
    model_fqn: str
    version: int

    model_schema: Optional[Dict] = None
    custom_metrics: Optional[List[Dict]] = Field(default_factory=list)
    created_by: str
    created_at: datetime
    updated_at: datetime
    monitoring_enabled: bool


# Kept only the fields relevant for the user
class DatasetData(BaseModel):
    data_id: str
    features: Dict[str, EntityValue]
    actual: Dict[str, Optional[EntityValue]]
    raw_data: Dict
    created_at: Optional[datetime]
