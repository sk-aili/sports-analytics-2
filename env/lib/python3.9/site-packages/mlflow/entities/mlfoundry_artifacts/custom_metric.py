from pydantic import BaseModel

from mlflow.entities.mlfoundry_artifacts import utils
from mlflow.entities.mlfoundry_artifacts.enums import (
    CustomMetricType,
    CustomMetricValueType,
)


class CustomMetric(BaseModel):
    class Config:
        orm_mode = True
        validate_assignment = True
        use_enum_values = True
        allow_mutation = False
        smart_union = True
        extra = "allow"

    name: str
    value_type: CustomMetricValueType
    type: CustomMetricType

    @classmethod
    def from_proto(cls, message):
        return cls.parse_obj(utils.dict_from_proto(message))

    def to_proto(self):
        return utils.dict_to_proto(self.dict())
