import datetime
import uuid

from pydantic import BaseModel

from mlflow.entities.mlfoundry_artifacts.artifact import Artifact
from mlflow.entities.mlfoundry_artifacts.enums import ArtifactVersionTransitStatus


class ArtifactVersionInTransit(BaseModel):
    class Config:
        orm_mode = True
        validate_assignment = True
        use_enum_values = True
        allow_mutation = False
        smart_union = True

    version_id: uuid.UUID
    artifact_id: uuid.UUID
    artifact_storage_root: str
    status: ArtifactVersionTransitStatus
    created_at: datetime.datetime
    updated_at: datetime.datetime

    artifact: Artifact
