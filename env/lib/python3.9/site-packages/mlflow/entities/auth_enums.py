import enum

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST

ENTITY_ID_ALL = "*"


@enum.unique
class SubjectType(enum.Enum):
    user = "user"
    serviceaccount = "serviceaccount"

    @classmethod
    def _missing_(cls, value):
        raise MlflowException(f"Unknown Subject Type: {value}", error_code=BAD_REQUEST)


@enum.unique
class EntityType(enum.Enum):
    EXPERIMENT = "EXPERIMENT"

    @classmethod
    def _missing_(cls, value):
        raise MlflowException(f"Unknown Entity Type: {value}", error_code=BAD_REQUEST)

    def to_resource_type(self):
        if self.value == EntityType.EXPERIMENT.value:
            return "mlf-project"
        return self.value


# READ is any read operation
# UPDATE is updating an experiment -> any write operation to experiment go here. Like create/delete run, artifact, model-version etc.
# DELETE - delete an experiment
# MANAGE - manage permission control type of things. mark public/private etc
@enum.unique
class ExperimentAction(enum.IntEnum):
    READ = 0
    UPDATE = 1
    DELETE = 2
    MANAGE = 3

    @classmethod
    def _missing_(cls, value):
        raise MlflowException(f"Unknown Entity Type: {value}", error_code=BAD_REQUEST)

    # TODO (nikp1172) other microservices to send action in future
    @classmethod
    def from_artifact_or_model_role(cls, role):
        if role == "READ":
            return cls.READ
        if role == "WRITE":
            return cls.UPDATE
        if role == "ADMIN":
            return cls.UPDATE
        raise MlflowException(f"Unknown Role Type: {role}", error_code=BAD_REQUEST)


@enum.unique
class PrivacyType(enum.Enum):
    PUBLIC = "PUBLIC"
    PRIVATE = "PRIVATE"

    @classmethod
    def _missing_(cls, value):
        raise MlflowException(f"Unknown Privacy Type: {value}", error_code=BAD_REQUEST)
