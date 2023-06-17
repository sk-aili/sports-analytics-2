import enum

from mlflow.entities import ViewType

from mlfoundry.exceptions import MlFoundryException


class ViewType(ViewType):
    """
    Available Keys:
    ACTIVE_ONLY: 1 ,DELETED_ONLY: 2, ALL: 3
    """

    pass


class EnumMissingMixin:
    @classmethod
    def _missing_(cls, value):
        raise MlFoundryException(
            "%r is not a valid %s.  Valid types: %s"
            % (
                value,
                cls.__name__,
                ", ".join([repr(m.value) for m in cls]),
            )
        )


class FileFormat(EnumMissingMixin, enum.Enum):
    CSV = "csv"
    PARQUET = "parquet"


class ModelFramework(EnumMissingMixin, enum.Enum):
    SKLEARN = "sklearn"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    KERAS = "keras"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    FASTAI = "fastai"
    H2O = "h2o"
    ONNX = "onnx"
    SPACY = "spacy"
    STATSMODELS = "statsmodels"
    GLUON = "gluon"
    PADDLE = "paddle"
    TRANSFORMERS = "transformers"
    UNKNOWN = "unknown"


class DataSlice(EnumMissingMixin, enum.Enum):
    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"
    PREDICTION = "prediction"


class ModelType(EnumMissingMixin, enum.Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIMESERIES = "timeseries"
