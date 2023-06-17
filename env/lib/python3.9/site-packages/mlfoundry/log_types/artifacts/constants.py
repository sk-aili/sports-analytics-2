"""
.
├── .truefoundry/
│   └── metadata.json
└── files/
    └── model/
"""
import posixpath

INTERNAL_METADATA_DIR = ".truefoundry"
INTERNAL_METADATA_FILE_NAME = "metadata.json"
INTERNAL_METADATA_PATH = posixpath.join(
    INTERNAL_METADATA_DIR, INTERNAL_METADATA_FILE_NAME
)
FILES_DIR = "files"
MODEL_DIR_NAME = "model"
DESCRIPTION_MAX_LENGTH = 1024

# Link to docs here explaining schema consistency across model versions
MODEL_SCHEMA_UPDATE_FAILURE_HELP = """Model was logged successfully but failed to update the model schema because
it is inconsistent with the previous versions of the model. You can still fix the schema and update it using the
following:

```
import mlfoundry
from mlfoundry import ModelVersion, ModelSchema, Feature, FeatureValueType, PredictionValueType
client = mlfoundry.get_client()
model_version = ModelVersion(fqn="{fqn}")
model_version = ModelSchema(...) # or schema in dictionary format {{"features": [...], "prediction": ...}}
model_version.update()
```
"""
