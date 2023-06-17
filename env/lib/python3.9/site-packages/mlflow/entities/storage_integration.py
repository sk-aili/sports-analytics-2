import re
from enum import Enum
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, root_validator

from mlflow.exceptions import MlflowException


class IntegrationType(str, Enum):
    STORAGE_INTEGRATION = "STORAGE_INTEGRATION"


class IntegrationProvider(str, Enum):
    AWS_S3 = "aws-s3"
    GCP_GCS = "gcp-gcs"
    AZURE_BLOB = "azure-blob"


class StorageIntegrationMetadata(BaseModel):
    storageRoot: str


class GCSCredentials(BaseModel):
    class Config:
        extra = "allow"

    keyFileContent: Dict[str, Any]


class AWSS3Credentials(BaseModel):
    awsAccessKeyId: str
    awsSecretAccessKey: str
    region: Optional[str] = None


class AzureBlobCredentials(BaseModel):
    connectionString: str


class StorageIntegration(BaseModel):
    class Config:
        extra = "allow"

    id: str
    name: str
    fqn: str
    tenantName: str
    type: IntegrationType
    integrationProvider: IntegrationProvider

    metaData: StorageIntegrationMetadata
    authData: Optional[Union[AWSS3Credentials, GCSCredentials, AzureBlobCredentials]] = None

    def get_storage_root(self) -> str:
        if self.integrationProvider == IntegrationProvider.AZURE_BLOB:
            storageRoot = (
                self.metaData.storageRoot
                if self.metaData.storageRoot.endswith("/")
                else self.metaData.storageRoot + "/"
            )
            match = re.match(
                r"https://(?P<storage_account>[^.]+)\.blob\.core\.windows\.net/(?P<container_name>[^/]+)/(?P<path>.*)",
                storageRoot,
            )
            if not match:
                raise MlflowException(
                    "Invalid Azure Blob Storage URI: {}, for storage integration: {}".format(
                        storageRoot, self.fqn
                    )
                )
            container_name = match.group("container_name")
            storage_account_name = match.group("storage_account")
            path = match.group("path") or ""
            return f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/{path}"
        return self.metaData.storageRoot

    @root_validator(pre=True)
    def check_empty_auth_data(cls, values):
        if not values.get("authData"):
            values["authData"] = None
        return values

    @root_validator()
    def check_auth_data(cls, values):
        if not values.get("authData"):
            return values
        if values["integrationProvider"] == IntegrationProvider.AWS_S3:
            if not isinstance(values["authData"], AWSS3Credentials):
                raise ValueError("authData must be of type AWSS3Credentials")
        elif values["integrationProvider"] == IntegrationProvider.GCP_GCS:
            if not isinstance(values["authData"], GCSCredentials):
                raise ValueError("authData must be of type GCSCredentials")
        elif values["integrationProvider"] == IntegrationProvider.AZURE_BLOB:
            if not isinstance(values["authData"], AzureBlobCredentials):
                raise ValueError("authData must be of type AzureBlobCredentials")
        return values
