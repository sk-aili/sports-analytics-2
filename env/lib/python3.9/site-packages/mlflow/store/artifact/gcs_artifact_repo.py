import json
import logging
import os
import posixpath
import tempfile
import urllib.parse

from mlflow.entities import FileInfo
from mlflow.entities.storage_integration import GCSCredentials
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import UNAUTHORIZED
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import relative_path_to_artifact_path

logger = logging.getLogger(__name__)

_REQUIRED_SCOPES = (
    "https://www.googleapis.com/auth/devstorage.full_control",
    "https://www.googleapis.com/auth/devstorage.read_only",
    "https://www.googleapis.com/auth/devstorage.read_write",
    "https://www.googleapis.com/auth/cloud-platform",
)


class GCSArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on Google Cloud Storage.

    Assumes the google credentials are available in the environment,
    see https://google-cloud.readthedocs.io/en/latest/core/auth.html.
    """

    def __init__(
        self,
        artifact_uri,
        client=None,
        credentials: GCSCredentials = None,
        storage_integration_id=None,
    ):
        from google import auth
        from google.auth.exceptions import DefaultCredentialsError

        if client:
            self.gcs = client
        else:
            from google.cloud import storage as gcs_storage

            self.gcs = gcs_storage
        try:
            if not credentials or not credentials.keyFileContent:
                credentials, project = auth.default(scopes=_REQUIRED_SCOPES)
            else:
                with tempfile.NamedTemporaryFile(mode="w") as tmp:
                    json.dump(credentials.keyFileContent, tmp.file)
                    tmp.file.close()
                    credentials, project = auth.load_credentials_from_file(
                        tmp.name, scopes=_REQUIRED_SCOPES
                    )
            self._storage_client = self.gcs.Client(credentials=credentials)
        except Exception as e:
            logger.exception("Failed to initialize GCS Client.")
            raise MlflowException(
                message=(
                    "Could not initialize GCS client. "
                    "Please make sure that you have configured the "
                    "permissions of storage integration correctly. "
                ),
                error_code=UNAUTHORIZED,
            )
        self._credentials = self._storage_client._credentials
        super().__init__(artifact_uri)

    @staticmethod
    def parse_gcs_uri(uri):
        """Parse an GCS URI, returning (bucket, path)"""
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "gs":
            raise Exception("Not a GCS URI: %s" % uri)
        path = parsed.path
        if path.startswith("/"):
            path = path[1:]
        return parsed.netloc, path

    def _get_bucket(self, bucket):
        return self._storage_client.bucket(bucket)

    def _ensure_valid_credentials(self):
        # A hacky solution from https://stackoverflow.com/a/64245028/3697191
        from google.auth.transport import requests

        if self._credentials.valid:
            return
        self._credentials.refresh(requests.Request())
        if not self._credentials.valid:
            raise Exception("Failed to fetch valid credentials to Google Cloud Storage")

    def log_artifact(self, local_file, artifact_path=None):
        self._ensure_valid_credentials()
        (bucket, dest_path) = self.parse_gcs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))

        gcs_bucket = self._get_bucket(bucket)
        blob = gcs_bucket.blob(dest_path)
        blob.upload_from_filename(local_file)

    def log_artifacts(self, local_dir, artifact_path=None):
        self._ensure_valid_credentials()
        (bucket, dest_path) = self.parse_gcs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        gcs_bucket = self._get_bucket(bucket)

        local_dir = os.path.abspath(local_dir)
        for (root, _, filenames) in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = os.path.relpath(root, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                upload_path = posixpath.join(dest_path, rel_path)
            for f in filenames:
                path = posixpath.join(upload_path, f)
                gcs_bucket.blob(path).upload_from_filename(os.path.join(root, f))

    def list_artifacts(self, path=None):
        self._ensure_valid_credentials()
        (bucket, artifact_path) = self.parse_gcs_uri(self.artifact_uri)
        dest_path = artifact_path
        if path:
            dest_path = posixpath.join(dest_path, path)
        prefix = dest_path if dest_path.endswith("/") else dest_path + "/"

        bkt = self._get_bucket(bucket)

        infos = self._list_folders(bkt, prefix, artifact_path)

        results = bkt.list_blobs(prefix=prefix, delimiter="/")
        for result in results:
            # skip blobs matching current directory path as list_blobs api
            # returns subdirectories as well
            if result.name == prefix:
                continue
            blob_path = posixpath.relpath(result.name, artifact_path)
            infos.append(FileInfo(blob_path, False, result.size))

        return sorted(infos, key=lambda f: f.path)

    def _list_folders(self, bkt, prefix, artifact_path):
        self._ensure_valid_credentials()
        results = bkt.list_blobs(prefix=prefix, delimiter="/")
        dir_paths = set()
        for page in results.pages:
            dir_paths.update(page.prefixes)
        return [FileInfo(posixpath.relpath(path, artifact_path), True, None) for path in dir_paths]

    def _download_file(self, remote_file_path, local_path):
        self._ensure_valid_credentials()
        (bucket, remote_root_path) = self.parse_gcs_uri(self.artifact_uri)
        remote_full_path = posixpath.join(remote_root_path, remote_file_path)
        gcs_bucket = self._get_bucket(bucket)
        gcs_bucket.blob(remote_full_path).download_to_filename(local_path)

    def get_artifact_contents(self, remote_path: str):
        self._ensure_valid_credentials()
        (bucket, remote_root_path) = self.parse_gcs_uri(self.artifact_uri)
        remote_full_path = posixpath.join(remote_root_path, remote_path)
        gcs_bucket = self._get_bucket(bucket)
        # TODO (nikp1172) download_as_bytes will not work for large files,
        #   use https://cloud.google.com/python/docs/reference/storage/latest/google.cloud.storage.blob.Blob#google_cloud_storage_blob_Blob_open
        return gcs_bucket.blob(remote_full_path).download_as_bytes()

    @staticmethod
    def _verify_listed_object_contains_artifact_path_prefix(listed_object_path, artifact_path):
        if not listed_object_path.startswith(artifact_path):
            raise MlflowException(
                f"The path of the listed GCS object does not begin with the specified artifact path. "
                f"Artifact path: {artifact_path}. Object path: {listed_object_path}."
            )

    def delete_artifacts(self, artifact_path=None):
        # TODO (chiragjn): This is not the most efficient way to bulk delete things, we need to async this
        self._ensure_valid_credentials()
        (bucket, dest_path) = self.parse_gcs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)

        gcs_bucket = self._get_bucket(bucket)
        blobs = gcs_bucket.list_blobs(prefix=dest_path)
        for blob in blobs:
            self._verify_listed_object_contains_artifact_path_prefix(
                listed_object_path=blob.name, artifact_path=dest_path
            )
            blob.delete()

    def _get_signed_uri(self, method: str, artifact_path: str, expires_in: int = 1800) -> str:
        self._ensure_valid_credentials()
        (bucket, dest_path) = self.parse_gcs_uri(self.artifact_uri)
        gcs_bucket = self._get_bucket(bucket)
        blob_path = posixpath.join(dest_path, artifact_path)
        blob = gcs_bucket.blob(blob_path)
        url = blob.generate_signed_url(
            version="v4",
            expiration=expires_in,
            method=method,
            service_account_email=self._credentials.service_account_email,
            access_token=self._credentials.token,
        )
        return url

    def get_read_signed_uri(self, artifact_path: str, expires_in: int = 1800) -> str:
        """
        Generates a v4 signed URL for downloading a blob.
        """
        return self._get_signed_uri(
            method="GET", artifact_path=artifact_path, expires_in=expires_in
        )

    def get_write_signed_uri(self, artifact_path: str, expires_in: int = 1800) -> str:
        """
        Generates a v4 signed URL for uploading a blob.
        """
        return self._get_signed_uri(
            method="PUT", artifact_path=artifact_path, expires_in=expires_in
        )
