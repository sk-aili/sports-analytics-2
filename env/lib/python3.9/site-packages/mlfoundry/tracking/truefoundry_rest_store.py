import typing
from functools import partial
from urllib.parse import urlparse

from mlflow.entities import FileInfo
from mlflow.protos.service_pb2 import ListArtifacts, MlflowService
from mlflow.store.tracking import rest_store
from mlflow.tracking._tracking_service.utils import _get_default_host_creds
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    _REST_API_PATH_PREFIX,
    extract_api_info_for_service,
    http_request_safe,
)

from mlfoundry.run_utils import append_path_to_rest_tracking_uri
from mlfoundry.tracking.auth_service import AuthService
from mlfoundry.tracking.entities import ArtifactCredential, AuthServerInfo

_METHOD_TO_INFO = extract_api_info_for_service(MlflowService, _REST_API_PATH_PREFIX)


class TruefoundryRestStore(
    rest_store.RestStore,
):
    def list_artifacts(self, run_id, path) -> typing.List[FileInfo]:
        infos = []
        page_token = None
        while True:
            if page_token:
                json_body = message_to_json(
                    ListArtifacts(run_id=run_id, path=path, page_token=page_token)
                )
            else:
                json_body = message_to_json(ListArtifacts(run_id=run_id, path=path))
            response = self._call_endpoint(ListArtifacts, json_body)
            artifact_list = response.files
            # If `path` is a file, ListArtifacts returns a single list element with the
            # same name as `path`. The list_artifacts API expects us to return an empty list in this
            # case, so we do so here.
            if (
                len(artifact_list) == 1
                and artifact_list[0].path == path
                and not artifact_list[0].is_dir
            ):
                return []
            for output_file in artifact_list:
                artifact_size = None if output_file.is_dir else output_file.file_size
                infos.append(
                    FileInfo(output_file.path, output_file.is_dir, artifact_size)
                )
            if len(artifact_list) == 0 or not response.next_page_token:
                break
            page_token = response.next_page_token
        return infos

    def get_artifact_read_credentials(self, run_id, path) -> ArtifactCredential:
        host_cred = self.get_host_creds()
        response = http_request_safe(
            host_creds=host_cred,
            endpoint="/api/2.0/mlflow/artifacts/credentials-read",
            method="get",
            params={"run_id": run_id, "path": path},
        )

        response = response.json()

        artifact_credential = ArtifactCredential(
            run_id=response["run_id"],
            path=response["path"],
            signed_uri=response["signed_uri"],
        )
        return artifact_credential

    def get_artifact_write_credential(self, run_id, path) -> ArtifactCredential:
        host_cred = self.get_host_creds()
        response = http_request_safe(
            host_creds=host_cred,
            endpoint="/api/2.0/mlflow/artifacts/credentials-write",
            method="get",
            params={"run_id": run_id, "path": path},
        )

        response = response.json()

        artifact_credential = ArtifactCredential(
            run_id=response["run_id"],
            path=response["path"],
            signed_uri=response["signed_uri"],
        )
        return artifact_credential

    def get_auth_server_info(self) -> AuthServerInfo:
        host_cred = self.get_host_creds()
        host_cred.token = None
        parsed_tracking_uri = urlparse(host_cred.host)
        host = parsed_tracking_uri.netloc
        response = http_request_safe(
            host_creds=host_cred,
            endpoint="/api/2.0/mlflow/tenant-id",
            method="get",
            params={"host_name": host},
            timeout=3,
            max_retries=0,
        )
        response = response.json()
        return AuthServerInfo.parse_obj(response)

    def get_auth_service(self) -> AuthService:
        return AuthService(self.get_auth_server_info())


def get_rest_store(tracking_uri: str) -> TruefoundryRestStore:
    tracking_uri = append_path_to_rest_tracking_uri(tracking_uri)
    get_cred = partial(_get_default_host_creds, tracking_uri)
    return TruefoundryRestStore(get_cred)
