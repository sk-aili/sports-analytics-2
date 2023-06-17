from mlflow.utils.rest_utils import MlflowHostCreds, http_request_safe

from mlfoundry.run_utils import append_servicefoundry_path_to_tracking_ui
from mlfoundry.tracking.entities import Token


class ServicefoundryService:
    def __init__(self, tracking_uri: str, token: str = None):
        self.host_creds = MlflowHostCreds(
            host=append_servicefoundry_path_to_tracking_ui(tracking_uri), token=token
        )

    def get_token_from_api_key(self, api_key: str) -> Token:
        response = http_request_safe(
            host_creds=self.host_creds,
            endpoint="/v1/oauth/api-key/token",
            method="get",
            params={"apiKey": api_key},
            timeout=3,
            max_retries=0,
        )
        response = response.json()
        return Token.parse_obj(response)

    def get_integration_from_id(self, integration_id: str):
        integration_id = integration_id or ""
        response = http_request_safe(
            host_creds=self.host_creds,
            endpoint=f"/v1/integrations/{integration_id}",
            method="get",
            timeout=3,
            max_retries=0,
        )
        return response.json()
