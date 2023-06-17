from __future__ import annotations

import abc
import atexit
import os
import threading
import weakref
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Optional

import mlflow
from mlflow.store.artifact.artifact_repository_registry import (
    _artifact_repository_registry,
)
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracking._tracking_service.utils import _tracking_store_registry
from mlflow.utils.rest_utils import MlflowHostCreds

from mlfoundry import env_vars
from mlfoundry.artifact.truefoundry_artifact_repo import TruefoundryArtifactRepository
from mlfoundry.exceptions import MlFoundryException
from mlfoundry.logger import logger
from mlfoundry.login import CredentialsFileContent, CredentialsFileManager
from mlfoundry.run_utils import (
    append_monitoring_path_to_rest_tracking_uri,
    append_path_to_rest_tracking_uri,
    resolve_tracking_uri,
)
from mlfoundry.tracking.entities import Token, UserInfo
from mlfoundry.tracking.servicefoundry_service import ServicefoundryService
from mlfoundry.tracking.truefoundry_rest_store import (
    TruefoundryRestStore,
    get_rest_store,
)

if TYPE_CHECKING:
    from mlfoundry.mlfoundry_run import MlFoundryRun

TOKEN_REFRESH_LOCK = threading.RLock()


class CredentialProvider(abc.ABC):
    @property
    @abc.abstractmethod
    def token(self) -> Token:
        ...

    @staticmethod
    @abc.abstractmethod
    def can_provide() -> bool:
        ...

    @property
    @abc.abstractmethod
    def tracking_uri(self) -> str:
        ...


class EnvCredentialProvider(CredentialProvider):
    def __init__(self):
        logger.debug("Using env var credential provider")
        self._tracking_uri = resolve_tracking_uri(tracking_uri=None)
        self._auth_service = get_rest_store(
            tracking_uri=self._tracking_uri
        ).get_auth_service()
        api_key = os.getenv(env_vars.API_KEY_GLOBAL)
        if not api_key:
            raise MlFoundryException(
                f"Value of {env_vars.API_KEY_GLOBAL} env var should be non-empty string"
            )
        servicefoundry_service = ServicefoundryService(tracking_uri=self._tracking_uri)
        self._token: Token = servicefoundry_service.get_token_from_api_key(api_key)

    @staticmethod
    def can_provide() -> bool:
        return env_vars.API_KEY_GLOBAL in os.environ

    @property
    def token(self) -> Token:
        with TOKEN_REFRESH_LOCK:
            if self._token.is_going_to_be_expired():
                logger.info("Refreshing access token")
                self._token = self._auth_service.refresh_token(self._token)
            return self._token

    @property
    def tracking_uri(self) -> str:
        return self._tracking_uri


class FileCredentialProvider(CredentialProvider):
    def __init__(self):
        logger.debug("Using file credential provider")
        self._cred_file = CredentialsFileManager()

        with self._cred_file:
            self._last_cred_file_content = self._cred_file.read()
            self._tracking_uri = self._last_cred_file_content.host
            self._token = self._last_cred_file_content.to_token()
            self._auth_service = get_rest_store(
                tracking_uri=self._tracking_uri
            ).get_auth_service()

    @staticmethod
    def can_provide() -> bool:
        with CredentialsFileManager() as cred_file:
            return cred_file.exists()

    @property
    def token(self) -> Token:
        with TOKEN_REFRESH_LOCK:
            if not self._token.is_going_to_be_expired():
                return self._token

            logger.info("Refreshing access token")
            with self._cred_file:
                new_cred_file_content = self._cred_file.read()
                new_token = new_cred_file_content.to_token()
                new_tracking_uri = new_cred_file_content.host

                if new_cred_file_content == self._last_cred_file_content:
                    self._token = self._auth_service.refresh_token(self._token)
                    self._last_cred_file_content = CredentialsFileContent(
                        host=self._tracking_uri,
                        access_token=self._token.access_token,
                        refresh_token=self._token.refresh_token,
                    )
                    self._cred_file.write(self._last_cred_file_content)
                    return self._token

                if (
                    new_tracking_uri == self._tracking_uri
                    and new_token.to_user_info() == self._token.to_user_info()
                ):
                    self._last_cred_file_content = new_cred_file_content
                    self._token = new_token
                    # recursive
                    return self.token

                raise MlFoundryException(
                    "Credentials on disk changed while mlfoundry was running."
                )

    @property
    def tracking_uri(self) -> str:
        return self._tracking_uri


SESSION_LOCK = threading.RLock()


class ActiveRuns:
    def __init__(self):
        self._active_runs: Dict[str, weakref.ReferenceType[MlFoundryRun]] = dict()

    def add_run(self, run: MlFoundryRun):
        with SESSION_LOCK:
            self._active_runs[run.run_id] = weakref.ref(run)

    def remove_run(self, run: MlFoundryRun):
        with SESSION_LOCK:
            if run.run_id in self._active_runs:
                del self._active_runs[run.run_id]

    def close_active_runs(self):
        with SESSION_LOCK:
            for run_ref in list(self._active_runs.values()):
                run = run_ref()
                if run:
                    run.end()
            self._active_runs.clear()


ACTIVE_RUNS = ActiveRuns()
atexit.register(ACTIVE_RUNS.close_active_runs)


class Session:
    def __init__(self, cred_provider: CredentialProvider):
        # Note: Whenever a new session is initialized all the active runs are ended
        self._closed = False
        self._cred_provider: CredentialProvider = cred_provider
        self._user_info: UserInfo = self._cred_provider.token.to_user_info()

    def close(self):
        logger.debug("Closing existing session")
        self._closed = True
        self._user_info = None
        self._cred_provider = None

    def _assert_not_closed(self):
        if self._closed:
            raise MlFoundryException(
                "This session has been deactivated.\n"
                "At a time only one `client` (received from "
                "`mlfoundry.get_client()` function call) can be used"
            )

    @property
    def token(self) -> Token:
        return self._cred_provider.token

    @property
    def user_info(self) -> UserInfo:
        self._assert_not_closed()
        return self._user_info

    @property
    def tracking_uri(self) -> str:
        return self._cred_provider.tracking_uri

    def __eq__(self, other: Session) -> bool:
        return (
            type(self._cred_provider) == type(other._cred_provider)
            and self.user_info == other.user_info
            and self.tracking_uri == other.tracking_uri
        )

    def get_mlfoundry_host_creds_builder(self) -> Callable[[], MlflowHostCreds]:
        tracking_uri = append_path_to_rest_tracking_uri(
            self._cred_provider.tracking_uri
        )

        def builder():
            self._assert_not_closed()
            return MlflowHostCreds(
                host=tracking_uri, token=self._cred_provider.token.access_token
            )

        return builder

    def get_monitoring_foundry_host_creds_builder(
        self,
    ) -> Callable[[], MlflowHostCreds]:
        tracking_uri = append_monitoring_path_to_rest_tracking_uri(
            self._cred_provider.tracking_uri
        )

        def builder():
            self._assert_not_closed()
            return MlflowHostCreds(
                host=tracking_uri, token=self._cred_provider.token.access_token
            )

        return builder


ACTIVE_SESSION: Optional[Session] = None


def get_active_session() -> Session:
    return ACTIVE_SESSION


def init_session() -> Session:
    with SESSION_LOCK:
        final_cred_provider = None
        for cred_provider in [EnvCredentialProvider, FileCredentialProvider]:
            if cred_provider.can_provide():
                final_cred_provider = cred_provider()
                break
        if final_cred_provider is None:
            raise MlFoundryException(
                "Please login using `mlfoundry login` command "
                "or `mlfoundry.login()` function call"
            )
        new_session = Session(cred_provider=final_cred_provider)

        global ACTIVE_SESSION
        if ACTIVE_SESSION and ACTIVE_SESSION == new_session:
            return ACTIVE_SESSION

        ACTIVE_RUNS.close_active_runs()

        if ACTIVE_SESSION:
            ACTIVE_SESSION.close()
        ACTIVE_SESSION = new_session

        host_creds_builder = ACTIVE_SESSION.get_mlfoundry_host_creds_builder()

        tfy_rest_store = TruefoundryRestStore(get_host_creds=host_creds_builder)
        artifact_repository = partial(
            TruefoundryArtifactRepository, rest_store=tfy_rest_store
        )
        _artifact_repository_registry.register("s3", artifact_repository)
        _artifact_repository_registry.register("gs", artifact_repository)
        _artifact_repository_registry.register("wasbs", artifact_repository)

        rest_store = RestStore(get_host_creds=host_creds_builder)

        def _get_rest_store(*args, **kwargs):
            return rest_store

        for scheme in ["http", "https"]:
            _tracking_store_registry.register(scheme, _get_rest_store)

        mlflow.set_tracking_uri(ACTIVE_SESSION.tracking_uri)
        logger.info(
            "Logged in to %r as %r (%s)",
            ACTIVE_SESSION.tracking_uri,
            ACTIVE_SESSION.user_info.user_id,
            ACTIVE_SESSION.user_info.email or ACTIVE_SESSION.user_info.user_type.value,
        )
        return ACTIVE_SESSION
