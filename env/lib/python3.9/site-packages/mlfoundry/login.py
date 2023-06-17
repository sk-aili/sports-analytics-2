from __future__ import annotations

import os
import threading
from functools import lru_cache, wraps
from pathlib import Path
from typing import Optional

import click
from filelock import FileLock, Timeout
from pydantic import BaseModel, Field, constr

from mlfoundry.env_vars import API_KEY_GLOBAL, TRACKING_HOST_GLOBAL
from mlfoundry.exceptions import MlFoundryException
from mlfoundry.logger import logger
from mlfoundry.run_utils import resolve_tracking_uri
from mlfoundry.tracking.entities import Token
from mlfoundry.tracking.servicefoundry_service import ServicefoundryService
from mlfoundry.tracking.truefoundry_rest_store import get_rest_store

CREDENTIALS_DIR = Path.home() / ".truefoundry"
CREDENTIALS_FILE = CREDENTIALS_DIR / "credentials.json"


OLD_CREDENTIALS_DIR = Path.home() / ".mlfoundry"
OLD_CREDENTIALS_FILE = OLD_CREDENTIALS_DIR / "credentials.netrc"

if OLD_CREDENTIALS_FILE.exists():
    logger.warning(
        "%s file is deprecated. You can delete this file now.", OLD_CREDENTIALS_FILE
    )


class CredentialsFileContent(BaseModel):
    access_token: constr(min_length=1) = Field(repr=False)
    refresh_token: Optional[constr(min_length=1)] = Field(repr=False)
    host: constr(min_length=1)

    class Config:
        allow_mutation = False

    def to_token(self) -> Token:
        return Token(access_token=self.access_token, refresh_token=self.refresh_token)


def _ensure_lock_taken(method):
    @wraps(method)
    def lock_guard(self, *method_args, **method_kwargs):
        if not self.lock_taken():
            raise Exception(
                "Trying to write to credential file without using with block"
            )
        return method(self, *method_args, **method_kwargs)

    return lock_guard


CRED_FILE_THREAD_LOCK = threading.RLock()


@lru_cache(maxsize=None)
def get_file_lock(lock_file_path: str) -> FileLock:
    return FileLock(lock_file_path)


class CredentialsFileManager:
    def __init__(
        self,
        credentials_file_path: Path = CREDENTIALS_FILE,
        lock_timeout: float = 60.0,
    ):
        credentials_file_path = credentials_file_path.absolute()

        logger.debug("credential file path %r", credentials_file_path)

        credentials_lock_file_path = f"{credentials_file_path}.lock"

        logger.debug("credential lock file path %r", credentials_lock_file_path)
        self._credentials_file_path = credentials_file_path

        cred_file_dir = credentials_file_path.parent
        cred_file_dir.mkdir(exist_ok=True, parents=True)

        self._file_lock = get_file_lock(credentials_lock_file_path)
        self._lock_timeout = lock_timeout
        self._lock_owner: Optional[int] = None

    def __enter__(self) -> CredentialsFileManager:
        # The lock objects are recursive locks, which means that once acquired, they will not block on successive lock requests:
        lock_aquired = CRED_FILE_THREAD_LOCK.acquire(timeout=self._lock_timeout)
        if not lock_aquired:
            raise MlFoundryException(
                "Could not aquire CRED_FILE_THREAD_LOCK"
                f" in {self._lock_timeout} seconds"
            )
        try:
            self._file_lock.acquire(timeout=self._lock_timeout)
        except Timeout as ex:
            raise MlFoundryException(
                f"Failed to aquire lock on credential file within {self._lock_timeout} seconds.\n"
                "Is any other process trying to login?"
            ) from ex
        logger.debug("Acquired file and thread lock to access credential file")
        self._lock_owner = threading.get_ident()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file_lock.release()
        CRED_FILE_THREAD_LOCK.release()
        logger.debug("Released file and thread lock to access credential file")
        self._lock_owner = None

    def lock_taken(self) -> bool:
        return self._lock_owner == threading.get_ident()

    @_ensure_lock_taken
    def read(self) -> CredentialsFileContent:
        try:
            return CredentialsFileContent.parse_file(self._credentials_file_path)
        except Exception as ex:
            raise MlFoundryException(
                "Error while reading the credentials file "
                f"{self._credentials_file_path}. Please login again "
                "using `mlfoundry login` command or `mlfoundry.login()` function"
            ) from ex

    @_ensure_lock_taken
    def write(self, credentials_file_content: CredentialsFileContent):
        if not isinstance(credentials_file_content, CredentialsFileContent):
            raise MlFoundryException(
                "Only object of type `CredentialsFileContent` is allowed. "
                f"Got {type(credentials_file_content)}"
            )
        logger.debug("Updating the credential file content")
        with open(self._credentials_file_path, "w", encoding="utf8") as file:
            file.write(credentials_file_content.json())

    @_ensure_lock_taken
    def exists(self) -> bool:
        return self._credentials_file_path.exists()


def login(
    tracking_uri: Optional[str] = None,
    relogin: bool = False,
    api_key: Optional[str] = None,
) -> bool:
    """Save API key in local file system for a given `tracking_uri`.

    Args:
        tracking_uri (Optional[str], optional): tracking_uri for the given API key
        relogin (bool, optional): Overwrites the existing API key for the `tracking_uri` if
            set to `True`. If set to `False` and an API key is already present for
            the given `tracking_uri`, then the existing API key is kept untouched.
            Default is `False`.
        api_key (Optional[str], optional): The API key for the given `tracking_uri`.
            If `api_key` is not passed, this function prompts for the API key.

    Returns:
        bool: Returns `True` if any credential was persisted.
    """
    from mlfoundry.session import EnvCredentialProvider

    if API_KEY_GLOBAL in os.environ and TRACKING_HOST_GLOBAL in os.environ:
        logger.warning(
            "Skipping login because environment variables %s and "
            "%s are set and will be used when running mlfoundry. "
            "If you want to relogin then unset these environment keys.",
            TRACKING_HOST_GLOBAL,
            API_KEY_GLOBAL,
        )
        return False

    if EnvCredentialProvider.can_provide():
        logger.warning(
            "TFY_API_KEY env var is already set. "
            "When running mlfoundry, it will use the api key to authorize.\n"
            "Login will just save the credentials on disk."
        )

    tracking_uri = resolve_tracking_uri(tracking_uri).strip("/")
    auth_service = get_rest_store(tracking_uri).get_auth_service()

    cred_file = CredentialsFileManager()

    with cred_file:
        if not relogin and cred_file.exists():
            cred_file_content = cred_file.read()
            if tracking_uri != cred_file_content.host:
                if click.confirm(
                    f"Already logged in to {cred_file_content.host!r}\n"
                    f"Do you want to relogin to {tracking_uri!r}?"
                ):
                    return login(
                        tracking_uri=tracking_uri, relogin=True, api_key=api_key
                    )
            user_info = cred_file_content.to_token().to_user_info()
            user_name_display_info = user_info.email or user_info.user_type.value
            print(
                f"Already logged in to {cred_file_content.host!r} as "
                f"{user_info.user_id!r} ({user_name_display_info!r})\n"
                "Please use `mlfoundry login --relogin` or `mlfoundry.login(relogin=True)` "
                "to force relogin"
            )
            return False

        if api_key:
            logger.debug("Logging in with api key")
            servicefoundry_service = ServicefoundryService(tracking_uri)
            token = servicefoundry_service.get_token_from_api_key(api_key=api_key)
            cred_file_content = CredentialsFileContent(
                access_token=token.access_token,
                refresh_token=token.refresh_token,
                host=tracking_uri,
            )
            cred_file.write(cred_file_content)
        else:
            device_code = auth_service.get_device_code()
            url_to_go = device_code.get_user_clickable_url(tracking_uri=tracking_uri)
            print(f"Opening:- {url_to_go}")
            print(
                "Please click on the above link if it is not "
                "automatically opened in a browser window."
            )
            click.launch(url_to_go)
            token = auth_service.get_token_from_device_code(
                device_code=device_code.device_code, timeout=120
            )
            cred_file_content = CredentialsFileContent(
                access_token=token.access_token,
                refresh_token=token.refresh_token,
                host=tracking_uri,
            )
            cred_file.write(cred_file_content)

    user_info = token.to_user_info()
    user_name_display_info = user_info.email or user_info.user_type.value
    print(
        f"logged in to {cred_file_content.host!r} as {user_info.user_id!r} ({user_name_display_info})"
    )
    return True
