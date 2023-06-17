import json
import logging
import os
import posixpath
import shutil
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import mlflow
from mlflow.tracking import MlflowClient

from mlfoundry.exceptions import MlFoundryException
from mlfoundry.log_types.artifacts.constants import DESCRIPTION_MAX_LENGTH

logger = logging.getLogger("mlfoundry")


def _copy_tree(src_path, dest_path, symlinks=False, ignore_dangling_symlinks=False):
    os.makedirs(dest_path, exist_ok=True)
    for item in os.listdir(src_path):
        src = os.path.join(src_path, item)
        dest = os.path.join(dest_path, item)
        if os.path.isdir(src):
            _copy_tree(
                src,
                dest,
                symlinks=symlinks,
                ignore_dangling_symlinks=ignore_dangling_symlinks,
            )
        else:
            shutil.copy2(src, dest, follow_symlinks=True)


def _get_mlflow_client() -> MlflowClient:
    tracking_uri = mlflow.get_tracking_uri()
    if not tracking_uri:
        # TODO (chiragjn): We should link docs here
        raise MlFoundryException(
            "mlfoundry client needs to be configured before ModelVersion can be initialised. Please use"
            "mlfoundry.get_client() to configure and initialise mlfoundry"
        )
    return MlflowClient()


def is_destination_path_dirlike(dest_path) -> bool:
    if not dest_path:
        return True

    if dest_path.endswith(os.sep) or dest_path.endswith(posixpath.sep):
        return True

    if os.path.exists(dest_path) and os.path.isdir(dest_path):
        return True

    return False


def _copy_additional_files(
    root_dir: str,
    files_dir: str,  # relative to root dir e.g. "files/"
    model_dir: Optional[str],  # relative to files_dir e.g "model/"
    additional_files: Sequence[Tuple[Union[str, Path], Optional[str]]],
):
    """

    File copying examples:
        # non ambiguous
        # a.txt -> /tmp/                            result /tmp/a.txt
        # a.txt -> /tmp/a/                          result /tmp/a/a.txt
        # a.txt -> /tmp/a/b/c/d.txt                 result /tmp/a/b/c/d.txt
        # .gitignore -> /tmp/.gitignore             result /tmp/.gitignore

        # ambiguous but destination directory exists
        # a.txt -> /tmp                             result /tmp/a.txt
        # a.txt -> /tmp/a (and /tmp/a/ exists)      result /tmp/a/a.txt

        # ambiguous - when the destination can't be reliably distinguished as a directory
        # a -> /tmp/a                                result /tmp/a
        # a -> /tmp/b                                result /tmp/b
        # a -> /tmp/a.txt                            result /tmp/a.txt
        # .gitignore -> /tmp/.gitinclude             result /tmp/.gitinclude
        # a.txt -> /tmp/a                            result /tmp/a
    """
    for src_path, dest_path in additional_files:
        src_path = str(src_path)
        if not os.path.exists(src_path):
            raise MlFoundryException(
                f"Source path {src_path!r} in `additional_files` does not exist."
            )
        dest_path = dest_path or ""
        normalized_path = os.path.normpath(dest_path)
        if dest_path.endswith(os.sep) or dest_path.endswith(posixpath.sep):
            normalized_path += os.sep
        dest_path = normalized_path.lstrip(os.sep)

        if model_dir and dest_path.startswith(model_dir):
            raise MlFoundryException(
                f"Destination path {dest_path!r} in `additional_files` conflicts with "
                f"reserved path {model_dir!r}/ which is being used to serialize the model. "
                f"Please provide a different destination path."
            )

        files_abs_dir = os.path.join(root_dir, files_dir)
        dest_abs_path = os.path.join(files_abs_dir, dest_path)

        if os.path.isfile(src_path):
            _src = src_path
            if is_destination_path_dirlike(dest_abs_path):
                os.makedirs(dest_abs_path, exist_ok=True)
                _dst = os.path.relpath(
                    os.path.join(dest_abs_path, os.path.basename(_src)), files_abs_dir
                )
            else:
                os.makedirs(os.path.dirname(dest_abs_path), exist_ok=True)
                _dst = os.path.relpath(dest_abs_path, files_abs_dir)
            logger.info(f"Adding file {_src} as /{_dst}")
            shutil.copy2(src_path, dest_abs_path, follow_symlinks=True)
        elif os.path.isdir(src_path):
            os.makedirs(dest_abs_path, exist_ok=True)
            _src = src_path.rstrip("/")
            _dst = os.path.relpath(dest_abs_path, files_abs_dir).rstrip("/")
            logger.info(f"Adding contents of {_src}/ to /{_dst}/")
            _copy_tree(
                src_path=src_path,
                dest_path=dest_abs_path,
                symlinks=True,
                ignore_dangling_symlinks=False,
            )


def _validate_description(description: Optional[str]):
    if description is not None:
        if not isinstance(description, str):
            raise MlFoundryException(
                f"`description` must be either `None` or type `str`"
            )
        if len(description) > DESCRIPTION_MAX_LENGTH:
            raise MlFoundryException(
                f"`description` cannot be longer than {DESCRIPTION_MAX_LENGTH} characters"
            )


def _validate_artifact_metadata(metadata: Dict[str, Any]):
    if not isinstance(metadata, dict):
        raise MlFoundryException("`metadata` must be json serializable dict")
    try:
        json.dumps(metadata)
    except ValueError as ve:
        raise MlFoundryException("`metadata` must be json serializable dict") from ve


@lru_cache(maxsize=32)
def get_model_version_id_from_fqn(fqn) -> uuid.UUID:
    client = _get_mlflow_client()
    model_version = client.get_model_version_by_fqn(fqn=fqn)
    return model_version.id
