import argparse
import enum
import json
import os
import tempfile
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union
from weakref import ReferenceType

from mlflow.entities import RunStatus
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.logger import (
    _add_prefix,
    _convert_params,
    _sanitize_callable_params,
)

import mlfoundry
from mlfoundry.enums import EnumMissingMixin
from mlfoundry.logger import logger
from mlfoundry.mlfoundry_api import _resolve_ml_repo_name
from mlfoundry.mlfoundry_run import MlFoundryRun

LIGHTNING_ARTIFACTS_PATH = "lightning/"
LIGHTNING_MODELS_PATH = os.path.join(LIGHTNING_ARTIFACTS_PATH, "models")


class LogModelStrategy(EnumMissingMixin, enum.Enum):
    # To keep it consistent with transformers
    NO = "no"
    BEST_ONLY = "best"
    BEST_PLUS_LATEST = "best_plus_latest"
    CHECKPOINTS_ON_TRAIN_END = "checkpoints_on_train_end"
    # TODO: Add ALL strategy, and how to handle that?


def _get_artifact_path_for_checkpoint(checkpoint_path):
    return os.path.join(LIGHTNING_MODELS_PATH, os.path.basename(checkpoint_path))


class MlFoundryLightningLogger(LightningLoggerBase):
    """
    Pytorch Lightning Logger for tracking training run on MLFoundry

    Examples:

        ```
        import mlfoundry
        mlfoundry.login()
        # or set API key via `TFY_API_KEY` env variable

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint
        from mlfoundry.integrations.lightning import MlFoundryLightningLogger, LogModelStrategy

        mlf_logger = MlFoundryLightningLogger(
            project_name="lightning",
            run_name="my-lightning-run",
            flatten_params=True,
            log_model_strategy=LogModelStrategy.BEST_PLUS_LATEST,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_accuracy',
            mode='max',
            save_last=True,
            save_top_k=-1
            )
        trainer = Trainer(
            logger=mlf_logger,
            callbacks=[checkpoint_callback], # our model checkpoint callback
            ..
            )
        trainer.fit(model, training_loader, validation_loader)
        ```

        Logger can also be created from an existing run

        ```
        import mlfoundry
        mlfoundry.login()
        # or set API key via `TFY_API_KEY` env variable

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint
        from mlfoundry.integrations.lightning import MlFoundryLightningLogger, LogModelStrategy

        client = mlfoundry.get_client()
        run = client.create_run(project_name="lightning")

        mlf_logger = MlFoundryLightningLogger.from_run(
            run=run,
            auto_end_run=False,
            flatten_params=True,
            log_model_strategy=LogModelStrategy.BEST_PLUS_LATEST,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_accuracy',
            mode='max',
            save_last=True,
            save_top_k=-1
            )
        trainer = Trainer(
            logger=mlf_logger,
            callbacks=[checkpoint_callback], # our model checkpoint callback
            ..
            )
        trainer.fit(model, training_loader, validation_loader)

        run.end()
        ```
    """

    def __init__(
        self,
        ml_repo: Optional[str] = None,
        run_name: Optional[str] = None,
        log_model_strategy: Union[str, LogModelStrategy] = LogModelStrategy.NO,
        flatten_params: Optional[bool] = False,
        prefix: str = "",
        agg_key_funcs: Optional[
            Mapping[str, Callable[[Sequence[float]], float]]
        ] = None,
        agg_default_func: Optional[Callable[[Sequence[float]], float]] = None,
        project_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            ml_repo (str): name of the ML repo to create the run under
            run_name (Optional[str], optional): name of the run. When not provided a run name is automatically generated
            log_model_strategy (LogModelStrategy, optional): The strategy to use for logging models
                - LogModelStrategy.NO (default): Do not log any models
                - LogModelStrategy.BEST_ONLY: Log only the best model checkpoint.
                - LogModelStrategy.BEST_PLUS_LATEST: Log both the latest checkpoint and the best checkpoint (if
                    available) and different from the latest checkpoint. Note that `save_last` argument must be set
                    in the `ModelCheckpoint` callback, else latest checkpoints wont be saved/logged.
                - LogModelStrategy.CHECKPOINTS_ON_TRAIN_END: Log all available checkpoints that are available at the
                    end of training. Note that `save_top_k` must be set in the `ModelCheckpoint` callback,
                    else models from the previous steps wont be saved/logged

                In addition to checkpoints a metadata.json file is logged which stores references to the best checkpoint
                and the latest checkpoint and their corresponding scores

                The files will be logged under `mlf/lightning/models/` under the run artifacts.
            flatten_params (bool, optional): Flatten hierarchical dict
                By default, this is `False`

                For e.g.
                when set to `True`:
                `{'a': {'b': 'c'}}` will be logged as -> `{'a/b': 'c'}`

                when set to `False`:
                Individual keys will be logged ie) `'a':{'b':'c'}`
            prefix (str, optional): This prefix will be added in front of all metric_name logged.

        """
        super().__init__(agg_key_funcs=agg_key_funcs, agg_default_func=agg_default_func)

        log_model_strategy = LogModelStrategy(log_model_strategy)

        self._ml_repo = _resolve_ml_repo_name(
            ml_repo=ml_repo, project_name=project_name
        )
        self._run_name = run_name
        self._log_model_strategy = log_model_strategy
        self._flatten_params = flatten_params
        self._auto_end_run = True
        self._prefix = prefix

        self._run: Optional[MlFoundryRun] = None
        self._checkpoint_callback = None
        self._available_model_paths: List[str] = []

    @property
    @rank_zero_experiment  # Returns the real experiment on rank 0 and otherwise the DummyExperiment
    def run(self) -> MlFoundryRun:
        if self._run:
            return self._run

        if not self._run:
            self._run = mlfoundry.get_client().create_run(
                ml_repo=self._ml_repo,
                run_name=self._run_name,
            )
            self._run_name = self._run.run_name
            self._auto_end_run = True

        return self._run

    @property
    def name(self) -> str:
        return self.run.ml_repo

    @property
    def version(self) -> str:
        return self.run.run_name

    @classmethod
    def from_run(
        cls,
        run: MlFoundryRun,
        log_model_strategy: Union[str, LogModelStrategy] = LogModelStrategy.NO,
        flatten_params: Optional[bool] = False,
        auto_end_run: Optional[bool] = False,
        **kwargs,
    ):
        """
        Create a MLFoundry Lightning Logger from an existing MLFoundry run instance.

        Args:
            run (MlFoundryRun): `MlFoundry` run instance
            log_model_strategy (LogModelStrategy, optional): The strategy to use for logging models
                - LogModelStrategy.NO (default): Do not log any models
                - LogModelStrategy.BEST_ONLY: Log only the best model checkpoint.
                - LogModelStrategy.BEST_PLUS_LATEST: Log both the latest checkpoint and the best checkpoint (if
                    available) and different from the latest checkpoint. Note that `save_last` argument must be set
                    in the `ModelCheckpoint` callback, else latest checkpoints wont be saved/logged.
                - LogModelStrategy.CHECKPOINTS_ON_TRAIN_END: Log all available checkpoints that are available at the
                    end of training. Note that `save_top_k` must be set in the `ModelCheckpoint` callback,
                    else models from the previous steps wont be saved/logged

                In addition to checkpoints a metadata.json file is logged which stores references to the best checkpoint
                and the latest checkpoint and their corresponding scores

                The files will be logged under `mlf/lightning/models/` under the run artifacts.
            flatten_params (bool, optional): Flatten hierarchical dict
                By default, this is `False`

                For e.g.
                when set to `True`:
                `{'a': {'b': 'c'}}` will be logged as -> `{'a/b': 'c'}`

                when set to `False`:
                Individual keys will be logged ie) `'a':{'b':'c'}`
            auto_end_run (bool, optional): if to end the run when training finishes. By default, this is `False`
        """
        instance = cls(
            project_name=run.ml_repo,
            run_name=run.run_name,
            log_model_strategy=log_model_strategy,
            flatten_params=flatten_params,
            **kwargs,
        )
        instance._run = run
        instance._auto_end_run = auto_end_run

        return instance

    @rank_zero_only  # runs only on global rank 0. To prevent multiple logging if there is distributed training
    def log_hyperparams(self, params: argparse.Namespace, *args, **kwargs) -> None:
        """Record hyperparameters.

        Args:
            params (argparse.Namespace): containing the hyperparameters
            args: Optional positional arguments, depends on the specific logger being used
            kwargs: Optional keyword arguments, depends on the specific logger being used
        """
        params = _convert_params(params)  # convert to dict
        params = _sanitize_callable_params(params)  # convert callable params

        self.run.log_params(param_dict=params, flatten_params=self._flatten_params)

    @rank_zero_only
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Records metrics.
        This method logs metrics as soon as it received them

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded, defaults to 0
        """
        step = step or 0
        metrics = _add_prefix(metrics, self._prefix, "/")
        self.run.log_metrics(metric_dict=metrics, step=step)

    def _log_model_on_train_end(
        self, checkpoint_callback: "ReferenceType[ModelCheckpoint]", temp_dir
    ):

        latest_checkpoint_path = checkpoint_callback.last_model_path
        best_checkpoint_path = checkpoint_callback.best_model_path
        metadata = {"checkpoints": {}}

        checkpoints_to_log: List[str] = []
        if self._log_model_strategy == LogModelStrategy.BEST_ONLY:
            latest_checkpoint_path = None
            if best_checkpoint_path is not None and os.path.exists(
                best_checkpoint_path
            ):
                checkpoints_to_log.append(best_checkpoint_path)
            else:
                logger.warning(
                    f"`log_model_strategy` was set to BEST_ONLY but CheckpointCallback did not save any best "
                    f"checkpoint. Cannot automatically log model"
                )

        elif self._log_model_strategy == LogModelStrategy.CHECKPOINTS_ON_TRAIN_END:
            for model_path in self._available_model_paths:
                if os.path.exists(model_path):
                    checkpoints_to_log.append(model_path)

        elif self._log_model_strategy == LogModelStrategy.BEST_PLUS_LATEST:
            if latest_checkpoint_path and os.path.exists(latest_checkpoint_path):
                checkpoints_to_log.append(latest_checkpoint_path)
            if (
                best_checkpoint_path
                and best_checkpoint_path != latest_checkpoint_path
                and os.path.exists(best_checkpoint_path)
            ):
                checkpoints_to_log.append(best_checkpoint_path)
        else:
            raise NotImplementedError("Unreachable!")

        for checkpoint_path in checkpoints_to_log:
            checkpoint_name = os.path.basename(checkpoint_path)
            artifact_path = os.path.join(LIGHTNING_MODELS_PATH, checkpoint_name)
            logger.debug(
                f"Logging checkpoint at {checkpoint_path!r} to {artifact_path!r}"
            )
            self._run.log_artifact_deprecated(
                local_path=checkpoint_path, artifact_path=artifact_path
            )

        if best_checkpoint_path:
            best_model_artifact_path = _get_artifact_path_for_checkpoint(
                os.path.basename(best_checkpoint_path)
            )
            metadata["checkpoints"]["best"] = best_model_artifact_path

        if latest_checkpoint_path:
            latest_model_artifact_path = _get_artifact_path_for_checkpoint(
                os.path.basename(latest_checkpoint_path)
            )
            metadata["checkpoints"]["latest"] = latest_model_artifact_path

        metadata_file_name = "metadata.json"
        metadata_path = os.path.join(temp_dir, metadata_file_name)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        self._run.log_artifact_deprecated(
            local_path=metadata_path,
            artifact_path=LIGHTNING_MODELS_PATH,
        )

    def after_save_checkpoint(
        self, checkpoint_callback: "ReferenceType[ModelCheckpoint]"
    ) -> None:
        """Event called after model checkpoint callback saves a new checkpoint.

        Args:
            checkpoint_callback: the model checkpoint callback instance
        """
        if self._log_model_strategy != LogModelStrategy.NO:
            self._checkpoint_callback = checkpoint_callback

            checkpoints = [
                checkpoint_callback.last_model_path,
                checkpoint_callback.best_model_path,
            ] + list(checkpoint_callback.best_k_models.keys())

            for model_path in checkpoints:
                if model_path not in self._available_model_paths:
                    self._available_model_paths.append(model_path)

    def finalize(self, status: str) -> None:
        """Event called at the end of training. Do any processing that is necessary to finalize an run.

        Args:
            status: Status that the run finished with (e.g. success, failed, aborted)
        """
        if self._checkpoint_callback:
            with tempfile.TemporaryDirectory() as temp_dir:
                self._log_model_on_train_end(self._checkpoint_callback, temp_dir)

        if self._auto_end_run and self.run:
            if status == "success":
                self.run.end()
            else:
                self.run.end(status=RunStatus.FAILED)

    def __del__(self):
        if self._auto_end_run and self.run:
            self.run.end()
