import enum
import json
import os
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from transformers import IntervalStrategy, Trainer
from transformers.integrations import rewrite_logs
from transformers.trainer_callback import TrainerCallback
from transformers.utils import flatten_dict

import mlfoundry
from mlfoundry.enums import EnumMissingMixin
from mlfoundry.logger import logger
from mlfoundry.mlfoundry_api import _resolve_ml_repo_name
from mlfoundry.mlfoundry_run import MlFoundryRun

__all__ = ["HF_MODEL_PATH", "MlFoundryTrainerCallback", "LogModelStrategy"]


HF_ARTIFACTS_PATH = "hf/"
HF_MODELS_PATH = os.path.join(HF_ARTIFACTS_PATH, "models")
# intended to be imported while loading back the model
HF_MODEL_PATH = os.path.join(HF_MODELS_PATH, "model")


class LogModelStrategy(str, EnumMissingMixin, enum.Enum):
    NO = "no"
    BEST_ONLY = "best"
    BEST_PLUS_LATEST = "best_plus_latest"
    # ALL_CHECKPOINTS = "all_checkpoints"  # TODO: Implement this later
    CHECKPOINTS_ON_TRAIN_END = "checkpoints_on_train_end"


def _get_artifact_path_for_checkpoint(checkpoint_path):
    return os.path.join(HF_MODELS_PATH, os.path.basename(checkpoint_path))


# TODO: Allow disabling logging via environment variable and configuring settings from env
# TODO: Add support for state.is_hyper_param_search
class MlFoundryTrainerCallback(TrainerCallback):
    """
    Huggingface Transformers Trainer Callback for tracking training run on MLFoundry

    Examples:

        ```
        import mlfoundry
        mlfoundry.login()
        # or set API key via `TFY_API_KEY` env variable

        from transformers import TrainingArguments, Trainer
        from mlfoundry.integrations.transformers import MlFoundryTrainerCallback, LogModelStrategy

        mlf_cb = MlFoundryTrainerCallback(
            project_name="huggingface",
            run_name="my-hf-run",
            flatten_params=True,
            log_model_strategy=LogModelStrategy.BEST_PLUS_LATEST,
        )

        args = TrainingArguments(..., report_to=[])
        trainer = Trainer(..., args=args, callbacks=[mlf_cb])
        trainer.train()
        ```

        Callback can also be created from an existing run

        ```
        import mlfoundry
        mlfoundry.login()
        # or set API key via `TFY_API_KEY` env variable

        from transformers import TrainingArguments, Trainer
        import mlfoundry
        from mlfoundry.integrations.transformers import MlFoundryTrainerCallback, LogModelStrategy

        client = mlfoundry.get_client()
        run = client.create_run(project_name="huggingface", run_name="my-hf-run")

        mlf_cb = MlFoundryTrainerCallback.from_run(
            run=run,
            auto_end_run=False,
            flatten_params=True,
            log_model_strategy=LogModelStrategy.BEST_PLUS_LATEST,
        )

        args = TrainingArguments(..., report_to=[])
        trainer = Trainer(..., args=args, callbacks=[mlf_cb])
        trainer.train()

        run.end()
        ```
    """

    def __init__(
        self,
        ml_repo: Optional[str] = None,
        run_name: Optional[str] = None,
        flatten_params: bool = False,
        log_model_strategy: Union[str, LogModelStrategy] = LogModelStrategy.NO,
        project_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            ml_repo (str): name of the ML Repo to create the run under
            run_name (Optional[str], optional): name of the run. When not provided a run name is automatically generated
            flatten_params (bool, optional): if to flatten the args and model config dictionaries before logging them,
                By default, this is `False`

                For e.g.

                when set to True,
                `config = {"id2label": {"0": "hello", "1": "bye"}}` will be logged as two parameters as
                `{"config/id2label.0": "hello", "config/id2label.1": "bye"}`

                when set to False,
                `config = {"id2label": {"0": "hello"}}` will be logged as a single parameter as
                `{"config/id2label": '{"0": "hello", "1": "bye"}'}`

            log_model_strategy (LogModelStrategy, optional): The strategy to use for logging models
                - LogModelStrategy.NO (default): Do not log any models
                - LogModelStrategy.BEST_ONLY: Log only the best model checkpoint.
                    Note that `args.metric_for_best_model` and `args.args.save_strategy` needs to be set
                    correctly for this to work
                - LogModelStrategy.BEST_PLUS_LATEST: Log both the latest checkpoint and the best checkpoint (if
                    available) and different from the latest checkpoint
                - LogModelStrategy.CHECKPOINTS_ON_TRAIN_END: Log all available checkpoints in `args.output_dir` at the
                    end of training.

                In addition to checkpoints a metadata.json file is logged which stores references to the best checkpoint
                and the latest checkpoint

                The files will be logged under `mlf/huggingface_models/model/` under the run artifacts.
            log_model (bool, optional): **DEPRECATED**, use `log_model_strategy` instead.
                If to log the generated model artifacts at the end of training. By default, this is `False`.
                If set to True this will set `log_model_strategy` to `LogModelStrategy.BEST_PLUS_LATEST`. Please
                see the description for `log_model_strategy`
        """
        log_model_strategy = LogModelStrategy(log_model_strategy)
        if "log_model" in kwargs:
            _option = (
                LogModelStrategy.BEST_PLUS_LATEST.value
                if kwargs["log_model"]
                else LogModelStrategy.NO.value
            )
            warnings.warn(
                f"`log_model` argument has been deprecated, please use the `log_model_strategy` argument instead. "
                f"E.g. log_model_strategy={_option}",
                FutureWarning,
            )
            if kwargs["log_model"] and log_model_strategy == LogModelStrategy.NO:
                log_model_strategy = LogModelStrategy.BEST_PLUS_LATEST

        self._ml_repo = _resolve_ml_repo_name(
            ml_repo=ml_repo, project_name=project_name
        )
        self._run_name = run_name
        self._run: Optional[MlFoundryRun] = None
        self._auto_end_run = True

        self._flatten_params = flatten_params
        self._log_model_strategy = log_model_strategy
        self._MAX_PARAM_VAL_LENGTH = 250
        self._MAX_PARAMS_TAGS_PER_BATCH = 100
        self._initialized = False
        self._last_save_step = -1

    @classmethod
    def from_run(
        cls,
        run: MlFoundryRun,
        auto_end_run: bool = False,
        flatten_params: bool = False,
        log_model_strategy: Union[str, LogModelStrategy] = LogModelStrategy.NO,
        **kwargs,
    ) -> "MlFoundryTrainerCallback":
        """
        Create a MLFoundry Huggingface Transformers Trainer callback from an existing MLFoundry run instance

        Args:
            run (MlFoundryRun): `MlFoundry` run instance
            auto_end_run (bool, optional): if to end the run when training finishes. By default, this is `False`
            flatten_params (bool, optional): if to flatten the args and model config dictionaries before logging them,
                By default, this is `False`

                For e.g.

                when set to True,
                `config = {"id2label": {"0": "hello", "1": "bye"}}` will be logged as two parameters as
                `{"config/id2label.0": "hello", "config/id2label.1": "bye"}`

                when set to False,
                `config = {"id2label": {"0": "hello"}}` will be logged as a single parameter as
                `{"config/id2label": '{"0": "hello", "1": "bye"}'}`

            log_model_strategy (LogModelStrategy, optional): The strategy to use for logging models
                - LogModelStrategy.NO (default): Do not log any models
                - LogModelStrategy.BEST_ONLY: Log only the best model checkpoint.
                    Note that `args.metric_for_best_model` and `args.args.save_strategy` needs to be set
                    correctly for this to work
                - LogModelStrategy.BEST_PLUS_LATEST: Log both the latest checkpoint and the best checkpoint (if
                    available) and different from the latest checkpoint
                - LogModelStrategy.CHECKPOINTS_ON_TRAIN_END: Log all available checkpoints in `args.output_dir` at the
                    end of training.

                In addition to checkpoints a metadata.json file is logged which stores references to the best checkpoint
                and the latest checkpoint

                The files will be logged under `mlf/huggingface_models/model/` under the run artifacts.
            log_model (bool, optional): **DEPRECATED**, use `log_model_strategy` instead.
                If to log the generated model artifacts at the end of training. By default, this is `False`.
                If set to True this will set `log_model_strategy` to `LogModelStrategy.BEST_PLUS_LATEST`. Please
                see the description for `log_model_strategy`
            **kwargs: Additional keyword arguments to pass to init

        Returns:
            MlFoundryTrainerCallback: an instance of `MlFoundryTrainerCallback`
        """
        instance = cls(
            ml_repo=run.ml_repo,
            run_name=run.run_name,
            flatten_params=flatten_params,
            log_model_strategy=log_model_strategy,
            **kwargs,
        )
        instance._run = run
        instance._auto_end_run = auto_end_run
        return instance

    def on_init_end(self, args, state, control, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        if self._log_model_strategy == LogModelStrategy.BEST_ONLY:
            if args.save_strategy == IntervalStrategy.NO:
                logger.warning(
                    f"`log_model_strategy` is set to BEST_ONLY but "
                    f"`args.save_strategy` is set to NO which means best model cannot be tracked automatically. "
                    f"Falling back to BEST_PLUS_LATEST."
                )
                self._log_model_strategy = LogModelStrategy.BEST_PLUS_LATEST
            elif args.metric_for_best_model is None:
                logger.warning(
                    f"`log_model_strategy` is set to BEST_ONLY but "
                    f"`args.metric_for_best_model` is set to None "
                    f"which means best model cannot be tracked automatically. "
                    f"Falling back to BEST_PLUS_LATEST."
                )
                self._log_model_strategy = LogModelStrategy.BEST_PLUS_LATEST
            # TODO (chiragjn): Check for a case when save steps is lower than save frequency

        if self._log_model_strategy == LogModelStrategy.BEST_PLUS_LATEST:
            if args.save_strategy == IntervalStrategy.NO:
                logger.warning(
                    f"`args.save_strategy` is set to NO "
                    f"which means best model cannot be tracked automatically. "
                    f"Only latest model will be saved."
                )
            elif args.metric_for_best_model is None:
                logger.warning(
                    f"`args.metric_for_best_model` is set to None "
                    f"which means best model cannot be tracked automatically. "
                    f"automatically. Only latest model will be saved."
                )

    def setup(self, args, state, model, **kwargs):
        if not state.is_world_process_zero:
            # If the current process is not the global main process in a distributed training setting do nothing
            return

        logger.info("Automatic MLFoundry logging enabled")

        if not self._run:
            self._auto_end_run = True
            self._run = mlfoundry.get_client().create_run(
                ml_repo=self._ml_repo,
                run_name=self._run_name,
            )

        args_dict = {f"args/{k}": v for k, v in args.to_dict().items()}
        if self._flatten_params:
            args_dict = flatten_dict(args_dict)

        if hasattr(model, "config") and model.config is not None:
            model_config_dict = {
                f"config/{k}": v for k, v in model.config.to_dict().items()
            }
            if self._flatten_params:
                model_config_dict = flatten_dict(model_config_dict)
        else:
            model_config_dict = {}

        params: Dict[str, Any] = {**args_dict, **model_config_dict}
        for name, value in list(params.items()):
            # internally, all values are converted to str
            if len(str(value)) > self._MAX_PARAM_VAL_LENGTH:
                logger.warning(
                    f'Trainer is attempting to log a value of "{value}" for key "{name}" as a parameter. '
                    f"MlFoundry's log_params() only accepts values no longer than {self._MAX_PARAM_VAL_LENGTH} "
                    f"characters so we dropped this attribute. Pass `flatten_params=True` during init to "
                    f"flatten the parameters and avoid this message."
                )
                del params[name]

        # MlFoundry cannot log more than 100 values in one go, so we have to split it
        params_items: List[Tuple[str, Any]] = list(params.items())
        for i in range(0, len(params_items), self._MAX_PARAMS_TAGS_PER_BATCH):
            self._run.log_params(
                dict(params_items[i : i + self._MAX_PARAMS_TAGS_PER_BATCH])
            )
        self._initialized = True

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """
        Event called at the beginning of training.
        """
        if not self._initialized:
            self.setup(args, state, model)

    def _get_latest_checkpoint_path(
        self,
        temp_dir,
        args,
        state,
        model=None,
        tokenizer=None,
        optimizer=None,
        lr_scheduler=None,
        **kwargs,
    ) -> Optional[str]:
        latest_checkpoint_path = None
        expected_latest_checkpoint_name = f"checkpoint-{state.global_step}"
        expected_latest_checkpoint_path = os.path.join(
            args.output_dir, expected_latest_checkpoint_name
        )

        # first check if a checkpoint for latest global step exists on disk
        if os.path.exists(expected_latest_checkpoint_path):
            latest_checkpoint_path = expected_latest_checkpoint_path
        # else check if current `model` is not (re)loaded from the best checkpoint
        # (although best cloud possibly be the latest too), then save it
        elif not args.load_best_model_at_end or state.best_model_checkpoint is None:
            latest_checkpoint_path = os.path.join(
                temp_dir, expected_latest_checkpoint_name
            )
            fake_trainer = Trainer(
                args=args,
                model=model,
                tokenizer=tokenizer,
                optimizers=(optimizer, lr_scheduler),
            )
            fake_trainer.save_model(latest_checkpoint_path)
        # else fallback to the last known save step to be the latest
        elif self._last_save_step > -1:
            # TODO: Check if there is a possibility that this gets deleted in rotation logic
            expected_latest_checkpoint_path = os.path.join(
                args.output_dir, f"checkpoint-{self._last_save_step}"
            )
            if os.path.exists(expected_latest_checkpoint_path):
                latest_checkpoint_path = expected_latest_checkpoint_path

        return latest_checkpoint_path

    def _get_best_checkpoint_path(self, state) -> Optional[str]:
        best_checkpoint_path = None
        if state.best_model_checkpoint is not None and os.path.exists(
            state.best_model_checkpoint
        ):
            best_checkpoint_path = state.best_model_checkpoint
        return best_checkpoint_path

    def _legacy_log_model(
        self,
        temp_dir: str,
        best_checkpoint_path: Optional[str],
        latest_checkpoint_path: Optional[str],
    ):
        # TODO: this is kept only for backward compatibility and duplicates a checkpoint
        #       this should be removed
        checkpoint_path = best_checkpoint_path or latest_checkpoint_path
        if not checkpoint_path:
            return

        self._run.log_artifact_deprecated(
            local_path=checkpoint_path, artifact_path=HF_MODEL_PATH
        )
        metadata = {"checkpoint_name": os.path.basename(checkpoint_path)}
        metadata_filepath = os.path.join(temp_dir, ".mlfoundry.json")
        with open(metadata_filepath, "w") as f:
            json.dump(metadata, f)
        self._run.log_artifact_deprecated(
            local_path=metadata_filepath, artifact_path=HF_MODEL_PATH
        )

    def _log_model_on_train_end(
        self,
        temp_dir,
        args,
        state,
        control,
        **kwargs,
    ):
        metadata = {"checkpoints": {}}
        latest_checkpoint_path = self._get_latest_checkpoint_path(
            temp_dir=temp_dir, args=args, state=state, **kwargs
        )
        latest_checkpoint_name = (
            os.path.basename(latest_checkpoint_path) if latest_checkpoint_path else None
        )
        best_checkpoint_path = self._get_best_checkpoint_path(state=state)
        best_checkpoint_name = (
            os.path.basename(best_checkpoint_path) if best_checkpoint_path else None
        )

        checkpoints_to_log: List[str] = []
        if self._log_model_strategy == LogModelStrategy.BEST_ONLY:
            latest_checkpoint_path = None
            if best_checkpoint_path is not None:
                checkpoints_to_log.append(best_checkpoint_path)
            else:
                logger.warning(
                    f"`log_model_strategy` was set to BEST_ONLY but Trainer did not save any best "
                    f"checkpoint. Cannot automatically log model"
                )
        elif self._log_model_strategy == LogModelStrategy.CHECKPOINTS_ON_TRAIN_END:
            # Log all "checkpoint-\d+" in args.output_dir
            output_dir_contents = os.listdir(args.output_dir)
            for file_or_dir in output_dir_contents:
                if os.path.isdir(file_or_dir) and file_or_dir.startswith("checkpoint-"):
                    checkpoint_path = os.path.join(args.output_dir, file_or_dir)
                    checkpoints_to_log.append(checkpoint_path)
            if (
                latest_checkpoint_path
                and latest_checkpoint_name not in output_dir_contents
            ):
                # it is possible that state.global_step > self._last_save_step
                checkpoints_to_log.append(latest_checkpoint_path)
        elif self._log_model_strategy == LogModelStrategy.BEST_PLUS_LATEST:
            if latest_checkpoint_path:
                checkpoints_to_log.append(latest_checkpoint_path)
            if best_checkpoint_path and best_checkpoint_name != latest_checkpoint_name:
                checkpoints_to_log.append(best_checkpoint_path)
        else:
            raise NotImplementedError("Unreachable!")

        for checkpoint_path in checkpoints_to_log:
            checkpoint_name = os.path.basename(checkpoint_path)
            artifact_path = os.path.join(HF_MODELS_PATH, checkpoint_name)
            logger.debug(
                f"Logging checkpoint at {checkpoint_path!r} to {artifact_path!r}"
            )
            self._run.log_artifact_deprecated(
                local_path=checkpoint_path, artifact_path=artifact_path
            )

        if best_checkpoint_path:
            metadata["checkpoints"]["best"] = _get_artifact_path_for_checkpoint(
                best_checkpoint_path
            )

        if latest_checkpoint_path:
            metadata["checkpoints"]["latest"] = _get_artifact_path_for_checkpoint(
                latest_checkpoint_path
            )

        metadata_file_name = "metadata.json"
        metadata_path = os.path.join(temp_dir, metadata_file_name)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        self._run.log_artifact_deprecated(
            local_path=metadata_path,
            artifact_path=HF_MODELS_PATH,
        )

        self._legacy_log_model(
            temp_dir=temp_dir,
            best_checkpoint_path=best_checkpoint_path,
            latest_checkpoint_path=latest_checkpoint_path,
        )

    def on_train_end(self, args, state, control, **kwargs):
        """
        Event called at the end of training.
        """
        if not self._initialized or not state.is_world_process_zero:
            return
        if self._log_model_strategy != LogModelStrategy.NO:
            logger.info("Logging artifacts. This may take time.")
            with tempfile.TemporaryDirectory() as temp_dir:
                self._log_model_on_train_end(
                    temp_dir=temp_dir, args=args, state=state, control=control, **kwargs
                )
        if self._auto_end_run and self._run:
            self._run.end()

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """
        Event called after logging the last logs.
        """
        if not self._initialized:
            self.setup(args, state, model)
        if not state.is_world_process_zero:
            return
        logs = rewrite_logs(logs)
        metrics = {}
        for k, v in logs.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                metrics[k] = v
            else:
                logger.warning(
                    f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}" as a metric. '
                    f"MlFoundry's log_metric() only accepts float and int types so we dropped this attribute."
                )
        self._run.log_metrics(metric_dict=metrics, step=state.global_step)

    def on_save(self, args, state, control, **kwargs):
        self._last_save_step = state.global_step

    def __del__(self):
        if self._auto_end_run and self._run:
            self._run.end()
