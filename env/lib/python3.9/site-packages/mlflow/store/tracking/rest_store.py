import typing
import uuid

from mlflow.entities import (
    SENTINEL,
    Artifact,
    ArtifactType,
    ArtifactVersion,
    ArtifactVersionStatus,
    CustomMetric,
    Experiment,
    Feature,
    FileInfo,
    Metric,
    Model,
    ModelSchema,
    ModelVersion,
    Run,
    RunInfo,
    RunLog,
    SignedURL,
    ViewType,
)
from mlflow.entities.mlfoundry_artifacts.utils import dict_to_proto
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.protos import mlfoundry_artifacts_pb2 as mlfa_pb2
from mlflow.protos.service_pb2 import (
    CreateExperiment,
    CreateRun,
    DeleteExperiment,
    DeleteRun,
    DeleteTag,
    GetExperiment,
    GetExperimentByName,
    GetLatestRunLog,
    GetMetricHistory,
    GetRun,
    GetRunByFqn,
    HardDeleteRun,
    ListExperiments,
    ListRunLogs,
    LogBatch,
    LogMetric,
    LogModel,
    LogParam,
    MlflowService,
    RestoreExperiment,
    RestoreRun,
    RunLogInput,
    SearchRuns,
    SetExperimentTag,
    SetTag,
    StoreRunLogs,
    UpdateExperiment,
    UpdateRun,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    _REST_API_PATH_PREFIX,
    call_endpoint,
    extract_api_info_for_service,
)

_METHOD_TO_INFO = extract_api_info_for_service(MlflowService, _REST_API_PATH_PREFIX)
_METHOD_TO_INFO.update(
    extract_api_info_for_service(mlfa_pb2.MlfoundryArtifactsService, _REST_API_PATH_PREFIX)
)


class RestStore(AbstractStore):
    """
    Client for a remote tracking server accessed via REST API calls

    :param get_host_creds: Method to be invoked prior to every REST request to get the
      :py:class:`mlflow.rest_utils.MlflowHostCreds` for the request. Note that this
      is a function so that we can obtain fresh credentials in the case of expiry.
    """

    def __init__(self, get_host_creds):
        super().__init__()
        self.get_host_creds = get_host_creds

    def _call_endpoint(self, api, json_body):
        endpoint, method = _METHOD_TO_INFO[api]
        response_proto = api.Response()
        return call_endpoint(self.get_host_creds(), endpoint, method, json_body, response_proto)

    def list_experiments(
        self,
        ids=None,
        view_type=ViewType.ACTIVE_ONLY,
        max_results=None,
        page_token=None,
    ):
        """
        :param view_type: Qualify requested type of experiments.
        :param max_results: If passed, specifies the maximum number of experiments desired. If not
                            passed, the server will pick a maximum number of results to return.
        :param page_token: Token specifying the next page of results. It should be obtained from
                            a ``list_experiments`` call.
        :param ids: list of experiment ids - which you want to return
                    [Currently not implemented for rest store]
        :return: A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
                 :py:class:`Experiment <mlflow.entities.Experiment>` objects. The pagination token
                 for the next page can be obtained via the ``token`` attribute of the object.
        """
        req_body = message_to_json(
            ListExperiments(view_type=view_type, max_results=max_results, page_token=page_token)
        )
        response_proto = self._call_endpoint(ListExperiments, req_body)
        experiments = [Experiment.from_proto(x) for x in response_proto.experiments]
        # If the response doesn't contain `next_page_token`, `response_proto.next_page_token`
        # returns an empty string (default value for a string proto field).
        token = (
            response_proto.next_page_token if response_proto.HasField("next_page_token") else None
        )
        return PagedList(experiments, token)

    def create_experiment(self, name, tags=None, description=None, storage_integration_fqn=None):
        """
        Create a new experiment.
        If an experiment with the given name already exists, throws exception.

        :param name: Desired name for an experiment
        :param tags: Experiment tags to set upon experiment creation
        :param description: Experiment description to set upon experiment creation
        :param storage_integration_id: Storage integration id to set upon experiment creation. can be none (default will be chosen)

        :return: experiment_id (string) for the newly created experiment if successful, else None

        """
        tag_protos = [tag.to_proto() for tag in tags] if tags else []
        req_body = message_to_json(
            CreateExperiment(
                name=name,
                tags=tag_protos,
                description=description,
                storage_integration_fqn=storage_integration_fqn,
            )
        )
        response_proto = self._call_endpoint(CreateExperiment, req_body)
        return response_proto.experiment_id

    def get_experiment(self, experiment_id):
        """
        Fetch the experiment from the backend store.

        :param experiment_id: String id for the experiment

        :return: A single :py:class:`mlflow.entities.Experiment` object if it exists,
        otherwise raises an Exception.
        """
        req_body = message_to_json(GetExperiment(experiment_id=str(experiment_id)))
        response_proto = self._call_endpoint(GetExperiment, req_body)
        return Experiment.from_proto(response_proto.experiment)

    def delete_experiment(self, experiment_id):
        req_body = message_to_json(DeleteExperiment(experiment_id=str(experiment_id)))
        self._call_endpoint(DeleteExperiment, req_body)

    def restore_experiment(self, experiment_id):
        req_body = message_to_json(RestoreExperiment(experiment_id=str(experiment_id)))
        self._call_endpoint(RestoreExperiment, req_body)

    def rename_experiment(self, experiment_id, new_name):
        req_body = message_to_json(
            UpdateExperiment(experiment_id=str(experiment_id), new_name=new_name)
        )
        self._call_endpoint(UpdateExperiment, req_body)

    def update_experiment(self, experiment_id, description):
        req_body = message_to_json(
            UpdateExperiment(experiment_id=str(experiment_id), description=description)
        )
        self._call_endpoint(UpdateExperiment, req_body)

    def get_run(self, run_id):
        """
        Fetch the run from backend store

        :param run_id: Unique identifier for the run

        :return: A single Run object if it exists, otherwise raises an Exception
        """
        req_body = message_to_json(GetRun(run_uuid=run_id, run_id=run_id))
        response_proto = self._call_endpoint(GetRun, req_body)
        return Run.from_proto(response_proto.run)

    def update_run_info(self, run_id, run_status=None, end_time=None, description=SENTINEL):
        """Updates the metadata of the specified run."""
        updated_run_info = {}
        if run_status is not None:
            updated_run_info["status"] = run_status
        if end_time is not None:
            updated_run_info["end_time"] = end_time
        if description is not SENTINEL:
            updated_run_info["description"] = description or ""
        req_body = message_to_json(UpdateRun(run_uuid=run_id, run_id=run_id, **updated_run_info))
        response_proto = self._call_endpoint(UpdateRun, req_body)
        return RunInfo.from_proto(response_proto.run_info)

    def create_run(self, experiment_id, user_id, start_time, tags, name, description):
        """
        Create a run under the specified experiment ID, setting the run's status to "RUNNING"
        and the start time to the current time.

        :param experiment_id: ID of the experiment for this run
        :param user_id: ID of the user launching this run
        :param start_time: timestamp of the initialization of the run
        :param tags: tags to apply to this run at initialization

        :return: The created Run object
        """
        tag_protos = [tag.to_proto() for tag in tags]
        req_body = message_to_json(
            CreateRun(
                experiment_id=str(experiment_id),
                user_id=user_id,
                start_time=start_time,
                tags=tag_protos,
                name=name,
                description=description or "",
            )
        )
        response_proto = self._call_endpoint(CreateRun, req_body)
        run = Run.from_proto(response_proto.run)
        return run

    def log_metric(self, run_id, metric):
        """
        Log a metric for the specified run

        :param run_id: String id for the run
        :param metric: Metric instance to log
        """
        req_body = message_to_json(
            LogMetric(
                run_uuid=run_id,
                run_id=run_id,
                key=metric.key,
                value=metric.value,
                timestamp=metric.timestamp,
                step=metric.step,
            )
        )
        self._call_endpoint(LogMetric, req_body)

    def log_param(self, run_id, param):
        """
        Log a param for the specified run

        :param run_id: String id for the run
        :param param: Param instance to log
        """
        req_body = message_to_json(
            LogParam(run_uuid=run_id, run_id=run_id, key=param.key, value=param.value)
        )
        self._call_endpoint(LogParam, req_body)

    def set_experiment_tag(self, experiment_id, tag):
        """
        Set a tag for the specified experiment

        :param experiment_id: String ID of the experiment
        :param tag: ExperimentRunTag instance to log
        """
        req_body = message_to_json(
            SetExperimentTag(experiment_id=experiment_id, key=tag.key, value=tag.value)
        )
        self._call_endpoint(SetExperimentTag, req_body)

    def set_tag(self, run_id, tag):
        """
        Set a tag for the specified run

        :param run_id: String ID of the run
        :param tag: RunTag instance to log
        """
        req_body = message_to_json(
            SetTag(run_uuid=run_id, run_id=run_id, key=tag.key, value=tag.value)
        )
        self._call_endpoint(SetTag, req_body)

    def delete_tag(self, run_id, key):
        """
        Delete a tag from a run. This is irreversible.
        :param run_id: String ID of the run
        :param key: Name of the tag
        """
        req_body = message_to_json(DeleteTag(run_id=run_id, key=key))
        self._call_endpoint(DeleteTag, req_body)

    def get_metric_history(self, run_id, metric_key):
        """
        Return all logged values for a given metric.

        :param run_id: Unique identifier for run
        :param metric_key: Metric name within the run

        :return: A list of :py:class:`mlflow.entities.Metric` entities if logged, else empty list
        """
        req_body = message_to_json(
            GetMetricHistory(run_uuid=run_id, run_id=run_id, metric_key=metric_key)
        )
        response_proto = self._call_endpoint(GetMetricHistory, req_body)
        return [Metric.from_proto(metric) for metric in response_proto.metrics]

    def _search_runs(
        self, experiment_ids, filter_string, run_view_type, max_results, order_by, page_token
    ):
        experiment_ids = [str(experiment_id) for experiment_id in experiment_ids]
        sr = SearchRuns(
            experiment_ids=experiment_ids,
            filter=filter_string,
            run_view_type=ViewType.to_proto(run_view_type),
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
        )
        req_body = message_to_json(sr)
        response_proto = self._call_endpoint(SearchRuns, req_body)
        runs = [Run.from_proto(proto_run) for proto_run in response_proto.runs]
        # If next_page_token is not set, we will see it as "". We need to convert this to None.
        next_page_token = None
        if response_proto.next_page_token:
            next_page_token = response_proto.next_page_token
        return runs, next_page_token

    def delete_run(self, run_id):
        req_body = message_to_json(DeleteRun(run_id=run_id))
        self._call_endpoint(DeleteRun, req_body)

    def hard_delete_run(self, run_id):
        req_body = message_to_json(HardDeleteRun(run_id=run_id))
        self._call_endpoint(HardDeleteRun, req_body)

    def restore_run(self, run_id):
        req_body = message_to_json(RestoreRun(run_id=run_id))
        self._call_endpoint(RestoreRun, req_body)

    def get_experiment_by_name(
        self,
        experiment_name,
        tenant_name: typing.Optional[str] = None,
    ):
        try:
            req_body = message_to_json(GetExperimentByName(experiment_name=experiment_name))
            response_proto = self._call_endpoint(GetExperimentByName, req_body)
            return Experiment.from_proto(response_proto.experiment)
        except MlflowException as e:
            if e.error_code == databricks_pb2.ErrorCode.Name(
                databricks_pb2.RESOURCE_DOES_NOT_EXIST
            ):
                return None
            raise e

    def log_batch(self, run_id, metrics, params, tags):
        metric_protos = [metric.to_proto() for metric in metrics]
        param_protos = [param.to_proto() for param in params]
        tag_protos = [tag.to_proto() for tag in tags]
        req_body = message_to_json(
            LogBatch(metrics=metric_protos, params=param_protos, tags=tag_protos, run_id=run_id)
        )
        self._call_endpoint(LogBatch, req_body)

    def record_logged_model(self, run_id, mlflow_model):
        req_body = message_to_json(LogModel(run_id=run_id, model_json=mlflow_model.to_json()))
        self._call_endpoint(LogModel, req_body)

    def insert_run_logs(self, run_uuid: str, run_logs: typing.List[RunLog]):
        request_body_proto = StoreRunLogs(run_uuid=run_uuid)
        request_body_proto.run_logs.extend(
            RunLogInput(
                key=run_log.key,
                step=run_log.step,
                timestamp=run_log.timestamp,
                log_type=run_log.log_type,
                value=run_log.value,
                artifact_path=run_log.artifact_path,
            )
            for run_log in run_logs
        )

        request_body = message_to_json(request_body_proto)
        self._call_endpoint(StoreRunLogs, request_body)

    def get_latest_run_log(self, run_uuid: str, key: str, log_type: str) -> RunLog:
        request_body_proto = GetLatestRunLog(run_uuid=run_uuid, key=key, log_type=log_type)
        request_body = message_to_json(request_body_proto)
        response_proto = self._call_endpoint(GetLatestRunLog, request_body)
        return RunLog.from_proto(response_proto.run_log)

    def list_run_logs(
        self,
        run_uuid: str,
        key: typing.Optional[str] = None,
        log_type: typing.Optional[str] = None,
        steps: typing.Optional[typing.List[int]] = None,
    ) -> typing.List[RunLog]:
        request_body_proto = ListRunLogs(
            run_uuid=run_uuid, key=key or "", log_type=log_type or "", steps=steps or []
        )
        request_body = message_to_json(request_body_proto)
        response_proto = self._call_endpoint(ListRunLogs, request_body)
        return [RunLog.from_proto(run_log) for run_log in response_proto.run_logs]

    def get_run_by_fqn(self, fqn: str) -> Run:
        request_body_proto = GetRunByFqn(run_fqn=fqn)
        request_body = message_to_json(request_body_proto)
        response_proto = self._call_endpoint(GetRunByFqn, request_body)
        return Run.from_proto(response_proto.run)

    # Mlfoundry Artifacts methods
    # TODO (chiragjn): consider moving these to another store/mlfoundry_artifacts/rest_store.py
    # TODO (chiragjn): implement list apis for artifacts and models
    # TODO (chiragjn): get_artifact* and get_model* methods break LSP, they return T instead of Optional[T]

    def create_artifact_version(
        self,
        experiment_id: typing.Union[int, str],
        artifact_type: ArtifactType,
        name: str,
    ) -> uuid.UUID:
        message = mlfa_pb2.CreateArtifactVersion(
            experiment_id=str(experiment_id), name=name, artifact_type=artifact_type.to_proto()
        )
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.CreateArtifactVersion, request_body)
        return uuid.UUID(response_proto.id)

    def get_artifact_by_id(self, artifact_id: uuid.UUID, **kwargs) -> Artifact:
        message = mlfa_pb2.GetArtifact(id=str(artifact_id))
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.GetArtifact, request_body)
        return Artifact.from_proto(response_proto.artifact)

    def get_artifact_by_fqn(
        self,
        fqn: str,
    ) -> Artifact:
        message = mlfa_pb2.GetArtifactByFqn(fqn=fqn)
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.GetArtifactByFqn, request_body)
        return Artifact.from_proto(response_proto.artifact)

    def notify_failure_for_artifact_version(
        self,
        version_id: uuid.UUID,
    ):
        message = mlfa_pb2.NotifyArtifactVersionFailure(id=str(version_id))
        request_body = message_to_json(message)
        self._call_endpoint(mlfa_pb2.NotifyArtifactVersionFailure, request_body)

    def list_files_for_artifact_version(
        self, version_id: uuid.UUID, path: typing.Optional[str] = None
    ) -> typing.List[FileInfo]:
        message = mlfa_pb2.ListFilesForArtifactVersion(id=str(version_id), path=path)
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.ListFilesForArtifactVersion, request_body)
        return [FileInfo.from_proto(f) for f in response_proto.files]

    def get_signed_urls_for_artifact_version_read(
        self, version_id: uuid.UUID, paths: typing.List[str]
    ) -> typing.List[SignedURL]:
        message = mlfa_pb2.GetSignedURLsForArtifactVersionRead(id=str(version_id), paths=paths)
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(
            mlfa_pb2.GetSignedURLsForArtifactVersionRead, request_body
        )
        return [SignedURL.from_proto(s) for s in response_proto.signed_urls]

    def get_signed_urls_for_artifact_version_write(
        self, version_id: uuid.UUID, paths: typing.List[str]
    ) -> typing.List[SignedURL]:
        message = mlfa_pb2.GetSignedURLsForArtifactVersionWrite(id=str(version_id), paths=paths)
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(
            mlfa_pb2.GetSignedURLsForArtifactVersionWrite, request_body
        )
        return [SignedURL.from_proto(s) for s in response_proto.signed_urls]

    def finalize_artifact_version(
        self,
        version_id: uuid.UUID,
        run_uuid: str = None,
        description: typing.Optional[str] = None,
        # this is only `Optional` because argument default should be {}
        artifact_metadata: typing.Optional[typing.Dict[str, typing.Any]] = None,
        data_path: typing.Optional[str] = None,
        step: int = 0,
        artifact_size: typing.Optional[int] = None,
        # unused args
        created_by: typing.Optional[str] = None,
    ) -> ArtifactVersion:
        artifact_metadata = artifact_metadata or {}
        message = mlfa_pb2.FinalizeArtifactVersion(
            id=str(version_id),
            run_uuid=run_uuid,
            description=description,
            artifact_metadata=dict_to_proto(artifact_metadata),
            data_path=data_path,
            step=step,
            artifact_size=artifact_size,
        )
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.FinalizeArtifactVersion, request_body)
        return ArtifactVersion.from_proto(response_proto.artifact_version)

    def get_artifact_version_by_id(
        self,
        version_id: uuid.UUID,
        # unused kwargs
        status: typing.Optional[ArtifactVersionStatus] = None,
    ) -> ArtifactVersion:
        message = mlfa_pb2.GetArtifactVersion(id=str(version_id))
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.GetArtifactVersion, request_body)
        return ArtifactVersion.from_proto(response_proto.artifact_version)

    def get_artifact_version_by_fqn(
        self,
        fqn: str,
        # unused kwargs
        status: typing.Optional[ArtifactVersionStatus] = None,
    ) -> ArtifactVersion:
        message = mlfa_pb2.GetArtifactVersionByFqn(fqn=fqn)
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.GetArtifactVersionByFqn, request_body)
        return ArtifactVersion.from_proto(response_proto.artifact_version)

    def update_artifact_version(
        self,
        version_id: uuid.UUID,
        description: typing.Optional[str] = SENTINEL,
        artifact_metadata: typing.Dict[str, typing.Any] = SENTINEL,
    ) -> ArtifactVersion:
        kwargs = dict(description=description)
        if artifact_metadata is not SENTINEL:
            kwargs["artifact_metadata"] = dict_to_proto(artifact_metadata)
        kwargs = {k: v for k, v in kwargs.items() if v is not SENTINEL}
        message = mlfa_pb2.UpdateArtifactVersion(id=str(version_id), **kwargs)
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.UpdateArtifactVersion, request_body)
        return ArtifactVersion.from_proto(response_proto.artifact_version)

    def delete_artifact_version(self, version_id: uuid.UUID):
        message = mlfa_pb2.DeleteArtifactVersion(id=str(version_id))
        request_body = message_to_json(message)
        self._call_endpoint(mlfa_pb2.DeleteArtifactVersion, request_body)

    def get_model_by_id(
        self,
        model_id: uuid.UUID,
    ) -> Model:
        message = mlfa_pb2.GetModel(id=str(model_id))
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.GetModel, request_body)
        return Model.from_proto(response_proto.model)

    def get_model_by_fqn(
        self,
        fqn: str,
    ) -> Model:
        message = mlfa_pb2.GetModelByFqn(fqn=fqn)
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.GetModelByFqn, request_body)
        return Model.from_proto(response_proto.model)

    def get_model_by_name(
        self,
        experiment_id: int,
        name: str,
    ) -> Model:
        message = mlfa_pb2.GetModelByName(experiment_id=str(experiment_id), name=name)
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.GetModelByName, request_body)
        return Model.from_proto(response_proto.model)

    def create_model_version(
        self,
        artifact_version_id: uuid.UUID,
        description: typing.Optional[str] = SENTINEL,
        artifact_metadata: typing.Dict[str, typing.Any] = SENTINEL,
        internal_metadata: typing.Dict[str, typing.Any] = SENTINEL,
        data_path: typing.Optional[str] = SENTINEL,
        step: typing.Optional[int] = SENTINEL,
    ) -> ModelVersion:
        kwargs = dict(
            description=description,
            data_path=data_path,
            internal_metadata=internal_metadata,
            step=step,
        )
        if artifact_metadata is not SENTINEL:
            kwargs["artifact_metadata"] = dict_to_proto(artifact_metadata)
        if internal_metadata is not SENTINEL:
            kwargs["internal_metadata"] = dict_to_proto(internal_metadata)
        kwargs = {k: v for k, v in kwargs.items() if v is not SENTINEL}
        message = mlfa_pb2.CreateModelVersion(
            artifact_version_id=str(artifact_version_id), **kwargs
        )
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.CreateModelVersion, request_body)
        return ModelVersion.from_proto(response_proto.model_version)

    def get_model_version_by_id(
        self,
        version_id: uuid.UUID,
        # unused kwargs
        status: typing.Optional[ArtifactVersionStatus] = None,
    ) -> ModelVersion:
        message = mlfa_pb2.GetModelVersion(id=str(version_id))
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.GetModelVersion, request_body)
        return ModelVersion.from_proto(response_proto.model_version)

    def get_model_version_by_fqn(
        self,
        fqn: str,
        # unused kwargs
        status: typing.Optional[ArtifactVersionStatus] = None,
    ) -> ModelVersion:
        message = mlfa_pb2.GetModelVersionByFqn(fqn=fqn)
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.GetModelVersionByFqn, request_body)
        return ModelVersion.from_proto(response_proto.model_version)

    def update_model_version(
        self,
        version_id: uuid.UUID,
        description: typing.Optional[str] = SENTINEL,
        artifact_metadata: typing.Dict[str, typing.Any] = SENTINEL,
        model_schema: ModelSchema = SENTINEL,
        model_framework: typing.Optional[str] = None,
    ) -> ModelVersion:
        kwargs = dict(description=description, model_framework=model_framework)
        if artifact_metadata is not SENTINEL:
            kwargs["artifact_metadata"] = dict_to_proto(artifact_metadata)
        if model_schema is not SENTINEL:
            kwargs["model_schema"] = model_schema.to_proto()
        kwargs = {k: v for k, v in kwargs.items() if v is not SENTINEL}
        message = mlfa_pb2.UpdateModelVersion(id=str(version_id), **kwargs)
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.UpdateModelVersion, request_body)
        return ModelVersion.from_proto(response_proto.model_version)

    def add_features_to_model_version(
        self, version_id: uuid.UUID, features: typing.List[Feature]
    ) -> ModelVersion:
        message = mlfa_pb2.AddFeaturesToModelVersion(
            id=str(version_id), features=[feature.to_proto() for feature in features]
        )
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.AddFeaturesToModelVersion, request_body)
        return ModelVersion.from_proto(response_proto.model_version)

    def add_custom_metrics_to_model_version(
        self,
        version_id: uuid.UUID,
        custom_metrics: typing.List[CustomMetric],
    ) -> ModelVersion:
        message = mlfa_pb2.AddCustomMetricsToModelVersion(
            id=str(version_id), custom_metrics=[cm.to_proto() for cm in custom_metrics]
        )
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.AddCustomMetricsToModelVersion, request_body)
        return ModelVersion.from_proto(response_proto.model_version)

    def list_artifacts(
        self,
        experiment_id: typing.Union[int, str],
        name: str,
        artifact_types: typing.Optional[typing.List[ArtifactType]] = None,
        max_results: typing.Optional[int] = None,
        page_token: typing.Optional[str] = None,
    ) -> PagedList[Artifact]:
        message = mlfa_pb2.ListArtifacts(
            experiment_id=str(experiment_id), name=name, artifact_types=artifact_types,
            max_results=max_results, page_token=page_token
        )
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.ListArtifacts, request_body)
        return PagedList(
            [Artifact.from_proto(av) for av in response_proto.artifacts],
            token=response_proto.next_page_token,
            total=response_proto.total,
        )

    def list_artifact_versions(
        self,
        artifact_id: uuid.UUID,
        max_results: typing.Optional[int] = None,
        page_token: typing.Optional[str] = None,
        **kwargs,
    ) -> PagedList[ArtifactVersion]:
        message = mlfa_pb2.ListArtifactVersions(
            artifact_id=str(artifact_id), max_results=max_results, page_token=page_token
        )
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.ListArtifactVersions, request_body)
        return PagedList(
            [ArtifactVersion.from_proto(av) for av in response_proto.artifact_versions],
            token=response_proto.next_page_token,
            total=response_proto.total,
        )

    def list_models(
        self,
        experiment_id: typing.Union[int, str],
        name: str,
        max_results: typing.Optional[int] = None,
        page_token: typing.Optional[str] = None,
    ) -> PagedList[Model]:
        message = mlfa_pb2.ListModels(
            experiment_id=str(experiment_id), name=name,
            max_results=max_results, page_token=page_token
        )
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.ListModels, request_body)
        return PagedList(
            [Model.from_proto(m) for m in response_proto.models],
            token=response_proto.next_page_token,
            total=response_proto.total,
        )

    def list_model_versions(
        self,
        model_id: uuid.UUID,
        max_results: typing.Optional[int] = None,
        page_token: typing.Optional[str] = None,
        **kwargs,
    ) -> PagedList[ModelVersion]:
        message = mlfa_pb2.ListModelVersions(
            model_id=str(model_id), max_results=max_results, page_token=page_token
        )
        request_body = message_to_json(message)
        response_proto = self._call_endpoint(mlfa_pb2.ListModelVersions, request_body)
        return PagedList(
            [ModelVersion.from_proto(mv) for mv in response_proto.model_versions],
            token=response_proto.next_page_token,
            total=response_proto.total,
        )


class DatabricksRestStore(RestStore):
    """
    Databricks-specific RestStore implementation that provides different fallback
    behavior when hitting the GetExperimentByName REST API fails - in particular, we only
    fall back to ListExperiments when the server responds with ENDPOINT_NOT_FOUND, rather than
    on all internal server errors. This implementation should be deprecated once
    GetExperimentByName is available everywhere.
    """

    def get_experiment_by_name(self, experiment_name):
        try:
            req_body = message_to_json(GetExperimentByName(experiment_name=experiment_name))
            response_proto = self._call_endpoint(GetExperimentByName, req_body)
            return Experiment.from_proto(response_proto.experiment)
        except MlflowException as e:
            if e.error_code == databricks_pb2.ErrorCode.Name(
                databricks_pb2.RESOURCE_DOES_NOT_EXIST
            ):
                return None
            elif e.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.ENDPOINT_NOT_FOUND):
                # Fall back to using ListExperiments-based implementation.
                for experiments in self._paginate_list_experiments(ViewType.ALL):
                    for experiment in experiments:
                        if experiment.name == experiment_name:
                            return experiment
                return None
            raise e

    def _paginate_list_experiments(self, view_type):
        page_token = None
        while True:
            experiments = self.list_experiments(view_type=view_type, page_token=page_token)
            yield experiments

            if not experiments.token:
                break
            page_token = experiments.token
