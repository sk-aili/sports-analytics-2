import functools
import typing
import uuid
from abc import ABCMeta, abstractmethod

from pydantic import BaseModel

from mlflow.entities import (
    SENTINEL,
    Artifact,
    ArtifactType,
    ArtifactVersion,
    ArtifactVersionInTransit,
    ArtifactVersionStatus,
    ArtifactVersionTransitStatus,
    Columns,
    CustomMetric,
    Feature,
    FileInfo,
    LatestRunLog,
    Metric,
    Model,
    ModelSchema,
    ModelVersion,
    Run,
    RunInfo,
    RunLog,
    SignedURL,
    SubjectType,
    ViewType,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT

_ET = typing.TypeVar("_ET", bound=BaseModel)


def _err_if_not_exist_wrapper(entity_name):
    def wrapper(
        fn: typing.Callable[..., typing.Optional[_ET]]
    ) -> typing.Callable[..., typing.Optional[_ET]]:
        @functools.wraps(fn)
        def inner(*args, err_if_not_exist=False, **kwargs) -> typing.Optional[_ET]:
            entity = fn(*args, **kwargs)
            if err_if_not_exist and not entity:
                # TODO: this might be too verbose if input args are deep nested objects or don't have str
                raise MlflowException(
                    f"No {entity_name} found with given arguments: {args[1:]!s}, {kwargs!s}",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            return entity

        return inner

    return wrapper


class AbstractStore:
    """
    Abstract class for Backend Storage.
    This class defines the API interface for front ends to connect with various types of backends.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        """
        Empty constructor for now. This is deliberately not marked as abstract, else every
        derived class would be forced to create one.
        """
        pass

    @abstractmethod
    def list_experiments(
        self,
        ids=None,
        view_type=ViewType.ACTIVE_ONLY,
        max_results=None,
        page_token=None,
    ):
        """
        :ids: list of experiment ids which will be fetched with the api
        :param view_type: Qualify requested type of experiments.
        :param max_results: If passed, specifies the maximum number of experiments desired. If not
                            passed, all experiments will be returned. However, certain server
                            backend may apply its own limit. Check returned ``PagedList`` token to
                            see if additional experiments are available.
        :param page_token: Token specifying the next page of results. It should be obtained from
                            a ``list_experiments`` call.
        :return: A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
                 :py:class:`Experiment <mlflow.entities.Experiment>` objects. The pagination token
                 for the next page can be obtained via the ``token`` attribute of the object.
        """
        pass

    @abstractmethod
    def create_experiment(
        self, name, tags, description, storage_integration_id=None, storage_integration_fqn=None
    ):
        """
        Create a new experiment.
        If an experiment with the given name already exists, throws exception.

        :param name: Desired name for an experiment
        :param tags: Experiment tags to set upon experiment creation
        :param description: Experiment description to set upon experiment creation
        :param storage_integration_id: Storage integration id to set upon experiment creation. can be none (default will be chosen)
        :param storage_integration_fqn: Storage integration fqn to set upon experiment creation. can be none (default will be chosen)

        :return: experiment_id (string) for the newly created experiment if successful, else None.
        """
        pass

    @abstractmethod
    def get_experiment(self, experiment_id, view_type=ViewType.ALL):
        """
        Fetch the experiment by ID from the backend store.

        :param experiment_id: String id for the experiment

        :return: A single :py:class:`mlflow.entities.Experiment` object if it exists,
            otherwise raises an exception.

        """
        pass

    def get_experiment_by_name(
        self,
        experiment_name,
        tenant_name: typing.Optional[str] = None,
    ):
        """
        Fetch the experiment by name from the backend store.
        This is a base implementation using ``list_experiments``, derived classes may have
        some specialized implementations.

        :param experiment_name: Name of experiment

        :return: A single :py:class:`mlflow.entities.Experiment` object if it exists.
        """
        for experiment in self.list_experiments(ViewType.ALL):
            if experiment.name == experiment_name:
                return experiment
        return None

    @abstractmethod
    def delete_experiment(self, experiment_id):
        """
        Delete the experiment from the backend store. Deleted experiments can be restored until
        permanently deleted.

        :param experiment_id: String id for the experiment
        """
        pass

    def hard_delete_experiment(self, experiment_id):
        raise NotImplementedError()

    @abstractmethod
    def restore_experiment(self, experiment_id):
        """
        Restore deleted experiment unless it is permanently deleted.

        :param experiment_id: String id for the experiment
        """
        pass

    @abstractmethod
    def rename_experiment(self, experiment_id, new_name):
        """
        Update an experiment's name. The new name must be unique.

        :param experiment_id: String id for the experiment
        """
        pass

    @abstractmethod
    def get_run(self, run_id):
        """
        Fetch the run from backend store. The resulting :py:class:`Run <mlflow.entities.Run>`
        contains a collection of run metadata - :py:class:`RunInfo <mlflow.entities.RunInfo>`,
        as well as a collection of run parameters, tags, and metrics -
        :py:class`RunData <mlflow.entities.RunData>`. In the case where multiple metrics with the
        same key are logged for the run, the :py:class:`RunData <mlflow.entities.RunData>` contains
        the value at the latest timestamp for each metric. If there are multiple values with the
        latest timestamp for a given metric, the maximum of these values is returned.

        :param run_id: Unique identifier for the run.

        :return: A single :py:class:`mlflow.entities.Run` object, if the run exists. Otherwise,
                 raises an exception.
        """
        pass

    @abstractmethod
    def update_run_info(self, run_id, run_status, end_time, description):
        """
        Update the metadata of the specified run.

        :return: :py:class:`mlflow.entities.RunInfo` describing the updated run.
        """
        pass

    @abstractmethod
    def create_run(self, experiment_id, user_id, start_time, tags, name, description):
        """
        Create a run under the specified experiment ID, setting the run's status to "RUNNING"
        and the start time to the current time.

        :param experiment_id: String id of the experiment for this run
        :param user_id: ID of the user launching this run

        :return: The created Run object
        """
        pass

    @abstractmethod
    def delete_run(self, run_id):
        """
        Delete a run.

        :param run_id
        """
        pass

    @abstractmethod
    def restore_run(self, run_id):
        """
        Restore a run.

        :param run_id
        """
        pass

    def log_metric(self, run_id, metric):
        """
        Log a metric for the specified run

        :param run_id: String id for the run
        :param metric: :py:class:`mlflow.entities.Metric` instance to log
        """
        self.log_batch(run_id, metrics=[metric], params=[], tags=[])

    def log_param(self, run_id, param):
        """
        Log a param for the specified run

        :param run_id: String id for the run
        :param param: :py:class:`mlflow.entities.Param` instance to log
        """
        self.log_batch(run_id, metrics=[], params=[param], tags=[])

    def set_experiment_tag(self, experiment_id, tag):
        """
        Set a tag for the specified experiment

        :param experiment_id: String id for the experiment
        :param tag: :py:class:`mlflow.entities.ExperimentTag` instance to set
        """
        pass

    def set_tag(self, run_id, tag):
        """
        Set a tag for the specified run

        :param run_id: String id for the run
        :param tag: :py:class:`mlflow.entities.RunTag` instance to set
        """
        self.log_batch(run_id, metrics=[], params=[], tags=[tag])

    @abstractmethod
    def get_metric_history(self, run_id, metric_key):
        """
        Return a list of metric objects corresponding to all values logged for a given metric.

        :param run_id: Unique identifier for run
        :param metric_key: Metric name within the run

        :return: A list of :py:class:`mlflow.entities.Metric` entities if logged, else empty list
        """
        pass

    def search_runs(
        self,
        experiment_ids,
        filter_string,
        run_view_type,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
        order_by=None,
        page_token=None,
    ):
        """
        Return runs that match the given list of search expressions within the experiments.

        :param experiment_ids: List of experiment ids to scope the search
        :param filter_string: A search filter string.
        :param run_view_type: ACTIVE_ONLY, DELETED_ONLY, or ALL runs
        :param max_results: Maximum number of runs desired.
        :param order_by: List of order_by clauses.
        :param page_token: Token specifying the next page of results. It should be obtained from
            a ``search_runs`` call.

        :return: A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
            :py:class:`Run <mlflow.entities.Run>` objects that satisfy the search expressions.
            If the underlying tracking store supports pagination, the token for the next page may
            be obtained via the ``token`` attribute of the returned object; however, some store
            implementations may not support pagination and thus the returned token would not be
            meaningful in such cases.
        """
        runs, token = self._search_runs(
            experiment_ids, filter_string, run_view_type, max_results, order_by, page_token
        )
        return PagedList(runs, token)

    @abstractmethod
    def _search_runs(
        self, experiment_ids, filter_string, run_view_type, max_results, order_by, page_token
    ):
        """
        Return runs that match the given list of search expressions within the experiments, as
        well as a pagination token (indicating where the next page should start). Subclasses of
        ``AbstractStore`` should implement this method to support pagination instead of
        ``search_runs``.

        See ``search_runs`` for parameter descriptions.

        :return: A tuple of ``runs`` and ``token`` where ``runs`` is a list of
            :py:class:`mlflow.entities.Run` objects that satisfy the search expressions,
            and ``token`` is the pagination token for the next page of results.
        """
        pass

    def list_run_infos(
        self,
        experiment_id,
        run_view_type,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
        order_by=None,
        page_token=None,
    ):
        """
        Return run information for runs which belong to the experiment_id.

        :param experiment_id: The experiment id which to search
        :param run_view_type: ACTIVE_ONLY, DELETED_ONLY, or ALL runs
        :param max_results: Maximum number of results desired.
        :param order_by: List of order_by clauses.
        :param page_token: Token specifying the next page of results. It should be obtained from
            a ``list_run_infos`` call.

        :return: A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
            :py:class:`RunInfo <mlflow.entities.RunInfo>` objects that satisfy the search
            expressions. If the underlying tracking store supports pagination, the token for the
            next page may be obtained via the ``token`` attribute of the returned object; however,
            some store implementations may not support pagination and thus the returned token would
            not be meaningful in such cases.
        """
        search_result = self.search_runs(
            [experiment_id], None, run_view_type, max_results, order_by, page_token
        )
        return PagedList([run.info for run in search_result], search_result.token)

    @abstractmethod
    def log_batch(self, run_id, metrics, params, tags):
        """
        Log multiple metrics, params, and tags for the specified run

        :param run_id: String id for the run
        :param metrics: List of :py:class:`mlflow.entities.Metric` instances to log
        :param params: List of :py:class:`mlflow.entities.Param` instances to log
        :param tags: List of :py:class:`mlflow.entities.RunTag` instances to log

        :return: None.
        """
        pass

    @abstractmethod
    def record_logged_model(self, run_id, mlflow_model):
        """
        Record logged model information with tracking store. The list of logged model infos is
        maintained in a mlflow.models tag in JSON format.

        Note: The actual models are logged as artifacts via artifact repository.

        :param run_id: String id for the run
        :param mlflow_model: Model object to be recorded.

        The default implementation is a no-op.

        :return: None.
        """
        pass

    def get_columns(self, experiment_id: str) -> Columns:
        raise NotImplementedError()

    def list_metric_history(
        self, run_id: str, metric_keys: typing.Iterator[str]
    ) -> typing.Dict[str, typing.List[Metric]]:
        raise NotImplementedError()

    def insert_run_logs(self, run_uuid: str, run_logs: typing.List[RunLog]):
        raise NotImplementedError()

    def get_latest_run_log(self, run_uuid: str, key: str, log_type: str) -> RunLog:
        raise NotImplementedError()

    def list_run_logs(
        self,
        run_uuid: str,
        key: typing.Optional[str],
        log_type: typing.Optional[str],
        steps: typing.Optional[typing.List[int]],
    ) -> typing.List[RunLog]:
        raise NotImplementedError()

    def get_run_by_fqn(self, fqn: str) -> Run:
        raise NotImplementedError()

    def hard_delete_run(self, run_id: str) -> None:
        raise NotImplementedError()

    def update_experiment(self, experiment_id: str, description: typing.Optional[str]):
        raise NotImplementedError()

    def get_run_info(self, run_id: str) -> RunInfo:
        raise NotImplementedError()

    def list_latest_run_logs(
        self,
        run_uuid: str,
        key: typing.Optional[str] = None,
        log_type: typing.Optional[str] = None,
    ) -> typing.List[LatestRunLog]:
        raise NotImplementedError()

    # Mlfoundry Artifacts methods
    # TODO (chiragjn): consider moving these to another store/mlfoundry_artifacts/abstract_store.py

    def create_artifact(
        self,
        experiment_id: int,
        artifact_type: ArtifactType,
        name: str,
        created_by: str,
        description: typing.Optional[str] = None,
    ) -> Artifact:
        raise NotImplementedError()

    @_err_if_not_exist_wrapper("...")
    def get_artifact(
        self,
        experiment_id: int,
        artifact_type: ArtifactType,
        name: str,
    ) -> typing.Optional[Artifact]:
        raise NotImplementedError()

    @_err_if_not_exist_wrapper("...")
    def get_artifact_by_id(
        self,
        artifact_id: uuid.UUID,
        view_type: ViewType = ViewType.ACTIVE_ONLY,
    ) -> typing.Optional[Artifact]:
        raise NotImplementedError()

    @_err_if_not_exist_wrapper("...")
    def get_artifact_by_fqn(
        self,
        fqn: str,
    ) -> typing.Optional[Artifact]:
        raise NotImplementedError()

    def delete_artifact_by_id(
        self,
        artifact_id: uuid.UUID,
    ) -> None:
        raise NotImplementedError()

    def list_artifacts(
        self,
        experiment_id: typing.Union[int, str],
        name: str,
        artifact_types: typing.Optional[typing.List[ArtifactType]] = None,
        max_results: typing.Optional[int] = None,
        page_token: typing.Optional[str] = None,
    ) -> PagedList[Artifact]:
        raise NotImplementedError()

    def get_experiment_id_from_artifact_id(
        self,
        artifact_id: uuid.UUID,
    ) -> int:
        raise NotImplementedError()

    def get_experiment_id_from_artifact_version_id(
        self,
        version_id: uuid.UUID,
    ) -> int:
        raise NotImplementedError()

    def get_experiment_id_from_artifact_version_in_transit_id(
        self,
        version_id: uuid.UUID,
    ) -> int:
        raise NotImplementedError()

    def create_artifact_version(
        self,
        experiment_id: typing.Union[int, str],
        artifact_type: ArtifactType,
        name: str,
    ) -> uuid.UUID:
        raise NotImplementedError()

    def create_artifact_version_in_transit(
        self,
        artifact: Artifact,
    ) -> ArtifactVersionInTransit:
        raise NotImplementedError()

    @_err_if_not_exist_wrapper("...")
    def get_artifact_version_in_transit(
        self,
        version_id: uuid.UUID,
        status: ArtifactVersionTransitStatus,
    ) -> typing.Optional[ArtifactVersionInTransit]:
        raise NotImplementedError()

    def update_artifact_version_in_transit_status(
        self,
        version_id: uuid.UUID,
        new_status: ArtifactVersionTransitStatus,
        current_status: ArtifactVersionTransitStatus = ArtifactVersionTransitStatus.CREATED,
    ):
        raise NotImplementedError()

    def list_files_for_artifact_version(
        self, version_id: uuid.UUID, path: typing.Optional[str] = None
    ) -> typing.List[FileInfo]:
        raise NotImplementedError()

    def get_signed_urls_for_artifact_version_read(
        self, version_id: uuid.UUID, paths: typing.List[str]
    ) -> typing.List[SignedURL]:
        raise NotImplementedError()

    def get_signed_urls_for_artifact_version_write(
        self, version_id: uuid.UUID, paths: typing.List[str]
    ) -> typing.List[SignedURL]:
        raise NotImplementedError()

    def notify_failure_for_artifact_version(
        self,
        version_id: uuid.UUID,
    ):
        raise NotImplementedError()

    def finalize_artifact_version(
        self,
        version_id: uuid.UUID,
        run_uuid: str,
        created_by: typing.Optional[str] = None,
        description: typing.Optional[str] = None,
        # this is only `Optional` because argument default should be {}
        artifact_metadata: typing.Optional[typing.Dict[str, typing.Any]] = None,
        data_path: typing.Optional[str] = None,
        step: typing.Optional[int] = None,
        artifact_size: typing.Optional[int] = None,
    ) -> ArtifactVersion:
        raise NotImplementedError()

    @_err_if_not_exist_wrapper("...")
    def get_artifact_version_by_id(
        self, version_id: uuid.UUID, status: typing.Optional[ArtifactVersionStatus] = None
    ) -> typing.Optional[ArtifactVersion]:
        raise NotImplementedError()

    @_err_if_not_exist_wrapper("...")
    def get_artifact_version_by_fqn(
        self, fqn: str, status: typing.Optional[ArtifactVersionStatus] = None
    ) -> typing.Optional[ArtifactVersion]:
        raise NotImplementedError()

    def list_artifact_versions(
        self,
        artifact_id: uuid.UUID,
        artifact_types: typing.Optional[typing.List[ArtifactType]] = None,
        statuses: typing.Optional[typing.List[ArtifactVersionStatus]] = None,
        max_results: typing.Optional[int] = None,
        page_token: typing.Optional[str] = None,
    ) -> PagedList[ArtifactVersion]:
        raise NotImplementedError()

    def update_artifact_version(
        self,
        version_id: uuid.UUID,
        description: typing.Optional[str] = SENTINEL,
        artifact_metadata: typing.Dict[str, typing.Any] = SENTINEL,
    ) -> ArtifactVersion:
        raise NotImplementedError()

    def delete_artifact_version(self, version_id: uuid.UUID):
        raise NotImplementedError()

    @_err_if_not_exist_wrapper("...")
    def get_model_by_name(
        self,
        experiment_id: int,
        name: str,
    ) -> typing.Optional[Model]:
        raise NotImplementedError()

    @_err_if_not_exist_wrapper("...")
    def get_model_by_id(
        self,
        model_id: uuid.UUID,
    ) -> typing.Optional[Model]:
        raise NotImplementedError()

    @_err_if_not_exist_wrapper("...")
    def get_model_by_fqn(
        self,
        fqn: str,
    ) -> typing.Optional[Model]:
        raise NotImplementedError()

    def list_models(
        self,
        experiment_id: typing.Union[int, str],
        name: str,
        max_results: typing.Optional[int] = None,
        page_token: typing.Optional[str] = None,
    ) -> PagedList[Model]:
        raise NotImplementedError()

    def create_model_version(
        self,
        artifact_version_id: uuid.UUID,
        description: typing.Optional[str] = SENTINEL,
        artifact_metadata: typing.Dict[str, typing.Any] = SENTINEL,
        internal_metadata: typing.Dict[str, typing.Any] = SENTINEL,
        data_path: typing.Optional[str] = SENTINEL,
        step: typing.Optional[int] = SENTINEL,
    ) -> ModelVersion:
        raise NotImplementedError()

    @_err_if_not_exist_wrapper("...")
    def get_model_version_by_id(
        self, version_id: uuid.UUID, status: typing.Optional[ArtifactVersionStatus] = None
    ) -> typing.Optional[ModelVersion]:
        raise NotImplementedError()

    @_err_if_not_exist_wrapper("...")
    def get_model_version_by_fqn(
        self, fqn: str, status: typing.Optional[ArtifactVersionStatus] = None
    ) -> typing.Optional[ModelVersion]:
        raise NotImplementedError()

    def list_model_versions(
        self,
        model_id: uuid.UUID,
        statuses: typing.Optional[typing.List[ArtifactVersionStatus]] = None,
        max_results: typing.Optional[int] = None,
        page_token: typing.Optional[str] = None,
    ) -> PagedList[ModelVersion]:
        raise NotImplementedError()

    def update_model_version(
        self,
        version_id: uuid.UUID,
        description: typing.Optional[str] = SENTINEL,
        artifact_metadata: typing.Dict[str, typing.Any] = SENTINEL,
        model_schema: ModelSchema = SENTINEL,
        model_framework: typing.Optional[str] = None,
    ) -> ModelVersion:
        raise NotImplementedError()

    def add_features_to_model_version(
        self, version_id: uuid.UUID, features: typing.List[Feature]
    ) -> ModelVersion:
        raise NotImplementedError()

    def add_custom_metrics_to_model_version(
        self,
        version_id: uuid.UUID,
        custom_metrics: typing.List[CustomMetric],
    ) -> ModelVersion:
        raise NotImplementedError()
