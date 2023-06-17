import typing

from mlflow.protos.service_pb2 import Columns as ProtoColumns


# "Columns", this name is too broad. Think whether this
# will collide with some other concept
class Columns:
    def __init__(
        self,
        metric_names: typing.Iterator[str],
        param_names: typing.Iterator[str],
        tag_names: typing.Iterator[str],
    ):
        self._metric_names = metric_names
        self._param_names = param_names
        self._tag_names = tag_names

    def to_proto(self) -> ProtoColumns:
        columns = ProtoColumns()
        columns.metric_names.extend(self._metric_names)
        columns.param_names.extend(self._param_names)
        columns.tag_names.extend(self._tag_names)
        return columns
