from typing import Any, Dict

from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct


def dict_from_proto(message) -> Dict[str, Any]:
    return json_format.MessageToDict(message) or {}


def dict_to_proto(value: Dict[str, Any]) -> Struct:
    message = Struct()
    message.update(value)
    return message
