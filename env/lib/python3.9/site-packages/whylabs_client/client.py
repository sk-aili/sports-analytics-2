import os
import tempfile
import urllib3
from datetime import datetime, timezone
from typing import Dict, Union

from . import ApiClient
from . import Configuration
from .apis import LogApi
from .model.segment_tag import SegmentTag


class WhyLabsClient:
    def __init__(self, org_id: str = None, api_key: str = None, endpoint: str = None, configuration=None,
                 pool_threads=1):
        if org_id is None:
            org_id = os.getenv('WHYLABS_ORG_ID')
        if api_key is None:
            api_key = os.getenv('WHYLABS_API_KEY')

        if org_id is None:
            raise EnvironmentError('org_id is not set. Pass it via parameter or WHYLABS_ORG_ID environment variable')
        if api_key is None:
            raise EnvironmentError('api_key is not set. Pass it via parameter or WHYLABS_API_KEY environment variable')

        self._org_id = org_id
        _endpoint = endpoint or "https://api.whylabsapp.com"
        if configuration is None:
            configuration = Configuration(
                host=_endpoint
            )
        else:
            configuration.host = _endpoint
        configuration.api_key['ApiKeyAuth'] = api_key
        configuration.retries = urllib3.util.Retry(
            total=5, 
            status=5, 
            backoff_factor=0.3, 
            status_forcelist=[429, 500, 502, 503, 504]
        )

        self._client = ApiClient(configuration, pool_threads=pool_threads)
        self._log = LogApi(api_client=self._client)

    def log(self,
            dataset_id: str,
            dataset_timestamp: datetime = None,
            segment_tags: Dict[str, str] = None,
            file=None,
            byte_array: Union[bytes, bytearray] = None):
        if file is not None and byte_array is not None:
            raise ValueError('Cannot set both file and byte_array parameters')
        if file is None and byte_array is None:
            raise ValueError('Must pass a file or a binary array')

        if dataset_timestamp is None:
            dataset_timestamp = datetime.now(tz=timezone.utc)
        tags = []
        if segment_tags is not None:
            for k, v in segment_tags.items():
                tags.append(SegmentTag(key=k, value=v))
        tmp_file = None
        if file is None:
            tmp_file = tempfile.NamedTemporaryFile()
            file = tmp_file.name
            with open(file, 'wb') as f:
                f.write(byte_array)

        with open(file, 'rb') as f:
            self._log.log(org_id=self._org_id,
                          model_id=dataset_id,
                          dataset_timestamp=int(dataset_timestamp.timestamp() * 1000),
                          segment_tags=tags,
                          file=f,
                          )
        if tmp_file is not None:
            tmp_file.close()
