from datetime import date, datetime
import logging
import json
from requests import sessions

from amplitude_tracker.version import VERSION

_session = sessions.Session()


def post(write_key, timeout=15, **kwargs):
    """Post the `kwargs` to the API"""
    log = logging.getLogger('amplitude')
    body = kwargs
    url = 'https://api.amplitude.com/2/httpapi'
    body.update({
        'api_key': write_key,
        'options': {'min_id_length': 0}
    })
    log.debug('making request: %s', body)
    data = json.dumps(body, cls=DatetimeSerializer)
    res = _session.post(url, data=data, timeout=timeout)

    if res.status_code == 200:
        log.debug('data uploaded successfully')
        return res

    try:
        payload = res.json()
        log.debug('received response: %s', payload)
        raise APIError(res.status_code, payload['code'], payload['error'])
    except ValueError:
        raise APIError(res.status_code, 'unknown', res.text)


class APIError(Exception):

    def __init__(self, status, code, message):
        self.message = message
        self.status = status
        self.code = code

    def __str__(self):
        msg = "[Amplitude] {0}: {1} ({2})"
        return msg.format(self.code, self.message, self.status)


class DatetimeSerializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()

        return json.JSONEncoder.default(self, obj)
