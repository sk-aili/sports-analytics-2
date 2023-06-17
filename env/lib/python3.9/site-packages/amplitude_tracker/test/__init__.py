from contextlib import contextmanager
import functools
import unittest
import pkgutil
import logging
import sys
import json
import requests
from mock import patch, MagicMock


def all_names():
    for _, modname, _ in pkgutil.iter_modules(__path__):
        yield 'amplitude_tracker.test.' + modname


def all():
    logging.basicConfig(stream=sys.stderr)
    return unittest.defaultTestLoader.loadTestsFromNames(all_names())


@contextmanager
def patch_requests(http_code=None, response=None):
    response = response or {}
    http_code = http_code or 200
    def get_response(url, **kwargs):
        resp = requests.Response()
        resp.url = url
        resp.status_code = http_code
        resp._content = json.dumps(response).encode("utf-8")
        return resp

    with patch.multiple('requests.Session', post=MagicMock(side_effect=get_response)):
        yield {'post': requests.post}


def patch_amplitude_request(http_code=None, response=None):
    def decorator(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            with patch_requests(http_code=http_code, response=response):
                return func(*args, **kwargs)
        return inner
    return decorator
