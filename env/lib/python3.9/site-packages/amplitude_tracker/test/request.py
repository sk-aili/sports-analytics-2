from datetime import datetime, date
import unittest
import json
import requests

from amplitude_tracker.request import post, DatetimeSerializer
from amplitude_tracker.test import patch_amplitude_request


class TestRequests(unittest.TestCase):

    @patch_amplitude_request()
    def test_valid_request(self):
        res = post('testsecret', events=[{
            'user_id': 'user_id',
            'event_type': 'python event',
        }])
        self.assertEqual(res.status_code, 200)

    def test_invalid_request_error(self):
        self.assertRaises(Exception, post, 'testsecret',
                          'https://api.amplitude.com', False, '[{]')

    def test_invalid_host(self):
        self.assertRaises(Exception, post, 'testsecret',
                          'api.amplitude.com/', events=[])

    def test_datetime_serialization(self):
        data = {'created': datetime(2012, 3, 4, 5, 6, 7, 891011)}
        result = json.dumps(data, cls=DatetimeSerializer)
        self.assertEqual(result, '{"created": "2012-03-04T05:06:07.891011"}')

    def test_date_serialization(self):
        today = date.today()
        data = {'created': today}
        result = json.dumps(data, cls=DatetimeSerializer)
        expected = '{"created": "%s"}' % today.isoformat()
        self.assertEqual(result, expected)

    @patch_amplitude_request()
    def test_should_not_timeout(self):
        res = post('testsecret', events=[{
            'user_id': 'user_id',
            'event_type': 'python event',
        }], timeout=15)
        self.assertEqual(res.status_code, 200)

    def test_should_timeout(self):
        with self.assertRaises((requests.ConnectTimeout, requests.ReadTimeout)):
            post('testsecret', events=[{
                'user_id': 'user_id',
                'event_type': 'python event',
            }], timeout=0.0001)
