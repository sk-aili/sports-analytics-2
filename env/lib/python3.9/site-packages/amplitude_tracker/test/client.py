from datetime import date
import unittest
import six
import mock
import time

from amplitude_tracker.version import VERSION
from amplitude_tracker.client import Client
from amplitude_tracker.test import patch_amplitude_request


class TestClient(unittest.TestCase):

    def fail(self, e, batch):
        """Mark the failure handler"""
        self.failed = True

    def setUp(self):
        self.failed = False
        self.client = Client('testsecret', on_error=self.fail)

    def test_requires_write_key(self):
        self.assertRaises(AssertionError, Client)

    def test_empty_flush(self):
        self.client.flush()

    @patch_amplitude_request()
    def test_basic_track(self):
        client = self.client
        success, msg = client.track(
            user_id='userId', event_type='python test event')
        client.flush()
        self.assertTrue(success)
        self.assertFalse(self.failed)

        self.assertEqual(msg['event_type'], 'python test event')
        self.assertTrue(isinstance(msg['time'], int))
        self.assertEqual(msg['user_id'], 'userId')
        self.assertEqual(msg['user_properties'], {})
        self.assertEqual(msg['event_properties'], {'lib_version': VERSION})

    @patch_amplitude_request()
    def test_stringifies_user_id(self):
        # A large number that loses precision in node:
        # node -e "console.log(157963456373623802 + 1)" > 157963456373623800
        client = self.client
        success, msg = client.track(
            user_id=157963456373623802, event_type='python test event')
        client.flush()
        self.assertTrue(success)
        self.assertFalse(self.failed)

        self.assertEqual(msg['user_id'], '157963456373623802')

    @patch_amplitude_request()
    def test_advanced_track(self):
        client = self.client
        success, msg = client.track(
            user_id='userId',
            event_type='python test event',
            user_properties={'property': 'value'},
            ip='192.168.0.1',
            time=1234,
            insert_id='insertId')

        self.assertTrue(success)

        self.assertEqual(msg['time'], 1234)
        self.assertEqual(msg['user_properties'], {'property': 'value'})
        self.assertEqual(msg['event_properties'], {'lib_version': VERSION})
        self.assertEqual(msg['ip'], '192.168.0.1')
        self.assertEqual(msg['event_type'], 'python test event')
        self.assertEqual(msg['insert_id'], 'insertId')
        self.assertEqual(msg['user_id'], 'userId')
        self.assertEqual(msg['platform'], 'amplitude-python')

    @patch_amplitude_request()
    def test_flush(self):
        client = self.client
        # set up the consumer with more requests than a single batch will allow
        for i in range(1000):
            success, msg = client.track(
                user_id=157963456373623803, event_type='python test event')
        # We can't reliably assert that the queue is non-empty here; that's
        # a race condition. We do our best to load it up though.
        client.flush()
        # Make sure that the client queue is empty after flushing
        self.assertTrue(client.queue.empty())

    @patch_amplitude_request()
    def test_shutdown(self):
        client = self.client
        # set up the consumer with more requests than a single batch will allow
        for i in range(1000):
            success, msg = client.track(
                user_id=157963456373623803, event_type='python test event')
        client.shutdown()
        # we expect two things after shutdown:
        # 1. client queue is empty
        # 2. consumer thread has stopped
        self.assertTrue(client.queue.empty())
        for consumer in client.consumers:
            self.assertFalse(consumer.is_alive())

    @patch_amplitude_request()
    def test_synchronous(self):
        client = Client('testsecret', sync_mode=True)

        success, message = client.track(
            user_id=157963456373623803, event_type='python test event')
        self.assertFalse(client.consumers)
        self.assertTrue(client.queue.empty())
        self.assertTrue(success)

    @patch_amplitude_request()
    def test_overflow(self):
        client = Client('testsecret', max_queue_size=1)
        # Ensure consumer thread is no longer uploading
        client.join()

        for i in range(10):
            client.track(
                user_id=157963456373623803, event_type='python test event')

        success, msg = client.track(
                user_id=157963456373623803, event_type='python test event')
        # Make sure we are informed that the queue is at capacity
        self.assertFalse(success)

    @patch_amplitude_request()
    def test_success_on_invalid_write_key(self):
        client = Client('bad_key', on_error=self.fail)
        client.track('user_id', 'event_type')
        client.flush()
        self.assertFalse(self.failed)

    def test_unicode(self):
        Client(six.u('unicode_key'))

    @patch_amplitude_request()
    def test_numeric_user_id(self):
        self.client.track(1234, 'python event')
        self.client.flush()
        self.assertFalse(self.failed)

    def test_debug(self):
        Client('bad_key', debug=True)

    @patch_amplitude_request()
    def test_identify_with_date_object(self):
        client = self.client
        success, msg = client.track(
            user_id=157963456373623803,
            event_type='python test event',
            user_properties={'birthdate': date(1981, 2, 2)})
        client.flush()
        self.assertTrue(success)
        self.assertFalse(self.failed)

        self.assertEqual(
            msg['user_properties'], {'birthdate': date(1981, 2, 2)})

    @patch_amplitude_request()
    def test_gzip(self):
        client = Client('testsecret', on_error=self.fail, gzip=True)
        for _ in range(10):
            client.track(
                user_id=157963456373623803,
                event_type='python test event',
                user_properties={'trait': 'value'})
        client.flush()
        self.assertFalse(self.failed)

    @patch_amplitude_request()
    def test_user_defined_flush_at(self):
        client = Client('testsecret', on_error=self.fail,
                        flush_at=10, flush_interval=3)

        def mock_post_fn(*args, **kwargs):
            self.assertEqual(len(kwargs['events']), 10)

        # the post function should be called 2 times, with a batch size of 10
        # each time.
        with mock.patch('amplitude_tracker.consumer.post', side_effect=mock_post_fn) \
                as mock_post:
            for _ in range(20):
                client.track(
                    user_id=157963456373623803,
                    event_type='python test event',
                    user_properties={'trait': 'value'})
            time.sleep(1)
            self.assertEqual(mock_post.call_count, 2)

    @patch_amplitude_request()
    def test_user_defined_timeout(self):
        client = Client('testsecret', timeout=10)
        for consumer in client.consumers:
            self.assertEqual(consumer.timeout, 10)

    @patch_amplitude_request()
    def test_default_timeout_15(self):
        client = Client('testsecret')
        for consumer in client.consumers:
            self.assertEqual(consumer.timeout, 15)
