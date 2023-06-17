import unittest

import amplitude_tracker


class TestModule(unittest.TestCase):

    def failed(self):
        self.failed = True

    def setUp(self):
        self.failed = False
        amplitude_tracker.write_key = 'testsecret'
        amplitude_tracker.on_error = self.failed

    def test_no_write_key(self):
        amplitude_tracker.write_key = None
        self.assertRaises(Exception, amplitude_tracker.track)

    def test_no_host(self):
        amplitude_tracker.host = None
        self.assertRaises(Exception, amplitude_tracker.track)

    def test_track(self):
        amplitude_tracker.track('userId', 'python module event')
        amplitude_tracker.flush()

    def test_flush(self):
        amplitude_tracker.flush()
