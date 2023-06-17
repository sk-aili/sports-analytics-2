from uuid import uuid4
import time
import logging
import numbers
import atexit

from six import string_types

from amplitude_tracker.utils import clean
from amplitude_tracker.consumer import Consumer
from amplitude_tracker.request import post
from amplitude_tracker.version import VERSION

try:
    import queue
except ImportError:
    import Queue as queue


ID_TYPES = (numbers.Number, string_types)


class Client(object):
    """Create a new Amplitude client."""
    log = logging.getLogger('amplitude')

    def __init__(self, write_key=None, host=None, debug=False,
                 max_queue_size=10000, send=True, on_error=None, flush_at=100,
                 flush_interval=0.5, gzip=False, max_retries=3,
                 sync_mode=False, timeout=15, thread=1):
        require('write_key', write_key, string_types)

        self.queue = queue.Queue(max_queue_size)
        self.write_key = write_key
        self.on_error = on_error
        self.debug = debug
        self.send = send
        self.sync_mode = sync_mode
        self.host = host
        self.gzip = gzip
        self.timeout = timeout

        if debug:
            self.log.setLevel(logging.DEBUG)

        if sync_mode:
            self.consumers = None
        else:
            # On program exit, allow the consumer thread to exit cleanly.
            # This prevents exceptions and a messy shutdown when the
            # interpreter is destroyed before the daemon thread finishes
            # execution. However, it is *not* the same as flushing the queue!
            # To guarantee all messages have been delivered, you'll still need
            # to call flush().
            if send:
                atexit.register(self.join)
            for n in range(thread):
                self.consumers = []
                consumer = Consumer(
                    self.queue, write_key, host=host, on_error=on_error,
                    flush_at=flush_at, flush_interval=flush_interval,
                    gzip=gzip, retries=max_retries, timeout=timeout,
                )
                self.consumers.append(consumer)

                # if we've disabled sending, just don't start the consumer
                if send:
                    consumer.start()

    def track(self, user_id=None, event_type=None, user_properties=None,
            event_properties=None, insert_id=None, **kwargs):
        user_properties = user_properties or {}
        event_properties = event_properties or {}
        require('user_id', user_id, ID_TYPES)
        require('event_type', event_type, string_types)
        require('user_properties', user_properties, dict)
        require('event_properties', event_properties, dict)
        event_properties.update({
            'lib_version': VERSION
        })
        msg = {
            'user_id': user_id,
            'insert_id': insert_id,
            'event_type': event_type,
            'user_properties': user_properties,
            'event_properties': event_properties,
        }
        msg.update(kwargs)
        return self._enqueue(msg)

    def _enqueue(self, msg):
        """Push a new `msg` onto the queue, return `(success, msg)`"""
        msg['platform'] = 'amplitude-python'
        msg['insert_id'] = msg.get('insert_id', uuid4())
        msg['time'] = msg.get('time', int(round(time.time() * 1000)))

        require('time', msg['time'], int)
        require('event_type', msg['event_type'], string_types)
        require('user_properties', msg['user_properties'], dict)
        require('event_properties', msg['event_properties'], dict)
        msg['user_id'] = stringify_id(msg.get('user_id', None))

        msg = clean(msg)
        self.log.debug('queueing: %s', msg)

        # if send is False, return msg as if it was successfully queued
        if not self.send:
            return True, msg

        if self.sync_mode:
            self.log.debug('enqueued with blocking track.')
            post(self.write_key, timeout=self.timeout, events=[msg])

            return True, msg

        try:
            self.queue.put(msg, block=False)
            self.log.debug('message enqueued.')
            return True, msg
        except queue.Full:
            self.log.warning('amplitude queue is full')
            return False, msg

    def flush(self):
        """Forces a flush from the internal queue to the server"""
        queue = self.queue
        size = queue.qsize()
        queue.join()
        # Note that this message may not be precise, because of threading.
        self.log.debug('successfully flushed about %s items.', size)

    def join(self):
        """Ends the consumer thread once the queue is empty.
        Blocks execution until finished
        """
        for consumer in self.consumers:
            consumer.pause()
            try:
                consumer.join()
            except RuntimeError:
                # consumer thread has not started
                pass

    def shutdown(self):
        """Flush all messages and cleanly shutdown the client"""
        self.flush()
        self.join()


def require(name, field, data_type):
    """Require that the named `field` has the right `data_type`"""
    if not isinstance(field, data_type):
        msg = '{0} must have {1}, got: {2}'.format(name, data_type, field)
        raise AssertionError(msg)


def stringify_id(val):
    if val is None:
        return None
    if isinstance(val, string_types):
        return val
    return str(val)
