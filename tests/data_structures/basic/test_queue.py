import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from algorithms.data_structures.basic.queue import Queue


def test_queue_operations():
    queue = Queue()
    assert queue.is_empty()

    queue.enqueue(1)
    queue.enqueue(2)
    assert queue.peek() == 1
    assert queue.dequeue() == 1
    assert queue.dequeue() == 2
    assert queue.dequeue() is None
    assert queue.is_empty()
