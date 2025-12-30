from algorithms.data_structures.basic.stack import Stack
from algorithms.data_structures.basic.queue import Queue
from algorithms.data_structures.basic.linked_list import LinkedList


def test_stack_operations():
    stack = Stack()
    assert stack.is_empty()
    stack.push(1)
    stack.push(2)
    assert stack.peek() == 2
    assert stack.pop() == 2
    assert stack.pop() == 1
    assert stack.pop() is None
    assert stack.is_empty()


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


def test_linked_list_operations():
    linked = LinkedList()
    assert linked.find(1) is None
    assert not linked.delete(1)
    linked.append(1)
    linked.append(2)
    linked.append(3)
    node = linked.find(2)
    assert node is not None and node.value == 2
    assert linked.delete(2)
    assert linked.find(2) is None
    assert not linked.delete(4)
    assert linked.to_list() == [1, 3]
