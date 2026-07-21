from __future__ import annotations

from types import SimpleNamespace

from nvmeof import protocol as p
from nvmeof.controller import Namespace


class Buffer:
    addr = 0x100000
    length = 12 * 4096
    rkey = 123


class Queue:
    def __init__(self):
        self.commands = []

    def execute(self, command, **kwargs):
        self.commands.append(command)
        return p.Completion(0, 0, 1, 0, 1)


def test_read_splits_at_controller_transfer_limit():
    queue = Queue()
    controller = SimpleNamespace(io=queue, max_transfer_bytes=4 * 4096, timeout=1)
    info = p.NamespaceInfo(1, 100, 100, 0, 4096, 0)
    namespace = Namespace(controller, info)
    namespace.read(Buffer(), slba=10, blocks=10)
    assert len(queue.commands) == 3
    assert [struct_nlb(command) for command in queue.commands] == [4, 4, 2]
    assert [struct_slba(command) for command in queue.commands] == [10, 14, 18]


def struct_nlb(command):
    return int.from_bytes(command[48:50], "little") + 1


def struct_slba(command):
    return int.from_bytes(command[40:48], "little")
