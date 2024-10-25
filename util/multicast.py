# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import Generic, TypeVar, Literal, overload, Any
from threading import Lock
from weakref import WeakSet, ref
from io import IOBase

from .logger import Logger
from .queue import Queue
from .exception import Expect

log = Logger(__name__)

C = TypeVar("C")  # Type of stream object
T = TypeVar("T")  # Type of stream payload
EP = TypeVar("EP", bound="EndPoint[C, T]")


class EndPoint(Generic[C, T]):
    def ref(self) -> C:
        """
        Access internally stored endpoint object.
        By default, returns endpoint itself.
        i.e. C == EndPoint[C, T]
        """
        return self

    def send(self, item: T) -> bool | None:
        """
        Send an item to the endpoint.
        Returns False if the endpoint is closed.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Close the endpoint if supported.
        """
        pass


class WeakEndPoint(EndPoint[ref[C], T]):
    def __init__(self, ep: C):
        self.ref = ref(ep)

    def send(self, item: T) -> bool:
        pipe = self.ref()
        if pipe is not None:
            return self._send(pipe, item)

    def close(self) -> None:
        pipe = self.ref()
        if pipe is not None:
            self._close(pipe)

    @staticmethod
    def _send(ep: C, item: T) -> bool:
        """
        Endpoint is still alive, send the item.
        """
        raise NotImplementedError

    @staticmethod
    def _close(ep: C) -> None:
        """
        Close the endpoint.
        """
        pass


class QueueEndPoint(WeakEndPoint[Queue[T], T]):
    @staticmethod
    def _send(ep, item):
        try:
            ep.put(item)
        except ep.Closed:
            return False
        return True

    @staticmethod
    def _close(ep):
        ep.close()


class FileEndPoint(EndPoint[IOBase, str]):
    def __init__(self, ep: IOBase):
        self.ref = lambda: ep

        @Expect(OSError)
        def send(item: str):
            if ep.writable() and ep.fileno() >= 0:
                ep.write(item)
                return True

        self.send = send

        @Expect(OSError)
        def close():
            ep.close()

        self.close = close

class Multicast(EndPoint["Multicast[T]", T]):
    @overload
    @classmethod
    def pipe() -> Queue[T]: ...

    @overload
    @classmethod
    def pipe(item: EndPoint[C, T]) -> C: ...

    @overload
    @classmethod
    def pipe(item: C) -> C: ...

    def __init__(self):
        lock = Lock()
        closed: bool = False
        endpoints: set[EndPoint[Any, T]] = set()

        def pipe(item: EndPoint | None = None, **kwargs):
            """
            Add a new endpoint (Queue | IOBase) as a multicast endpoint.
            Create a new queue using args and kwargs if not provided.
            Returns the newly inserted queue.
            """
            nonlocal lock, closed, endpoints
            ep = None
            if item is None:
                item: Queue[T] = Queue(**kwargs)
            else:
                assert len(kwargs) == 0, kwargs
            if isinstance(item, EndPoint):
                ep, item = item, item.ref()
            elif isinstance(item, Queue):
                ep = QueueEndPoint(item)
            elif isinstance(item, IOBase):
                ep = FileEndPoint(item)
            with lock:
                if closed:
                    raise RuntimeError(f"Multicast endpoint is closed")
                endpoints.add(ep)
            return item

        def send(item: T):
            """
            Send an item to all endpoints.
            Always returns true unless closed.
            """
            nonlocal lock, closed, endpoints
            inactive = set()
            with lock:
                if closed:
                    return False
                snapshot = endpoints.copy()
            for ep in snapshot:
                if not ep.send(item):
                    inactive.add(ep)
            with lock:
                endpoints.difference_update(inactive)
            return True

        def close():
            """
            Close all endpoints.
            """
            nonlocal lock, closed, endpoints
            with lock:
                closed = True
                snapshot = endpoints.copy()
                endpoints.clear()
            for ep in snapshot:
                ep.close()

        self.pipe = pipe
        self.send = send
        self.close = close
