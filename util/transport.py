# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from warnings import warn
from typing import Generic, Literal, TypeVar, Generator, Iterable, overload
from threading import Thread, Lock

from .logger import Logger
from .queue import Queue
from .multicast import Multicast, EndPoint
from .exception import Expect

log = Logger(__name__)

I = TypeVar("I")
O = TypeVar("O")
T = TypeVar("T")
C = TypeVar("C")

__names__ = dict[str, int]()
__name_lock__ = Lock()


def unique_name(name: str):
    with __name_lock__:
        if name in __names__:
            __names__[name] += 1
        else:
            __names__[name] = 0
        return f"{name}-{__names__[name]}"


def member_of(name: str):
    def wrapper(fn: callable):
        fn.__name__ = f"{name}::{fn.__name__}"
        return fn

    return wrapper


class Transport(EndPoint["Transport[I, O]", I], Generic[I, O]):
    """
    A transport is a generic data processing pipeline that reads from an input
    queue, perform transform to each item and then writes the output to its
    receivers. Implements single-input, multiple-output data processing model.
    ---
    Abstract methods to be implemented:
    - init(self) -> None (optional)
    - spin(self) -> Generator[O, None, None] (optional)
    - transform(self, arg: I) -> Generator[O, None, None] (required)
    ---
    Exposed methods for external interaction:
    - start(self, mode: Literal["thread", "coroutine"] = "thread", rate_limit: float = None) -> self
    - send(self, item: I) -> bool
    - stop(self) -> None
    """

    def init(self):
        """
        Optional hook to perform additional initialization at the start
        of the loop thread.
        """
        pass

    def spin(self) -> Generator[O, None, None] | None:
        """
        Optional hook to perform task when no input data is available.
        This method allows the transport to perform additional task that
        does not correspond to any input data.
        """
        yield from []

    def transform(self, arg: I) -> Generator[O, None, None]:
        """
        Optional hook to perform transfor given a input data point.
        Each yielded item will be sent to output multicast.
        """
        yield arg

    def __init__(self, input: Queue[I] = None, **kwargs):
        """
        Create a transform reading from input queue and writing to output queue
        :param i: Input queue
        :param o: Output queue
        """
        super().__init__()
        SELF = unique_name(self.__class__.__name__)
        context_lock = Lock()
        output = Multicast[O]()
        self.pipe = output.pipe
        self.close = output.close

        for k, v in kwargs.items():
            setattr(self, k, v)

        class Nothing(Queue.Empty, Exception):
            pass

        @Queue.Loop()
        @member_of(SELF)
        def loop(send: callable) -> Generator[None, I | type[Nothing], None]:
            log.debug(f"{SELF} initializing")
            self.init()
            log.debug(f"{SELF} started")
            while True:
                item = yield
                try:
                    send(self.spin())
                except Exception as e:
                    log.error(f"{SELF}::spin: {e}")
                if item is Nothing:
                    continue
                try:
                    send(self.transform(item))
                except Exception as e:
                    log.error(f"{SELF}::transform: {e}")

        @Expect(StopIteration, Queue.Closed)
        @member_of(SELF)
        def run(
            task: Generator[None, I | type[Nothing], None], rate_limit: float = None
        ):
            if rate_limit is not None:
                assert isinstance(rate_limit, int | float)
                assert rate_limit > 0
                interval = 1.0 / rate_limit
            else:
                interval = None
                warn("overriding rate_limit might clog other threads")

            next(task)

            def recv():
                if queue is None:
                    raise Queue.Closed
                try:
                    return queue.get(block=False)
                except Queue.Empty:
                    return Nothing

            if interval is None:
                while True:
                    task.send(recv())
            else:
                from .iter import intervals

                for _ in intervals(interval):
                    task.send(recv())

        # Mode 1: Thread with an input queue
        queue: Queue[I] = None
        thread: Thread = None

        def recv_thread(results: Iterable[O] | None):
            if results:
                for item in results:
                    output.send(item)

        # Mode 2: Coroutine (generator) that reads from yield
        coroutine: Generator[None, None, None] = None
        deferred_outputs_lock = Lock()
        deferred_outputs = list[O]()

        def recv_coroutine(results: Iterable[O] | None):
            if results:
                deferred_outputs.extend(results)

        @member_of(SELF)
        def send(item: I | type[Nothing] = Nothing):
            nonlocal queue, coroutine, deferred_outputs
            success: bool = True  # likely
            snapshot = None
            with context_lock:
                if coroutine is not None:
                    with deferred_outputs_lock:
                        try:
                            if item is Nothing and input and input(coroutine.send):
                                pass
                            else:
                                coroutine.send(item)
                        except StopIteration:
                            coroutine = None
                            success = False
                        nonlocal deferred_outputs
                        deferred_outputs, snapshot = [], deferred_outputs
                elif queue is not None:
                    try:
                        if item is not Nothing:
                            queue.put(item)
                    except Queue.Closed:
                        success = False
                else:
                    log.warn(f"{SELF}::send: transport not running")
            if snapshot is not None:
                for item in snapshot:
                    output.send(item)
            return success

        self.send = send

        @member_of(SELF)
        def start(
            mode: Literal["thread", "coroutine"] = "thread", rate_limit: float = Nothing
        ) -> "Transport[I, O]":
            nonlocal self, thread, queue, coroutine
            with context_lock:
                match mode.lower():
                    case "thread":
                        assert coroutine is None, "Already running as coroutine"
                        if thread is None:
                            thread = Thread(
                                name=self.__class__.__name__,
                                target=run,
                                kwargs=dict(
                                    task=loop(recv_thread),
                                    rate_limit=(
                                        1e3 if rate_limit is Nothing else rate_limit
                                    ),
                                ),
                                daemon=True,
                            )
                            assert queue is None, "Queue already exists"
                            queue = Queue() if input is None else input
                        if not thread.is_alive():
                            thread.start()
                    case "coroutine":
                        assert thread is None, "Already running as thread"
                        if rate_limit is not None and rate_limit is not Nothing:
                            warn("rate_limit not supported in coroutine mode")
                        if coroutine is None:
                            coroutine = loop(recv_coroutine)
                            next(coroutine)
                    case _:
                        raise ValueError(f"Invalid mode: {mode}")
                return self

        self.start = start

        @member_of(SELF)
        def is_alive():
            with context_lock:
                if thread is not None and thread.is_alive():
                    return True
                if coroutine is not None:
                    return True
                return False

        self.is_alive = is_alive

        @member_of(SELF)
        def stop():
            nonlocal output, thread, queue, coroutine
            with context_lock:
                if thread is not None:
                    if queue is not None:
                        queue.close()
                    thread.join()
                    thread = None
                    queue = None
                if coroutine is not None:
                    coroutine = None
            # Close output after thread is stopped
            output.close()

        self.stop = stop

    @overload
    def __call__(self: "Transport[I, O]", **kw) -> Queue[O]: ...

    @overload
    def __call__(self: "Transport[I, O]", item: EndPoint[C, T]) -> C: ...

    @overload
    def __call__(self: "Transport[I, O]", item: T) -> T: ...

    def __call__(self, q: Queue[O] = None, *args, **kwargs):
        return self.pipe(q, *args, **kwargs)

    def __del__(self):
        self.stop()

    def __enter__(self):
        assert self.is_alive(), "[usage] with transport.start(): ..."
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return None


class Pool:
    class __Pool__(set[Transport]):
        mode: str
        kwargs: dict

        def __init__(self, *args: Transport, **kwargs):
            super().__init__()
            for tp in args:
                self.add(tp)
            self.kwargs = kwargs

        def __enter__(self):
            for p in self:
                p.start(mode=self.mode, **self.kwargs)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            for p in self:
                p.stop()
            return None

    class Thread(__Pool__):
        mode = "thread"

        def __init__(self, *args: Transport, rate_limit: float = 1000):
            super().__init__(*args, rate_limit=rate_limit)

        def loop(self, rate_limit: float = None):
            from .iter import intervals

            if rate_limit is not None:
                assert rate_limit > 0
                interval = 1.0 / rate_limit
                yield from intervals(interval)
            else:
                while True:
                    yield

    class Coroutine(__Pool__):
        mode = "coroutine"

        def __init__(self, *args: Transport, rate_limit: float = None):
            super().__init__(*args)

        def __enter__(self):
            for p in self:
                p.start(mode="coroutine")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            for p in self:
                p.stop()
            return None

        def tick(self):
            for p in self:
                p.send()

        def loop(self, rate_limit=2, auto_spin: bool = True):
            from time import time as now
            from .iter import intervals

            def sleep(duration: float = 0.0):
                deadline = now() + duration
                self.tick()
                while now() < deadline:
                    self.tick()

            if rate_limit is not None:
                assert rate_limit > 0
                interval = 1.0 / rate_limit
                yield from intervals(interval, sleep=sleep if auto_spin else None)
            else:
                while True:
                    if auto_spin:
                        self.tick()
                    yield
