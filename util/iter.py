from typing import TypeVar, Iterable

T = TypeVar("T")

def first_item(iterator: Iterable[T]) -> T:
    for item in iterator:
        return item
    raise AssertionError("Empty iterator")


def last_item(iterator: Iterable[T]) -> T:
    for item in iterator:
        pass
    assert "item" in locals(), "Empty iterator"
    return item


def skip(iterable, n: int = 0):
    """Skippable iterator"""
    counter = 0
    for item in iterable:
        if counter < n:
            counter += 1
            continue
        counter = 0
        yield item


def flatten(iterable: Iterable, depth: int = None):
    """Flatten nested iterables"""
    if depth is not None and depth == 0:
        yield iterable
        return
    elif depth is not None:
        depth -= 1
    for item in iterable:
        # Basic types are not iterable
        if isinstance(item, (str, bytes)):
            yield item
        # Check if item is iterable
        elif isinstance(item, Iterable):
            yield from flatten(item, depth)
        else:
            yield item

def intervals(interval: float, sleep: callable = None):
    assert isinstance(interval, int | float), type(interval)
    assert interval > 0, interval
    from time import time as now, sleep as sys_sleep
    sleep = sys_sleep if sleep is None else sleep
    t0 = now()
    while True:
        yield
        t0 += interval
        t1 = now()
        if t1 < t0:
            sleep(t0 - t1)
