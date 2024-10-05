def skip(iterable, n: int = 0):
    """Skippable iterator"""
    counter = 0
    for item in iterable:
        if counter < n:
            counter += 1
            continue
        counter = 0
        yield item


from collections.abc import Iterable


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
