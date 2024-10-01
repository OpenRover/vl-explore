def skip(iterable, n: int = 0):
    """Skippable iterator"""
    counter = 0
    for item in iterable:
        if counter < n:
            counter += 1
            continue
        counter = 0
        yield item
