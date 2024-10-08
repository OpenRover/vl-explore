def sign(value: float) -> int:
  """Return the sign of a value"""
  if value > 0:
    return 1
  elif value < 0:
    return -1
  else:
    return 0

def clamp(min: float, max: float) -> float:
  """lambda function to clamp a value between min and max"""
  assert min <= max
  def fn(value: float) -> float:
    return max if value > max else min if value < min else value
  return fn

def ang_diff(src: float, dst: float, half_period=180.0, direction=None) -> float:
  """
  Check the minimum angular distance between two angles.
  Positive if dst is clockwise from src, negative otherwise.
  """
  period = 2 * half_period
  diff = dst % period - src % period
  if diff > half_period:
    diff -= 2 * half_period
  elif diff < -half_period:
    diff += 2 * half_period
  if direction is not None and direction != 0:
    if diff < 0 and direction > 0:
      diff += period
    elif diff > 0 and direction < 0:
      diff -= period
  return diff
