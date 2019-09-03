"""
  Utilities's functions.
"""

def duck_type(arr, choice):
  """
    Checks if it's a list or a list of lists
    using duck typing.
  """
	try:
		len(arr[0])
		return arr[choice]
	except TypeError:
		return arr
