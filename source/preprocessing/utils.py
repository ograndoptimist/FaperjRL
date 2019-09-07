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
	

def write_vocabulary(path, vocabulary):
    with open(path, 'w') as file:
        if isinstance(vocabulary, dict):
            file.write(str(vocabulary))
        else:
            for word in vocabulary:
                file.write(str(word) + "\n")
