"""
  Utilities's functions.
"""

def duck_type(arr, index):
    """
      Checks if it's a list or a list of lists
      using duck typing.
    """
    try:
	len(arr[0])
	return arr[index]
    except TypeError:
	return arr
	

def save_content(path, content):
    with open(path, 'w') as file:
        if isinstance(content, dict):
            file.write(str(content))
        else:
            for item in content:
                file.write(str(item) + "\n")
		

def list_to_dictionary(list_):
    """
    	Builds a dictionary based on a list.
    """
    return {v: k for k, v in enumerate(list_)}
