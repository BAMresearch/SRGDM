## This function is from a third party
## Find author and cite

import sys

def getInstanceSize(obj, seen=None):
	"""Recursively finds size of objects"""
	size = sys.getsizeof(obj)
	if seen is None:
		seen = set()
	obj_id = id(obj)
	if obj_id in seen:
		return 0
	# Important mark as seen *before* entering recursion to gracefully handle
	# self-referential objects
	seen.add(obj_id)
	if isinstance(obj, dict):
		size += sum([getInstanceSize(v, seen) for v in obj.values()])
		size += sum([getInstanceSize(k, seen) for k in obj.keys()])
	elif hasattr(obj, '__dict__'):
		size += getInstanceSize(obj.__dict__, seen)
	elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
		size += sum([getInstanceSize(i, seen) for i in obj])
	return size


