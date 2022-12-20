import numpy as np
from copy import copy


class Vector:

	def __init__(self):

		self._self_check()

		## Enforce child attributes
		assert hasattr(self, '__str__')
		assert hasattr(self, '__add__')
		assert hasattr(self, '__neg__')
		assert hasattr(self, '__sub__')
		assert hasattr(self, '__eq__')
		#assert hasattr(self, '__div__')
		assert hasattr(self, 'getModule')
		assert hasattr(self, 'toArray')
		assert hasattr(self, 'toTuple')
		assert hasattr(self, 'isPositive')
		assert hasattr(self, 'isStrictlyPositive')
		assert hasattr(self, 'getNormalized')
		# assert hasattr(self, 'rotate')
		assert hasattr(self, '_self_check')

		return



class Vector2D(Vector):

	def __init__(self, v, data_type=float):

		## Process parameters
		if type(v) is type(self):
			x = v.x
			y = v.y
		elif type(v) is tuple and len(v) == 2:
			x = data_type(v[0])
			y = data_type(v[1])
		else:
			assert False


		## Member variables
		self.x = x
		self.y = y
		self._data_type = data_type


		## Init bases
		Vector.__init__(self)

		return



	## BASE CLASS IMPLEMENTATION -----------------------------------------------

	def __str__(self):
		return "("+str(self.x)+","+str(self.y)+")"



	def __add__(self, other):
		assert type(other) is type(self)
		x = self.x + other.x
		y = self.x + other.y
		return type(self)((x,y), self._data_type)



	def __neg__(self):
		return type(self)((-self.x, -self.y), self._data_type)



	def __sub__(self, other):
		assert type(other) is type(self)
		return self.__add__(other.__neg__())



	def __eq__(self, other):
		assert type(other) is type(self)
		return self.x == other.x and self.y == other.y



	def __truediv__(self,other):
		if type(other) is type(self):
			assert False, "Not implemented yet"
		elif self._data_type is float:
			x = self.x / other
			y = self.y / other
			assert type(x) is float and type(y) is float
			return Vector2D((x,y), float)
		else:
			assert False



	def toArray(self):
		return np.array((self.x, self.y))



	def toTuple(self):
		return (self.x, self.y)



	def isPositive(self):
		return self.x >= 0 and self.y >= 0



	def isStrictlyPositive(self):
		return self.x > 0 and self.y > 0


	def _self_check(self):
		assert type(self.x) is self._data_type
		assert type(self.y) is self._data_type



	def getModule(self):
		return float(np.sqrt(self.x**2 + self.y**2))

	"""
	def rotate(self, angle):
		self.x = self.x*np.cos(angle) - self.y*np.sin(angle)
		self.y = self.x*np.sin(angle) + self.y*np.cos(angle)
		return self
	"""

	def getNormalized(self):
		return Vector2D((self.x,self.y), float) / self.getModule()


	def asType(self, data_type):
		return Vector2D((self.x, self.y), data_type)




##==============================================================================
class Position2D(Vector2D):

	def __init__(self, position):
		assert type(position) is tuple
		Vector2D.__init__(self, v=position, data_type=float)
		return




##==============================================================================
class CellCoordinates2D(Vector2D):
	def __init__(self, cell):
		assert type(cell) is tuple
		assert type(cell[0]) is int and type(cell[1]) is int
		Vector2D.__init__(self, v=cell, data_type=int)
		assert self.isPositive()

		self.i = copy(self.x)
		self.j = copy(self.y)

		return















