from ....gdm.utils.vector import *

a = Vector2D((1,1))
b = Vector2D((2,2))
c = Vector2D(a+b)
d = Vector2D((3,1), int)


print(a+b)
print(-a)
print(a-b)
print(a.getModule())
print(a.toArray())
print(a.toTuple())
print(a.isPositive())
print(a==a)
print(a==b)
print(c)
print(d)
print(d.getNormalized())
print(a.getNormalized())
print(a.asType(int))

a = CellCoordinates2D((1,1))
a.x = 2
print(a.i)