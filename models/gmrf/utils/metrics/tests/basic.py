import numpy as np
from gdm.common import Lattice2DScalar, DiscreteScalarMap
from gdm.utils.metrics.basic import *

matrix_1 = np.ones((3,3))
matrix_2 = np.array(((1,2,3),(4,5,6),(7,8,9)))


lattice_1 = Lattice2DScalar.fromMatrix(matrix_1)
lattice_2 = Lattice2DScalar.fromMatrix(matrix_2)

error_lattice_12 = getErrorLattice(lattice_1, lattice_2)
error_12         = getError(lattice_1, lattice_2)
distance_12      = getDistance(lattice_1, lattice_2)
mse_12           = getMSE(lattice_1, lattice_2)
rmse_12          = getRMSE(lattice_1, lattice_2)

print(error_lattice_12.toMatrix())
print(error_12)
print(distance_12)
print(mse_12)
print(rmse_12)

assert(not (error_lattice_12.toMatrix()-(matrix_2-matrix_1)).all())
assert(error_12 == 36)
assert(np.isclose(distance_12, 14.29, atol=1e-01))
assert(np.isclose(mse_12, 22.66, atol=1e-01))
assert(np.isclose(rmse_12, 4.76, atol=1e-01))