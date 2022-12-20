import numpy as np
from gdm.common import Lattice2DScalar, MultivariateNormalPDFMap
from gdm.utils.metrics.mahalanobis import *

matrix_1 = np.ones((3,3))
matrix_2 = np.array(((1,2,3),(4,5,6),(7,8,9)))
matrix_3 = np.array(((1,1,1),(2,2,2),(3,3,3)))

lattice = Lattice2DScalar.fromMatrix(matrix_1)
mean = Lattice2DScalar.fromMatrix(matrix_2)
var = Lattice2DScalar.fromMatrix(matrix_3)
distribution = MultivariateNormalPDFMap(mean, var)

m = getMahalanobisDistance(distribution, lattice)
print(m)
assert(np.isclose(m, 8.925, atol=1e-01))