import numpy as np
from gdm.utils.metrics import getKLD
from gdm.common import Lattice2DScalar, Lattice2DScalarDistribution



matrix_1m = np.ones((3,3))
matrix_1s = np.ones((3,3))
matrix_2m = np.ones((3,3))
matrix_2s = np.ones((3,3))

p = Lattice2DScalarDistribution.fromMatrix(matrix_1m, matrix_1s)
q = Lattice2DScalarDistribution.fromMatrix(matrix_2m, matrix_2s)

kld = getKLD(p, q)
print(kld)
assert(np.isclose(kld, 0, atol=1e-01))



matrix_1m = np.ones((3,3))
matrix_1s = np.ones((3,3))
matrix_2m = np.ones((3,3))*2
matrix_2s = np.ones((3,3))*2

p = Lattice2DScalarDistribution.fromMatrix(matrix_1m, matrix_1s)
q = Lattice2DScalarDistribution.fromMatrix(matrix_2m, matrix_2s)

kld = getKLD(p, q)
print(kld)
assert(np.isclose(kld, 0.5*(4.5+4.5-9+6.23), atol=1e-01))