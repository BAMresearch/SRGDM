import numpy as np
from gdm.utils.metrics import getNLML
from gdm.common import Lattice2DScalar, MultivariateNormalPDFMap


def test_nlml(matrix_ref, matrix_mean, matrix_var, expected_nlml_value):

	lattice = Lattice2DScalar.fromMatrix(matrix_ref)
	mean = Lattice2DScalar.fromMatrix(matrix_mean)
	var = Lattice2DScalar.fromMatrix(matrix_var)
	distribution = MultivariateNormalPDFMap(mean, var)

	nlml = getNLML(distribution, lattice)
	print(nlml)
	assert (np.isclose(nlml, expected_nlml_value, atol=1e-02))




################################################################################
##  VARIED
################################################################################

matrix_1 = np.ones((3,3))
matrix_2 = np.ones((3,3))
matrix_3 = np.ones((3,3))
test_nlml(matrix_1, matrix_2, matrix_3, 16.54)



matrix_1 = np.ones((3,3))
matrix_2 = np.ones((3,3))*2
matrix_3 = np.ones((3,3))
test_nlml(matrix_1, matrix_2, matrix_3, 25.54)



matrix_1 = np.ones((3,3))
matrix_2 = np.ones((3,3))*2
matrix_3 = np.ones((3,3))*0.5
test_nlml(matrix_1, matrix_2, matrix_3, 28.31)



matrix_1 = np.ones((3,3))
matrix_2 = np.array(((1,2,3),(4,5,6),(7,8,9)))
matrix_3 = np.array(((1,1,1),(2,2,2),(3,3,3)))
test_nlml(matrix_1, matrix_2, matrix_3, 101.582)



################################################################################
##  1D
################################################################################

matrix_1 = np.expand_dims(np.ones((1))*0, axis=0)
matrix_2 = np.expand_dims(np.ones((1))*0, axis=0)
matrix_3 = np.expand_dims(np.ones((1))*1, axis=0)
test_nlml(matrix_1, matrix_2, matrix_3, 1.837)



matrix_1 = np.expand_dims(np.ones((1))*1, axis=0)
matrix_2 = np.expand_dims(np.ones((1))*0, axis=0)
matrix_3 = np.expand_dims(np.ones((1))*1, axis=0)
test_nlml(matrix_1, matrix_2, matrix_3, 2.838)



matrix_1 = np.expand_dims(np.ones((1))*0, axis=0)
matrix_2 = np.expand_dims(np.ones((1))*0, axis=0)
matrix_3 = np.expand_dims(np.ones((1))*0.25, axis=0)
test_nlml(matrix_1, matrix_2, matrix_3, 0.4515)



matrix_1 = np.expand_dims(np.ones((1))*0.5, axis=0)
matrix_2 = np.expand_dims(np.ones((1))*0, axis=0)
matrix_3 = np.expand_dims(np.ones((1))*0.25, axis=0)
test_nlml(matrix_1, matrix_2, matrix_3, 1.4515)



################################################################################
##  2D
################################################################################

matrix_1 = np.ones((2,1))*0
matrix_2 = np.ones((2,1))*0
matrix_3 = np.ones((2,1))
test_nlml(matrix_1, matrix_2, matrix_3, 3.675)


matrix_1 = np.ones((2,1))*0
matrix_1[0,0] = 1
matrix_2 = np.ones((2,1))*0
matrix_3 = np.ones((2,1))
test_nlml(matrix_1, matrix_2, matrix_3, 4.675)


matrix_1 = np.ones((2,1))
matrix_2 = np.ones((2,1))*0
matrix_3 = np.ones((2,1))
test_nlml(matrix_1, matrix_2, matrix_3, 5.675) # 2.838 * 2 = 2*NLML for 1D at 1 sigma distance