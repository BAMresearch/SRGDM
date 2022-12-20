import numpy as np
from ...common import LatticeScalar


def getErrorLattice(lattice_1, lattice_2):
	assert (isinstance(lattice_1, LatticeScalar))
	assert (isinstance(lattice_2, LatticeScalar))
	assert (lattice_1.shape == lattice_2.shape), str(lattice_1.shape) + "!=" + str(lattice_2.shape)

	error_matrix = lattice_2.toMatrix() - lattice_1.toMatrix()
	return LatticeScalar.fromMatrix(error_matrix)



def getError(lattice_1, lattice_2):
	error_matrix = getErrorLattice(lattice_1, lattice_2).toMatrix()
	error = error_matrix.sum()
	return error



def getDistance(lattice_1, lattice_2):
	error_matrix = getErrorLattice(lattice_1, lattice_2).toMatrix()
	distance = np.sqrt(np.multiply(error_matrix,error_matrix).sum())
	return distance



def getMSE(lattice_1, lattice_2):
	error_matrix = getErrorLattice(lattice_1, lattice_2).toMatrix()
	se = np.multiply(error_matrix,error_matrix).sum()
	n = lattice_1.shape[0]*lattice_1.shape[1]
	mse = se/n
	return mse



def getRMSE(lattice_1, lattice_2):
	mse = getMSE(lattice_1, lattice_2)
	rsme = np.sqrt(mse)
	return rsme


def testMetric(lattice_1, lattice_2, sigma):
	error = getErrorLattice(lattice_1, lattice_2).toMatrix().flatten()
	return (error * error * sigma.toMatrix().flatten()).sum()