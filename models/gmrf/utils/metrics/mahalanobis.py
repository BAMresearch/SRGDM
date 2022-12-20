import numpy as np
from gdm.common import LatticeScalar, MultivariateNormalPDFMap
from .basic import getErrorLattice


def getMahalanobisDistance(distribution_lattice, point_lattice):

	assert (isinstance(point_lattice, LatticeScalar))
	assert (isinstance(distribution_lattice, MultivariateNormalPDFMap))

	error = (point_lattice.toMatrix() - distribution_lattice.getMean().toMatrix()).flatten()
	variance_matrix = distribution_lattice.getVariance().toMatrix().flatten() + 0.000001

	d2 = np.multiply(error, error)
	m2 = np.divide(d2, variance_matrix)
	m =  np.sqrt(m2.sum())

	return m