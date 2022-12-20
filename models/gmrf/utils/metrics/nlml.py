from gdm.common import LatticeScalar, MultivariateNormalPDFMap
import numpy as np


def getNLML(normal_distribution, ground_truth):

	assert (isinstance(normal_distribution, MultivariateNormalPDFMap))
	assert (isinstance(ground_truth, LatticeScalar))

	error     = (normal_distribution.getMean().toMatrix() - ground_truth.toMatrix()).flatten()
	variance  = normal_distribution.getVariance().toMatrix().flatten() + 0.0000001
	assert error.shape == variance.shape
	d         = variance.shape[0]

	k   = d * np.log(2 * np.pi)
	smd = (error * error / variance).sum()
	log = np.log(variance).sum()

	nlml = k + smd + log
	return nlml
