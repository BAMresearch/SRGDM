from gdm.common import LatticeScalar, MultivariateNormalPDFMap
import numpy as np


def getDNLML(normal_distribution, ground_truth,mask):

	assert (isinstance(normal_distribution, MultivariateNormalPDFMap))
	assert (isinstance(ground_truth, LatticeScalar))

	## SAME AS NLML
	error     = (normal_distribution.getMean().toMatrix() - ground_truth.toMatrix()).flatten()
	variance  = normal_distribution.getVariance().toMatrix().flatten() + 0.0000001
	assert error.shape == variance.shape

	## PURGE ZERO ERROR
	mask = mask.flatten()
	derror = error[mask]
	dvariance = variance[mask]
	d         = dvariance.shape[0]
	assert derror.shape == dvariance.shape

	k   = d * np.log(2 * np.pi)
	smd = (derror * derror / dvariance).sum()
	log = np.log(dvariance).sum()  # log(det(DIAGONAL_MATRIX)) = sum(log(DIAGONAL MATRIX)). Sigma shoudl be diagonal, but i dont bother transofrming it because its not necessary

	dnlml = k + smd + log
	return dnlml



def getDNLML2(mean, ground_truth):
	assert (isinstance(mean, LatticeScalar))
	assert (isinstance(ground_truth, LatticeScalar))

	## SAME AS NLML
	error = (mean.toMatrix() - ground_truth.toMatrix()).flatten()

	inc = 0.1
	range = np.arange(0,1,inc)
	gt = ground_truth.toMatrix().flatten()
	w = np.zeros(gt.shape)
	sum = 0
	for i in range:
		s = ((i <= gt) & (gt < i+inc)).sum()
		w[(i <= gt) & (gt < i+inc)] = s
		sum += s
	w = (w.shape[0]-w)/(w.shape[0])

	return (error * error * w).sum()


