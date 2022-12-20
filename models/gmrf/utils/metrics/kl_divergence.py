import numpy as np
from ...common.probability_dist import MultivariateNormalPDF


def getKLD(p, q):
	assert (isinstance(p, MultivariateNormalPDF))
	assert (isinstance(q, MultivariateNormalPDF))
	assert (p.shape == q.shape)
	assert (p.resolution == q.resolution)

	p_mean     = p.getMean().toMatrix().flatten()
	q_mean     = q.getMean().toMatrix().flatten()
	p_variance = p.getVariance().toMatrix().flatten()
	q_variance = q.getVariance().toMatrix().flatten()

	trace = (p_variance / q_variance).sum()
	error = q_mean - p_mean
	smd_q = ((error ** 2) / q_variance).sum()
	k     = q_variance.shape[0]
	log   = (np.log(q_variance)).sum() - (np.log(p_variance)).sum()

	kld = 0.5 * (trace + smd_q - k + log)
	return kld