import billiard
import scipy
import numpy as np


p_solver = 0
p_num_variables = 0
		

def invOfColumn(col_num):
	global p_solver
	global p_num_variables
	
	e_c = (scipy.sparse.csc_matrix(([1],([col_num],[0])), shape=(p_num_variables,1))).toarray()
	return p_solver(e_c)[col_num]
	
	
def getDiagonalOfInverse(matrix, stop_at=0):

	global p_solver
	global p_num_variables

	if __name__ == '__main__' or __name__=='gdm.gmrf.pool_inverse':
		p_solver = scipy.sparse.linalg.factorized(matrix)
		p_num_variables = matrix.shape[0]
		
		if stop_at == 0:
			stop_at = p_num_variables
		
		with billiard.Pool(4) as pool:
			diagonal = pool.map(invOfColumn, [i for i in range(0,stop_at) ])
			pool.close()
			pool.terminate()
			return np.array(diagonal)
	else:
		return 0
	

"""
m = np.array(((1,0,0,0),(0,2,0,0),(0,0,3,0),(0,0,0,4)))
d = getDiagonalOfInverse(m)
print(d)
"""