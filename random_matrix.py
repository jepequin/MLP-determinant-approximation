import numpy as np

def generator(nb_samples,matrix_size,entries_range):
	### Generate nb_samples random matrices of size matrix_size with entries in range [0,entries_range) 
	matrices = []
	determinants = []
	for i in range(nb_samples):
		matrix = np.random.randint(entries_range[0], high = entries_range[1], size = (matrix_size,matrix_size))
		matrices.append(matrix.reshape(matrix_size**2,))
		determinants.append(np.array(np.linalg.det(matrix)).reshape(1,))
	return np.array(matrices), np.array(determinants)

