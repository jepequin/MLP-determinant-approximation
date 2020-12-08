from numpy import sqrt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from random_matrix import generator

### Select number of samples, matrix size and range of entries in matrices
nb_samples = 1000
matrix_size = 2
entries_range = [0, 100]

### Generate random matrices and determinants
matrices, determinants = generator(nb_samples, matrix_size, entries_range)

### Select number of layers and neurons
nb_layers = 1
nb_neurons = 16

### Create dense neural network with nb_layers hidden layers having nb_neurons neurons each
model = Sequential()
model.add(Dense(nb_neurons, input_dim = matrix_size**2, activation=lambda x:x**matrix_size))
for i in range(nb_layers-1):
	model.add(Dense(nb_neurons)) 
model.add(Dense(1))
### Add metrics=['accuracy'] to save 'accuracy' in history object
model.compile(loss='mse', optimizer='adam')

### Train and save model using train size of 0.66
history = model.fit(matrices, determinants, epochs = 400, batch_size = 100, verbose = 0, validation_split = 0.33)

### Get validation loss from object 'history' 
### Print RMSE and parameter values
print('''
Validation RMSE: {}
Number of layers: {}
Number of neurons: {}
Number of samples: {}
'''.format(sqrt(history.history['val_loss'][-1]),nb_layers,nb_neurons,nb_samples))
