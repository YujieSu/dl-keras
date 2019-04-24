# Model specification 
# how many layers? how many nodes? what activation function do you want to use 
# in each layer?

import numpy as np
from keras.layers import Dense
from keras.models import Sequential

# read the data and find the number of nodes in the input layer
predictors = np.loadtxt('predictors_data.csv', delimiter = ',')
n_cols = predictors.shape[1] # nodes in the input layer (how many columns)
# building the model
model = Sequential()

# add layers use the add method of the model
model.add(Dense(100, activation = 'relu', input_shape = (n_cols,)))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(1))
# Dense: standard layer type, 100 - number of nodes
# in the first layer, we need to specify input shapes, means input will have 
# n_col columns
# The last layer is the output layer, only has one node.
# This model has two hidden layers and an output layer.

# Compiling and fitting a model
# Set up the network for optimization (like internal function to do 
# back-propagation efficiently)
## specify the optimizer 
## 	-Controls the learning rate
##  - many options and mathematically complex
## 	- “Adam" is usually a good choice
## Loss function
## 	- MSE for regression 
## 	-
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting a model 
## Applying backpropagation and gradient descent with your data to update the 
## weights
## Scaling data before fitting can ease optimization
model.fit(predictiors, target)

# Classification models 
## loss function as "categorical_crossentropy"
## add metrics = ['accuracy']  print out the accuracy score at the end of each 
## epoch, make it easier to interpret
## output layer has separate node for each possible outcome and use 'softmax' 
## activation, ensures the  prediction sum to 1, so they can be interpreted like 
## probabilities.

from keras.utils import to_catrgorical # dummy coding the outcome
data = pd.read_csv('basketball_shot_log.csv')
predictors = data.drop(['shot_result'], axis=1).as_matrix()
target = to_categorical(data.shot_result)
model = Sequential()
model.add(Dense(100, activation = 'relu', input_shape=(n_cols,)))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', 
				metrics = ['accuracy'])
model.fit(predictors, target)

# Saving, reloading and using your model
from keras.models import load_model
model.save('model_file.h5')
my_model = load_model('mymodel.h5')
predictions = my_model.predict(data_to_predict_with)
probability_true = predictions[:,1]

# Model validation
model.compile(optimizer='adam', loss = 'categorical_vrossentropy', metrics = ['accuracy']
model.fit(predictors, target, validation_split = 0.3)

## early stopping
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience = 3)
model.fit(predictors, target, validation_split=0.3, epochs=20, 
			callbacks=[early_stopping_monitor])