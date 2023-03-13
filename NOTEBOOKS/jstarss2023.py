# -*- coding: utf-8 -*-

'''
This notebook brings the codes used in the paper submitted to Journal of Selected Topics On Applied Earth Observations and Remote Sensing (JSTARSS)

Title of the paper: New Mean Tropospheric Temperature models based
on Machine Learning Algorithms for Brazil
'''

**Importing the necessary libraries**
"""

import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM,SimpleRNN,GRU
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import joblib

"""





**Loading Data**"""

'''
Description of the datasets:
- Files with the name 'train.txt' are the data used to train the models;
- Files with the name 'target.txt' contais the target values, or the expected values for Tm, used in training as well;
- Files with the name 'test.txt' are the input in the stage of test the models;
- Files with the name 'eval.txt' contains the real Tm values and the values obtained by other models found in literature.  
'''

train_features=np.loadtxt('train.txt')
train_labels=np.loadtxt('target.txt')
test_features=np.loadtxt('test.txt')
evaluation=np.loadtxt('eval.txt')


# For Random Forest and Support Vector Regression, the Reshape operation must be performed.

train_features=train_features.reshape(-1,2)
train_labels=train_labels.reshape(-1,1)
test_features=test_features.reshape(-1,2)

#For Neural Networks, the data must be normalized.

train_features[:,0]=train_features[:,0]/1100
train_features[:,1]=train_features[:,1]/310
train_labels=train_labels/300
test_features[:,0]=test_features[:,0]/1100
test_features[:,1]=test_features[:,1]/310

#After normalization, a reshape operation must be performed on train and test inputs.

train_features = train_features.reshape(train_features.shape[0], train_features.shape[1], 1) 
test_features = test_features.reshape(test_features.shape[0], test_features.shape[1], 1)





"""**Simple Recurrent Neural Network (RNN)**"""

model = Sequential()

model.add(SimpleRNN(2, return_sequences=False,input_shape=(2, 1)))
model.add(Dense(20, activation="relu"))
#model.add(Dense(3, activation="tanh"))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")

"""**Long-Short Term Memory Recurrent Neural Network (LSTM)**"""

model = Sequential()

model.add(LSTM(3, return_sequences=False,input_shape=(2, 1)))
model.add(Dense(20, activation="relu"))
#model.add(Dense(2, activation="relu"))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")

"""**Gated Recurrent Unit Neural Network (GRU)**"""

#Gated Recurrent Unit (GRU) 

model = Sequential()

model.add(GRU(2, return_sequences=False,input_shape=(2, 1)))
model.add(Dense(10, activation="relu"))
#model.add(Dense(10, activation="sigmoid"))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")

"""**Model Fit**"""

#This cell must be run for the Neural Network models, according to the model you instantiate.

model.fit(train_features, train_labels, batch_size=20,epochs=50, verbose=1)

"""**Model Test**"""

#This cell performs the test of the neural network model.

predictions=model.predict(test_features) # Make predictions
pred=predictions*300 #Transforms the results to the scale of Tm values
np.savetxt('cagr_GRU.txt',pred[:,0]) #Save the results in a .txt file.
model.save('GRU_cagr.h5') #Save the keras model in a .h5 file

"""**Loading a keras model and making predictions**"""

# The names and values used in the examples are for representation purposes

model = keras.models.load_model('fedn_LSTM.h5') #Loads the model

#Making the predictions on a variable storing data from file

prediction=model(data) #Consider data the values of surface temperature and pressure read from a .txt file for example

#Making the predictions for one single input

prediction=model(([291.7,901.6]))



"""**Support Vector Regression (SVR)**"""

clf = svm.SVR( kernel="rbf", C=1, gamma=0.1) #Instantiating the model

clf.fit(train_features, train_labels) #Fit the model

predictions=clf.predict(test_features) #Test the model

np.savetxt('cagr_SVR.txt',predictions) #Save the predictions in a .txt file

joblib.dump(clf, "./SVR_cagr.joblib") #Save the object model

"""**Random Forest Regressor (RF)**"""

rf = RandomForestRegressor(n_estimators = 1000,max_depth=7, random_state = 1) #Instantiating the model

rf.fit(train_features, train_labels); #Fit the model

predictions = rf.predict(test_features) #Test the model

np.savetxt('cagr_RF.txt',predictions) #Save the predictions in a .txt file

joblib.dump(rf, "./RF_cagr.joblib") #Save the object model

"""**Statistical** **evaluation**"""

mse = mean_squared_error(evaluation[:,0], pred) #Computes the Mean Squared Error using the real Tm values and predicted Tm values 
rmse=np.sqrt(mse) #Computes the Root Mean Squared Error

#Computes the Huber Metric

gamma=5

erro=evaluation[:,0]-evaluation[:,8]
erro=np.abs(erro)
erros_menor=[]
erros_maior=[]
n=erro.shape[0]
for i in range(n):
  if erro[i]<=gamma:

      erros_menor.append(erro[i])
  else:
      erros_maior.append(erro[i])
    
  
erros_menor=np.asarray(erros_menor)
erros_maior=np.asarray(erros_maior)



huber_menor=(1/n)*np.sum(0.5*(erros_menor**2))

huber_maior=(1/n)*np.sum(gamma*(np.abs(erros_maior)-0.5*gamma))
huber_metric=(huber_maior+huber_menor)/2

huber_metric



