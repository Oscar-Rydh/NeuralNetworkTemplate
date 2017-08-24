import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###########################################
# Part 1 Data preprocessing               #
# Data needs to be represented correctly  #
###########################################
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

###########################################
# Part 2 Creating a ANN                   #
# Creating a nural network with regulizer #
###########################################
import keras
# Initializes the ANN
from keras.models import Sequential
# Used to create the network
from keras.layers import Dense
# Dropout regularization to reduce overfitting when the variance is high, or if the trainset acc is much better then testset
from keras.layers import Dropout

# Initilizing a nural network for classification
classifier = Sequential()

# Adding input layer and first hidden layer
# Choosing the number of nodes in a hidden layer is an art and is perfected via experimentation
# For the none artistic choose N = (inputnodes + outputnodes) / 2
# Units = number of nodes in layer
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim=11))
# Dropout start with rate 0.1 then 0.2 etc upto 0.4
classifier.add(Dropout(rate=0.1))

# Adding a second hidden layer.
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))
classifier.add(Dropout(rate=0.1))

# Adding the output layer. Units = 1 since its classifying, activation = sigmoid since we want the probability for the outcome
# Activation function must be softmax if we have more classes
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

# Compile the ANN (Apply stochastic gradiant descent or something else) 
# Refer udemy ANN step8 for info
# Optimizer: The algorithm to update weights (train)
# Loss: lossfunction of defining and calculating the cost
# Metrics: Criteria for evaluating the NN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])

# Fitting (train) the ANN to our trainingset
classifier.fit(X_train, y_train, batch_size=80, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Confusion matrix need true or false values, not probabilities
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


##########################################################
# Part 3 Evaluating the ANN with K-FOLD cross validation #
# The NN needs to have its accuracy validated for trust  #
##########################################################
# Evaluating the ANN with k-fold cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# K-fold cross validation expects a function to build the NN
def build_classifier():
    from keras.models import Sequential
    # Used to create the network
    from keras.layers import Dense
    classifier = Sequential()
    classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim=11))
    classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])
    return classifier

# Same constructor as for the Sequential fit function
k_fold_classifier = KerasClassifier(build_fn=build_classifier, batch_size=80, epochs=100)
# cv: nbr of folds
# n_jobs: number of cpus used
accuracies = cross_val_score(estimator=k_fold_classifier, X = X_train, y=y_train, cv=10, n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()


##############################################################
# Part 4 Improving the ANN hyper params with Grid search     #
# Using grid search to improve the accuracy to beat K-fold   #
##############################################################

# Improving the ANN with grid search
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Grid search expects a function to build the NN
# Tuning paramaters that already exists must be passed as arguments to the build
def build_classifier(optimizer):
    from keras.models import Sequential
    # Used to create the network
    from keras.layers import Dense
    classifier = Sequential()
    classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim=11))
    #classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics= ['accuracy'])
    return classifier

# Same constructor as for the Sequential fit function
grid_search_classifier = KerasClassifier(build_fn=build_classifier)

parameters = {'batch_size': [25, 32, 64, 128],
              'nb_epoch': [100, 500, 1000],
              'optimizer': ['adam', 'rmsprop']}


# Initialize the grid search
grid_search = GridSearchCV(estimator=grid_search_classifier, param_grid=parameters,
                           n_jobs=-1, cv=10, verbose=1, scoring = 'accuracy')
# Train the model
grid_search = grid_search.fit(X_train, y=y_train)
best_paramerters = grid_search.best_params_
best_accuracy = grid_search.best_score_






