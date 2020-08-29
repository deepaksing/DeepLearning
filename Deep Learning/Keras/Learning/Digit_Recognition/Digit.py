import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

#-------------->  LOAD THE DATA
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#--------------> NORMALIZE THE DATASET

# Normalize the train dataset
x_train = tf.keras.utils.normalize(x_train, axis=1)
# Normalize the test dataset
x_test = tf.keras.utils.normalize(x_test, axis=1)

#-------------->   BUILD THE MODEL

#to build the model we are going to create a model object

model = keras.models.Sequential()

# 1.

#then we will flatten the data --> the image is 28x28 pixels so flattening it would give us a 1D array
#that array would be of size 784(28*28)

model.add(keras.layers.Flatten())

# 2.

#we define a hidden layer and an activation function "relu"
#here we have to densely connected layer with both having 128 neurons

#first dense layer would be that having some types of curves and the activation function
#would find a perfect weights and values of each node from input
#then it passes these values to another densly connected layer 
#where we have some part of a letter 
#for eg:  if we see a number 2 we can identify it with an upper curve 
#and a stratight line below it

#so this is also how the layer works it finds the most matching part of a number and then fix its weight accordingly
#it breaks the letter into small pieces and then combine it at each layers

#and finally at output layer it gives the whole number
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))

# 3.

#last we create a output model having 10 neurons (0-9)

model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

#-------->  COMPILE THE MODEL

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#--------> TRAIN THE MODEL

model.fit(x=x_train, y=y_train, epochs=5)

#---------> Evaluate the model

# Evaluate the model performance
test_loss, test_acc = model.evaluate(x=x_test, y=y_test)
# Print out the model accuracy 

predictions = model.predict([x_test]) # Make prediction
print(np.argmax(predictions[0])) # Print out the number

plt.imshow(x_test[0], cmap="gray") # Import the image
plt.show() # Show the image