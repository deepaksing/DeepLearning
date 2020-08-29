import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

#-------------->  LOAD THE DATA

(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

#since the range of values are from 0 to 255 (color  range og RGB)
#we can simplify it for our convienience

train_data = train_data/255.0
test_data = test_data/255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#-------------->   BUILD THE MODEL

#here we will be having three layers the first one is the input layer 
#the input layer will be flattened into 1D array

#the second layer will be dense layer 
#dense means fully connected nodes, every nodes of input will be connected with every nodes of second layer
#the second layer will be the one which will work behind the scenes to give us accurate data

#the last layer will be the output layer which will contain all the labels

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(128, activation='relu'),
	keras.layers.Dense(10)
])	

#-------------->  COMPILE THE MODEL

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=8)

# test_loss, test_accu = model.evaluate(test_data, test_labels)
# print("Tested Acc: ", test_accu)
#probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = model.predict(test_data)

# gives 10 prediciton values for each data
#take the max value in the list , -> and thats the most perfect match for the data

#print(class_names[np.argmax(prediction[0])]) # argmax returns index of the best match and search that index in the className

#displaying in the graph


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):
  predicted_label = class_names[np.argmax(predictions[i])]
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(test_data[i], cmap=plt.cm.binary)
  plt.xlabel(class_names[test_labels[i]])
  if predicted_label == class_names[test_labels[i]]:
    color = 'blue'
  else:
    color = 'red'
  plt.title("Prediction :" + predicted_label, color=color)
  plt.tight_layout()
plt.show()