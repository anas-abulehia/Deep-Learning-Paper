#!/usr/bin/env python
# coding: utf-8

# # Deep Learning Fundamentals (Application)

#     Recongition of Handwritten numbers is an application on Deep Learning and AI technologies. This notebook shows the implementation of a Deep Learning algorithm to Solve MINST data set for 0~9 number (single digit). This example is considered as a HelloWorld program in AI.

# #### First: import library and Dataset ,
#         import tensor flow library our Framework and other necessary libraries. <br> import the dataset, it is 
#         divided train and test partitions.

# In[2]:


import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist                          # mnist is a dataset of 28x28 images of handwritten digits and their labels
(x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test


# #### Second: Data preprocessing. 
#     Usually we need to perfrom some preporessing on the raw data.<br>
#     In our case here, we normalize the images to have pixel value between 0 and 1 instead of 0 and 255

# In[3]:


x_train = tf.keras.utils.normalize(x_train,axis = 1) # normalize the pixels values to be between 0 and 1
x_test = tf.keras.utils.normalize(x_test, axis = 1)  # normalize the pixels values to be between 0 and 1 


# #### Third: Define and Delcare the model.<br>
#     here a preporcessing take place here it is flatten, it convert the image to a flat vector.
#     Define layers and their activation function, optimizer ,loss. 
#     It has 2 Deep layer 1 input and 1 output layer.
#     Now the model is configured and ready for training.

# In[4]:


model = tf.keras.models.Sequential()  # a basic feed-forward model
model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # our output layer. 10 units for 10 classes. Softmax for probability distribution

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track


# #### Fourth: Model Training
#     Train the model with 3 epochs. 

# In[5]:


model.fit(x_train, y_train, epochs=3)


# #### Fifth: Model Evaulation 
#     We evaluate the model by giving it the test data set.
#     In this step if we are not getting good result, we go back to step 3 to declare a new model 

# In[6]:


val_loss, val_acc = model.evaluate(x_test, y_test) 


# #### Six: Result inspection (optional)
#     The purpose is to see the images and compare them with the results of the Deep Learning Algorithm. (for presentation purpose only)

# In[9]:


import random
xrand = random.randint(0, 10000)
x_test[xrand].shape
import numpy as np
print("The prediction is ",np.argmax(model.predict(x_test[xrand:xrand+1])))# it retunrs the index of maximum element
plt.imshow(x_test[xrand],cmap='gray', vmin=0, vmax=.8)
plt.show()
print ("ground truth is")
print(y_test[xrand])


# In[ ]:





# In[ ]:




