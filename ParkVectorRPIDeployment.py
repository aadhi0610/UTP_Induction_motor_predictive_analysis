#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import pandas as pd
#import seaborn as sns
import tensorflow as tf




from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.applications.vgg16 import preprocess_input,VGG16
from tensorflow.keras.layers import MaxPooling2D,Conv2D,Dense,BatchNormalization,Dropout,GlobalAveragePooling2D,Flatten,Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report,confusion_matrix
#from tensorflow.keras.utils.vis_utils import plot_model
#import ipywidgets as widgets
from sklearn.preprocessing import OneHotEncoder
#import io
from PIL import Image
from IPython.display import display,clear_output
#import cv2
#from warnings import filterwarnings
#   from glob import glob


# In[6]:


len(os.listdir('/home/pi/Desktop/UTP'))


# In[7]:


# Prepere data
healthy = os.listdir('/home/pi/Desktop/UTP/Healthy')
unhealthy  = os.listdir('/home/pi/Desktop/UTP/Unhealthy')


# In[9]:


# Prepere input data
X_data =[]
for file in healthy:
    image=img.imread("/home/pi/Desktop/UTP/Healthy/"+file)
    #img = cv2.imread('/home/pi/Desktop/UTP/Healthy'+file)
    #face = cv2.resize(img, (224, 224) )
    #(b, g, r)=cv2.split(face) 
    #img=cv2.merge([r,g,b])
    X_data.append(image)

for file in unhealthy:
    image=img.imread("/home/pi/Desktop/UTP/Unhealthy/"+file)
    #img = cv2.imread('/home/pi/Desktop/UTP/Unhealthy'+file)
    #face = cv2.resize(img, (224, 224) )
    #(b, g, r)=cv2.split(face) 
    #img=cv2.merge([r,g,b])
    X_data.append(image)


# In[10]:


X = np.squeeze(X_data)
X.shape


# In[11]:


#show one training sample
from matplotlib import pyplot as plt
plt.imshow(X[5], interpolation='nearest')
plt.show()


# In[12]:


# normalize data
X = X.astype('float32')
X /= 255


# In[13]:


# Prepare outputs:
target_yes=np.full(len(healthy),1)
target_no=np.full(len(unhealthy),0)
Y=np.concatenate([target_yes,target_no])
Y


# In[14]:


len(Y)


# In[15]:


#  creation of x_train, y_train, x_test, y_test arrays
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]
print('number_of_train:', number_of_train)
print('number_of_test:', number_of_test)


# In[16]:


# Modıfy input and output for processing in Keras
X_train = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2]*3)
X_test = X_test.reshape(number_of_test,X_test.shape[1]*X_test.shape[2]*3)
print("X train flatten",X_train.shape)
print("X test flatten",X_test.shape)


# In[17]:


# Evaluating the ANN
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential # initialize neural network library
from tensorflow.keras.layers import Dense # build our layers library
def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 250)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 3)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))


# In[18]:


# Predictions on Test Datasets using ANN model
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
y_pred = y_pred>0.5


# In[19]:


X0 = np.reshape(X_test[0], (525, 700, 3))
X1 = np.reshape(X_test[1], (525, 700, 3))
X2 = np.reshape(X_test[2], (525, 700, 3))
X3 = np.reshape(X_test[3], (525, 700, 3))


# In[20]:


print(y_pred)
op=[]
for i in range(0,4):
  if y_pred[i] == 1:
    op.append("Healthy")
  else:
    op.append("UnHealthy")


print(op)

  


# In[21]:


plt.imshow(X0)
print ("Prediction:", op[0])


# In[22]:


plt.imshow(X1)
print ("Prediction:", op[1])


# In[23]:


plt.imshow(X2)
print ("Prediction:", op[2])


# In[24]:


plt.imshow(X3)
print ("Prediction:", op[3])

