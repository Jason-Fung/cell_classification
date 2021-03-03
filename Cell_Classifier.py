#!/usr/bin/env python
# coding: utf-8

# # Project 2: Label-Free Image Recognition of Cancer Cells in Blood

# In[2]:

# import all  libraries
# keras libraries
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array, load_img,array_to_img,ImageDataGenerator
from keras.callbacks import EarlyStopping

# get_ipython().run_line_magic('pylab', 'inline')
import numpy as np
import scipy.io
import sklearn
from sklearn.model_selection import train_test_split
import os 
import glob

import random
import pandas as pd

# Library for figures
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from sklearn.metrics import confusion_matrix
import itertools

# In[3]:

# uncomment this segment to unzip zip file of images 

# import zipfile
# with zipfile.ZipFile('cell_mat.zip', 'r') as zip_ref:
#     zip_ref.extractall()

# In[4]:

# split up cells to load in specific cell types
pwd = os.getcwd()
cell_types = ['HT29','Jurkat','PMBC','Whole_blood']
label_types = [0,1,2,3]

# reshaping dimensions and parameters
dim = 106
layer = 4
new_shape_mat = (dim,dim,layer)
new_shape_vis = (dim,dim)

# In[5]:
# Load in HT29 Dataset

path_mat = r''+ pwd + '\\images\\' + cell_types[0] + '\\single_cell_im' # path for corresponding .mat files
filename_mat = sorted(glob.glob(os.path.join(path_mat,'*.mat'))) # get all mat file path

print(len(filename_mat))

train_arr_0 = []
train_label_0 = []
for idx,fname in enumerate(filename_mat):
    im = scipy.io.loadmat(fname)["singleCell_Im"]
    for clr in range(0,4): # go through each layer of the image
        tmp_im = im[:,:,clr]
        tmp_im = tmp_im - tmp_im.min() # boost contrast
        im[:,:,clr] = tmp_im # edit the layer
        tmp_im = []

    # set up resizing parameters
    new_im = np.zeros(new_shape_mat)
    new_size = np.shape(new_im)
    old_size = np.shape(im)

    # following code pads the image and resize
    new_im[int((new_size[0]-old_size[0])/2):int((new_size[0]-old_size[0])/2)+old_size[0],
           int((new_size[1]-old_size[1])/2):int((new_size[1]-old_size[1])/2)+old_size[1]] += im

    train_label_0.append(label_types[0]) # store the labels
    train_arr_0.append(new_im)
print('shape of HT29 Cell Images: ',np.shape(train_arr_0))
print('Done loading HT29 Cell Images')


# In[6]:
# Load in Jurkat Dataset

path_mat = r''+ pwd + '\\images\\'+cell_types[1]+'\\single_cell_im' # path for corresponding .mat files
filename_mat = sorted(glob.glob(os.path.join(path_mat,'*.mat'))) # get all mat file path
    
train_arr_1 = []
train_label_1 = []
for idx,fname in enumerate(filename_mat):
    im = scipy.io.loadmat(fname)["singleCell_Im"]
    for clr in range(0,4): # go through each layer of the image
        tmp_im = im[:,:,clr]
        tmp_im = tmp_im - tmp_im.min() # boost contrast
        im[:,:,clr] = tmp_im # edit the layer
        tmp_im = []

    # set up resizing parameters
    new_im = np.zeros(new_shape_mat)
    new_size = np.shape(new_im)
    old_size = np.shape(im)

    # following code pads the image and resize
    new_im[int((new_size[0]-old_size[0])/2):int((new_size[0]-old_size[0])/2)+old_size[0],
           int((new_size[1]-old_size[1])/2):int((new_size[1]-old_size[1])/2)+old_size[1]] += im

    train_label_1.append(label_types[1]) # store the labels
    train_arr_1.append(new_im)
    
print('shape of Jurkat Cell Images: ',np.shape(train_arr_1))
print('Done loading Jurkat Cell Images')

# In[7]:
# Load in PMBC Dataset

path_mat = r''+ pwd + '\\images\\'+cell_types[2]+'\\single_cell_im' # path for corresponding .mat files
filename_mat = sorted(glob.glob(os.path.join(path_mat,'*.mat'))) # get all mat file path
    
train_arr_2 = []
train_label_2 = []
for idx,fname in enumerate(filename_mat):
    im = scipy.io.loadmat(fname)["singleCell_Im"]
    for clr in range(0,4): # go through each layer of the image
        tmp_im = im[:,:,clr]
        tmp_im = tmp_im - tmp_im.min() # boost contrast
        im[:,:,clr] = tmp_im # edit the layer
        tmp_im = []

    # set up resizing parameters
    new_im = np.zeros(new_shape_mat)
    new_size = np.shape(new_im)
    old_size = np.shape(im)

    # following code pads the image and resize
    new_im[int((new_size[0]-old_size[0])/2):int((new_size[0]-old_size[0])/2)+old_size[0],
           int((new_size[1]-old_size[1])/2):int((new_size[1]-old_size[1])/2)+old_size[1]] += im

    train_label_2.append(label_types[2]) # store the labels
    train_arr_2.append(new_im)
print('shape of PMBC Cell Images: ',np.shape(train_arr_2))   
print('Done loading PMBC Cell Images')


# In[ ]:
# Load in Whole Blood Cell Dataset

path_mat = r''+ pwd + '\\images\\'+cell_types[3]+'\\single_cell_im' # path for corresponding .mat files
filename_mat = sorted(glob.glob(os.path.join(path_mat,'*.mat'))) # get all mat file path
    
train_arr_3 = []
train_label_3 = []
for idx,fname in enumerate(filename_mat):
    im = scipy.io.loadmat(fname)["singleCell_Im"]
    for clr in range(0,4): # go through each layer of the image
        tmp_im = im[:,:,clr]
        tmp_im = tmp_im - tmp_im.min() # boost contrast
        im[:,:,clr] = tmp_im # edit the layer
        tmp_im = []

    # set up resizing parameters
    new_im = np.zeros(new_shape_mat)
    new_size = np.shape(new_im)
    old_size = np.shape(im)

    # following code pads the image and resize
    new_im[int((new_size[0]-old_size[0])/2):int((new_size[0]-old_size[0])/2)+old_size[0],
           int((new_size[1]-old_size[1])/2):int((new_size[1]-old_size[1])/2)+old_size[1]] += im

    train_label_3.append(label_types[3]) # store the labels
    train_arr_3.append(new_im)
print('shape of Whole Blood Cell Images: ',np.shape(train_arr_3))       
print('Done loading Whole Blood Cell Images')
# In[ ]: Process all the data 

# change list to arrays
np.array(train_arr_0)
np.array(train_arr_1)
np.array(train_arr_2)
np.array(train_arr_3)

np.array(train_label_0)
np.array(train_label_1)
np.array(train_label_2)
np.array(train_label_3)

# number of test images for each cell 
N_test = 200

# distribute data into training and testing 
train_arr_0, test_arr_0, train_label_0, test_label_0 = train_test_split(train_arr_0,train_label_0,test_size = N_test)
train_arr_1, test_arr_1, train_label_1, test_label_1 = train_test_split(train_arr_1,train_label_1,test_size = N_test)
train_arr_2, test_arr_2, train_label_2, test_label_2 = train_test_split(train_arr_2,train_label_2,test_size = N_test)
train_arr_3, test_arr_3, train_label_3, test_label_3 = train_test_split(train_arr_3,train_label_3,test_size = N_test)
# join all training data together
train_set = np.concatenate((train_arr_0,train_arr_1))
train_set = np.concatenate((train_set,train_arr_2))
train_set = np.concatenate((train_set,train_arr_3))

train_label = np.concatenate((train_label_0,train_label_1))
train_label = np.concatenate((train_label,train_label_2))
train_label = np.concatenate((train_label,train_label_3))
print(np.shape(train_set))

# join all testing data together
test_set = np.concatenate((test_arr_0,test_arr_1))
test_set = np.concatenate((test_set,test_arr_2))
test_set = np.concatenate((test_set,test_arr_3))

test_label = np.concatenate((test_label_0,test_label_1))
test_label = np.concatenate((test_label,test_label_2))
test_label = np.concatenate((test_label,test_label_3))
print(np.shape(test_set))

def process(dataset,labels,dim,layer):
    # function process() takes in the numpy array dataset and subtracts the mean image in the 
    # dataset. This includes resizing the images into a 1-D array (num_images, len(1-D Image),
    # subtracting the mean, then resizing it back into shape (num_images,dim,dim,layer).
    # Finally, shuffles the dataset. 
    
    tmp_dataset = dataset
    tmp_labels = labels
    tmp_dataset = np.reshape(tmp_dataset,(len(tmp_dataset),tmp_dataset[0].size))
    #normalize images
    mean_data = np.reshape(np.mean(tmp_dataset,axis = 1),(-1,1))
    tmp_dataset -= mean_data
    # reshape back into (num_images,dim,dim,layer)
    tmp_dataset = np.reshape(tmp_dataset,(np.shape(tmp_dataset)[0],dim,dim,layer))
    tmp_dataset = np.reshape(tmp_dataset,(-1,dim,dim,layer))
    # shuffle the dataset and labels in the same way
    indices = np.arange(tmp_dataset.shape[0])
    random.shuffle(indices)
    tmp_dataset = tmp_dataset[indices]
    tmp_labels = tmp_labels[indices]
    tmp_labels = keras.utils.to_categorical(tmp_labels, num_classes=(4))
    return tmp_dataset, tmp_labels

train_set,train_label = process(train_set,train_label,dim,4)
test_set,test_label = process(test_set,test_label,dim,4)
train_set, val_set, train_label, val_label = train_test_split(train_set,train_label,test_size = 0.1)

print('Done concatenating and finalizing datasets!')

# In[ ]:
# Define a convolutional structure for training

def small_vgg16(dim,layer):
    # scaled down and modified version of vgg16 that is suitable for single core processing
    
    model = Sequential()
    # first layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(dim,dim,layer),padding='same',name='block1_conv2_1'))
    model.add(Conv2D(32, (3, 3), activation='relu',padding='same',name='block1_conv2_2'))
    model.add(MaxPooling2D(pool_size=(2, 2),name='block1_MaxPooling'))
    model.add(Dropout(0.25))
    # second layer
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same',name='block2_conv2_1'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same',name='block2_conv2_2'))
    model.add(MaxPooling2D(pool_size=(2, 2),name='block2_MaxPooling'))
    model.add(Dropout(0.25))
    # third layer 
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same',name='block3_conv2_1'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same',name='block3_conv2_2'))
    # model.add(Conv2D(128, (3, 3), activation='relu',padding='same',name='block3_conv2_3')) 
    model.add(MaxPooling2D(pool_size=(2, 2),name='block3_MaxPooling'))
    model.add(Dropout(0.25))
    # fourth layer
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same',name='block4_conv2_1'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same',name='block4_conv2_2'))
    # model.add(Conv2D(256, (3, 3), activation='relu',padding='same',name='block4_conv2_3'))
    model.add(MaxPooling2D(pool_size=(2, 2),name='block4_MaxPooling'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu',name='final_output_1'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu',name='final_output_2'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax',name='class_output')) # 4 categories
    return model

# In[ ]:
# model parameters and options
batchsize = 100

model = small_vgg16(dim,layer)
sgd = SGD(lr=0.001, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3, verbose = 1, mode = 'auto')

# In[]:
# train in batches 

# save model weights and 
modelname = 'model.h5'
historyname = 'history.xlsx'
historysavepath = 'model'

# load in the data, if it doesnt exist then run the training 
try:
    load_model(modelname)
    history = pd.read_excel(historyname)
except:
    # run the fit model 
    history = model.fit(train_set,train_label, batch_size = batchsize, epochs = 20, 
                    validation_data = (val_set,val_label), callbacks = [early_stop] )
    model.evaluate(test_set,test_label)
    # save architecture and weights
    model.save(modelname)
    # save history performance 
    import collections
    hist = history.history
    for key, val in hist.items():
        numepo = len(np.asarray(val))
        break
    hist = collections.OrderedDict(hist)
    pd.DataFrame(hist).to_excel(historyname, index=False)

# In[]: ONLY RUN THIS IF THERE IS AN XLSX FILE FOR HISTORY
plt.figure(1)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'],loc = 'upper left')

plt.figure(2)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'],loc = 'upper right')


# In[]: ONLY RUN THIS IF THERE IS NO FILE XLSX FILE FOR HISTORY i.e., RIGHT AFTER RUNNING THE MODEL.FIT

# Plot learning curves 
# plt.figure(1)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'],loc = 'upper left')

# plt.figure(2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'],loc = 'upper right')


# In[]: Plot TSNE scatter plot for image classifier
# get the final output neuron 
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('final_output_2').output)
# predict test data
intermediate_output = intermediate_layer_model.predict(
        test_set, batch_size=batchsize, verbose=1)
# perform calculations on similarities between classes
Y = TSNE(n_components=2, init='random', random_state=0, perplexity=30, 
         verbose=1).fit_transform(intermediate_output.reshape(intermediate_output.shape[0],-1))

layer_output_label = np.argmax(test_label, axis=1)
df = pd.DataFrame(dict(x=Y[:,0], y=Y[:,1], label=layer_output_label))
groups = df.groupby('label')
# plotting parameters

plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(10,10))

labels_dict=['HT29','Jurkat','PMBC','Whole_blood']
ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
for label, group in groups:
    name = labels_dict[label]
    point,=ax.plot(group.x, group.y, marker='o', linestyle='', ms=8, label=name, alpha=0.8)

plt.xlabel('Metafeatures 1')
plt.ylabel('Metafeatures 2')
plt.title('t-SNE Scattering Plot')
ax.legend(prop={'size': 15})

# In[]: Plot Confusion Matrix
test_pred=model.predict(test_set)
# calculate confusion matrix
cnf_matrix = confusion_matrix(np.argmax(test_label, axis=1).reshape(-1,1),
                              np.argmax(test_pred, axis=1).reshape(-1,1))
plt.figure(figsize = (10,10))
plt.imshow(cnf_matrix,interpolation = 'nearest', cmap = plt.cm.Blues)
# put the text into the confusion matrix
fmt = '.2f' # put it into float
thresh = cnf_matrix.max() / 2.
for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
    plt.text(j, i, format(cnf_matrix[i, j], fmt),
             horizontalalignment="center",
             color="white" if cnf_matrix[i, j] > thresh else "black")

plt.colorbar()
x_tick_labels = ['HT29','Jurkat','PMBC','Whole_blood']
y_tick_labels = ['HT29','Jurkat','PMBC','Whole_blood']
x_num = np.arange(np.float(len(x_tick_labels)))
y_num = np.arange(np.float(len(y_tick_labels)))
plt.xticks(x_num,x_tick_labels)
plt.yticks(y_num,y_tick_labels)
plt.ylim(3.5,-0.5)
plt.grid(False)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()
