#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 08:30:08 2017

@author: alexey
"""

from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

conv_base = VGG16(weights='imagenet',
             include_top=False,
                  input_shape=(300, 225, 3))

#conv_base.summary()

# Define sources of data
base_dir = '/home/alexey/Documents/tires/images/run4_SET1AND2_3_classes/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 9, 7, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(300, 225),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

N_train=len(os.listdir(train_dir+'/1_BAD'))+len(os.listdir(train_dir+'/2_OK'))+len(os.listdir(train_dir+'/3_PERFECT'))
train_features, train_labels = extract_features(train_dir, N_train)
N_validation=len(os.listdir(validation_dir+'/1_BAD'))+len(os.listdir(validation_dir+'/2_OK'))+len(os.listdir(validation_dir+'/3_PERFECT'))
validation_features, validation_labels = extract_features(validation_dir, N_validation)
#validation_features, validation_labels = train_features, train_labels
#test_features, test_labels = extract_features(test_dir, 63)
test_features, test_labels = validation_features, validation_labels



#train_features = np.reshape(train_features, (448, 9 * 7 * 512))
#validation_features = np.reshape(validation_features, (63, 9 * 7 * 512))
#test_features = np.reshape(test_features, (63, 9 * 7 * 512))

#################################################################################################

from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(9, 7, 512)))
#model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.Flatten())

##### 
#model.add(layers.Dense(256, activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(3, activation='sigmoid'))

#####

model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))



#model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
#              loss='binary_crossentropy',
#              metrics=['acc'])

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

    
history = model.fit(train_features, train_labels,
                    epochs=80,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))


#################################################################################################


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'g', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

############################################ Try to build the full model by simple stacking two models: #########

full_model=models.Sequential()
full_model.add(conv_base)
full_model.add(model)

full_model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

# Saving the model:
save_dir='/home/alexey/Documents/tires/keras_models/'
full_model.save(save_dir+'2017_09_25_VGG_plus_softmax_doublecheck.h5')
########################################### Now try to predict class of arbitrary image: ################
import cv2

#
#from sklearn import preprocessing
#le = preprocessing.LabelEncoder()
#le.fit(["worn tire", "good tire"])
#le.fit(["a", "b", "d", "c", "e"])




directory_good='/home/alexey/Documents/tires/images/SMALL_initial_run2/selected_picture_validation/good/'
directory_bad='/home/alexey/Documents/tires/images/SMALL_initial_run2/selected_picture_validation/bad/'
tire_labels=['worn tire', 'good tire']

predicted_good=[]
for ii in range(28):
    print 'good_' + str(ii).zfill(3) + '.jpg'
    test_img = cv2.imread(directory_good+'good_' + str(ii).zfill(3) + '.jpg')
    input_test_image=np.expand_dims(test_img, axis=0)*(1./255)
    prob_good=full_model.predict(input_test_image)[0][0]
    tire_class=tire_labels[int(prob_good>0.5)]
    print "file   ", ii, "   predict   ",   "class=", tire_class, "prob_good=", prob_good
    predicted_good.append(tire_class)

# Now predicting bad:
predicted_bad=[]
for ii in range(35):
    print 'bad_' + str(ii).zfill(3) + '.jpg'
    test_img = cv2.imread(directory_bad+'bad_' + str(ii).zfill(3) + '.jpg')
    input_test_image=np.expand_dims(test_img, axis=0)*(1./255)
    prob_good=full_model.predict(input_test_image)[0][0]
    tire_class=tire_labels[int(prob_good>0.5)]
    print "file   ", ii, "   predict   ",   "class=", tire_class, "prob_good=", prob_good
    predicted_bad.append(tire_class)
    
tire_class, prob_good = predict_tire('bad_000.jpg')
print tire_class, "   probability of good tire is: ", prob_good 
