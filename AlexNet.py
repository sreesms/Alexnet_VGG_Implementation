import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout, MaxPooling2D,Conv2D,Flatten,BatchNormalization,Activation
# from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam

##Input data,..

##input image dimension 224x224


model = Sequential()

# 1st layer - convolution layer
model.add(Conv2D(kernel_size = (11,11),filters=96,strides = (4,4),input_shape = (227,227,3),padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
## Max Pooling
model.add(MaxPooling2D(pool_size = (2,2),strides =(2,2),padding='valid'))

##2nd layer -Convolution layer
model.add(Conv2D(kernel_size = (11,11),filters = 256,strides=(1,1),padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))

##3rd convolution layer
model.add(Conv2D(kernel_size= (3,3),filters=384,strides=(1,1),padding='valid'))
model.add(Activation('relu'))

##4th Convolution layer
model.add(Conv2D(kernel_size= (3,3),filters = 384,strides=(1,1),padding='valid'))
model.add(Activation('relu'))

##5th convloution layer
model.add(Conv2D(kernel_size=(3,3),filters=256,strides=(1,1),padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))

model.add(Flatten())

##1st Fully connected layer
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))

##2nd Fully connected layer
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))

##3rd fully connected layer
model.add(Dense(1000))
model.add(Activation('softmax'))

model.summary()

##Compile the model
# model.compile(loss='categorical_crossentropy',optimizer= 'adam',metrics=['accuracy'])



