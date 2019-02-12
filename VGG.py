import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation,Conv2D,MaxPooling2D,BatchNormalization

model = Sequential()

##1st layer
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',input_shape=(224,224,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2,2),strides=(2,2),padding='same'))

##2 nd layer
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))

##3rd layer
model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))

##4th layer
model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))

##5th layer
model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))

##Flatten layer
model.add(Flatten())

##1st Fully connected layer
model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

##2nd Fully connected layer
model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

##Output layer
model.add(Dense(1000))
model.add(BatchNormalization())
model.add(Activation('softmax'))

model.summary()