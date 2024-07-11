#Import related packages
import os
import tensorflow as tf
from tensorflow.keras import callbacks
from PIL import Image
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

#Import Data
filelst = os.listdir('data/')
filelst = ['data/'+v for v in filelst]
filelst.sort()
imgs = [np.loadtxt(file) for file in filelst]
img_x,img_y=(608,416)
dx = 608
dy = 416
imgs = [cv2.resize(v,(img_x, img_y)) for v in imgs]
X_train = np.array(imgs)
X_train = X_train.astype('float32')
X_train = np.array([[X_train[:,vv*dy:(vv+1)*dy, v*dx:(v+1)*dx] for v in range(img_x//dx)] for vv in range(img_y//dy)]).reshape(-1,dy,dx)[:,...,np.newaxis]
print(X_train.shape)

for i in range(The amount of data):
    if i==0:
        img=np.concatenate((X_train[0],X_train[1],X_train[2]), axis=2)# (400, 600, 3)
        X_train_1 = np.expand_dims(img, axis=0)# (1, 400, 600, 3)
    else:
        img=np.concatenate((X_train[j-1],X_train[j],X_train[j+1]), axis=2)
        img = np.expand_dims(img, axis=0)  # (1, 400, 600, 3)
        X_train_1 = np.concatenate((X_train_1, img), axis=0)
print(X_train_1.shape)

filelst = os.listdir('label/')
filelst = ['label/'+v for v in filelst]
filelst.sort()
manuals = [np.loadtxt(file) for file in filelst]
img_x,img_y=(608,416)
dx = 608
dy = 416
manuals = [cv2.resize(v,(img_x, img_y)) for v in manuals]
Y_train = np.array(manuals)
Y_train = Y_train.astype('float32')
Y_train = np.array([[Y_train[:,vv*dy:(vv+1)*dy, v*dx:(v+1)*dx] for v in range(img_x//dx)] for vv in range(img_y//dy)]).reshape(-1,dy,dx)[:,...,np.newaxis]
print(Y_train.shape)

#Network model DeepFPN
def convres_block(x, filters):
    res = Conv2D(filters, (1, 1), padding='same')(x)
    conv = Conv2D(filters, (3, 3), padding='same')(x)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(filters, (3, 3), padding='same')(x)
    conv = BatchNormalization()(conv)
    conv = conv + res
    conv = Activation('relu')(conv)
    return conv

def dilatedconv_block(x,filters):
    res0 = Conv2D(filters,(1, 1),padding='same')(x)
    ## 膨胀卷积
    conv0_1 = Conv2D(filters, (3, 3), padding='same', dilation_rate=1)(x) 
    conv0_1 = BatchNormalization()(conv0_1)

    conv0_2 = Conv2D(filters, (3, 3), padding='same', dilation_rate=2)(x) 
    conv0_2 = BatchNormalization()(conv0_2)

    conv0_3 = Conv2D(filters, (3, 3), padding='same', dilation_rate=3)(x)  
    conv0_3 = BatchNormalization()(conv0_3)

    conv0 = conv0_1+conv0_2+conv0_3 
    conv0 = Activation('relu')(conv0)
    conv0 = conv0 + res0
    conv0 = Activation('relu')(conv0)
    return conv0

def DeepFPN(img_rows, img_cols, img_chan=1):
    inputs = Input((img_rows, img_cols, img_chan))
#Input Shape 416*608
    conv1=convres_block(inputs, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2=convres_block(pool1, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3=convres_block(pool2, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4=dilatedconv_block(pool3,256)
    
    up5 = concatenate([Conv2DTranspose(128, (2, 2), strides=(
         2, 2), padding='same')(conv4), conv3], axis=3)
    conv5=convres_block(up5, 128)    
    up6 = concatenate([Conv2DTranspose(64, (2, 2), strides=(
         2, 2), padding='same')(conv5), conv2], axis=3)
    conv6=convres_block(up6, 64)    
    up7 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
         2, 2), padding='same')(conv6), conv1], axis=3)
    conv7=convres_block(up7, 32)
        
#208*304    
    conv12=convres_block(pool1, 64)
    pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
    conv22=convres_block(pool12, 128)
    pool22 = MaxPooling2D(pool_size=(2, 2))(conv22)
    conv32=convres_block(pool22, 256)
    pool32 = MaxPooling2D(pool_size=(2, 2))(conv32)
    
    conv42=dilatedconv_block(pool32,512)
        
    up52 = concatenate([Conv2DTranspose(256, (2, 2), strides=(
         2, 2), padding='same')(conv42), conv32], axis=3)
    conv52=convres_block(up52, 256)    
    up62 = concatenate([Conv2DTranspose(128, (2, 2), strides=(
         2, 2), padding='same')(conv52), conv22], axis=3)
    conv62=convres_block(up62, 128)
    up72 = concatenate([Conv2DTranspose(64, (2, 2), strides=(
         2, 2), padding='same')(conv62), conv12], axis=3)
    conv72=convres_block(up72, 64)
        
#104*152    
    conv13=convres_block(pool2, 128)
    pool13 = MaxPooling2D(pool_size=(2, 2))(conv13)
    conv23=convres_block(pool13, 256)
    pool23 = MaxPooling2D(pool_size=(2, 2))(conv23)
    conv33=convres_block(pool23, 512)
    pool33 = MaxPooling2D(pool_size=(2, 2))(conv33)
    conv43=dilatedconv_block(pool33,1024)
      
    up53 = concatenate([Conv2DTranspose(512, (2, 2), strides=(
         2, 2), padding='same')(conv43), conv33], axis=3)
    conv53=convres_block(up53, 512)
    up63 = concatenate([Conv2DTranspose(256, (2, 2), strides=(
         2, 2), padding='same')(conv53), conv23], axis=3)
    conv63=convres_block(up63, 256)
    up73 = concatenate([Conv2DTranspose(128, (2, 2), strides=(
         2, 2), padding='same')(conv63), conv13], axis=3)
    conv73=convres_block(up73, 128)
    
    
    P2 = Conv2D(32, (1, 1), padding='same')(conv73)  
#104*152*32
    P1 = Add()([UpSampling2D(size=(2, 2))(P2),Conv2D(32, (1, 1))(conv72)])   #208*304*32
    P0 = Add()([UpSampling2D(size=(2, 2))(P1),Conv2D(32, (1, 1))(conv7)])   #416*608*32
    
    P0 = Conv2D(32, (3, 3), padding='same')(P0)
    P0 = Conv2D(1, (1, 1), activation='sigmoid')(P0)
    model = Model(inputs=[inputs], outputs=[P0])
return model

#train
def dice_coef(y_true, y_pred):
    from tensorflow.keras import backend as K
    smooth = 1.
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    And = K.sum(y_truef * y_predf)
    return ((2. * And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))
print(X_train_1.shape)
model = DeepFPN(X_train_1.shape[1],X_train_1.shape[2],X_train_1.shape[3])
model.summary()
LR=0.001
EPOCHS=100
checkpointer = ModelCheckpoint(filepath='best.h5', verbose=1, monitor='val_dice_coef', 
                              mode='max', save_best_only=True)
model.compile(optimizer=Adam(lr=LR), loss='binary_crossentropy',metrics=[dice_coef])
history1=model.fit(X_train_1, Y_train, batch_size=8, epochs=EPOCHS,shuffle=True, validation_split=0.2,callbacks=[checkpointer])
