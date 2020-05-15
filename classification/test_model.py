# -*- coding: utf-8 -*-
#author:huan

import keras.layers as kl
import keras.initializers as ki
import keras.models as km

def conv_block(input_tensor, kernel_size, filters, stage, num_block,
               strides = (2, 2), trainable = True):
    
    filter1, filter2, filter3 = filters
    conv_name_base = 'conv' + str(stage) + '_block' + num_block
    
    #layer1
    x = kl.Conv2D(filter1, (1, 1), strides = strides, padding ='same',  
               # kernel_initializer = ki.Ones(),
               kernel_initializer = ki.he_normal(),                
               bias_initializer = ki.Zeros(),
               name = conv_name_base + '_1_conv', trainable = trainable)(input_tensor) 
    x = kl.BatchNormalization(name = conv_name_base + '_1_bn')(x) 
    x = kl.Activation('relu')(x)
    
    #layer2
    x = kl.Conv2D(filter2,(kernel_size, kernel_size), padding ='same',
               # kernel_initializer = ki.Ones(),
               kernel_initializer = ki.he_normal(),                
               bias_initializer = ki.Zeros(),
               name = conv_name_base + '_2_conv', trainable = trainable)(x) 
    x = kl.BatchNormalization(name = conv_name_base + '_2_bn')(x) 
    x = kl.Activation('relu')(x) 
    
    #layer3
    x = kl.Conv2D(filter3,(1, 1), padding = 'same',
               # kernel_initializer = ki.Ones(),
               kernel_initializer = ki.he_normal(),                
               bias_initializer = ki.Zeros(),
               name = conv_name_base +'_3_conv', trainable = trainable)(x) 
    x = kl.BatchNormalization(name = conv_name_base + '_3_bn')(x) 
    
    shortcut = kl.Conv2D(filter3, (1, 1), padding = 'same', strides = strides,
               # kernel_initializer = ki.Ones(),
               kernel_initializer = ki.he_normal(),                
               bias_initializer = ki.Zeros(),
               name=conv_name_base + '_0_conv', trainable = trainable)(input_tensor)
    shortcut = kl.BatchNormalization(name = conv_name_base + '_0_bn')(shortcut) 
    
    x = kl.Add()([x, shortcut])
    x = kl.Activation('relu')(x)    
    
    return x
 
def identity_block(input_tensor, kernel_size, filters, stage, num_block, trainable=True):
    
    filter1, filter2, filter3 = filters
    conv_name_base = 'conv' + str(stage) + '_block' + num_block
    #layer1
    x = kl.Conv2D(filter1, (1, 1), name = conv_name_base +'_1_conv',
               padding = 'same',        
               # kernel_initializer = ki.Ones(),
               kernel_initializer = ki.he_normal(),                
               bias_initializer = ki.Zeros(),
               trainable = trainable)(input_tensor) 
    x = kl.BatchNormalization(name = conv_name_base + '_1_bn')(x) 
    x = kl.Activation('relu')(x)
    
    #layer2  
    x = kl.Conv2D(filter2,(kernel_size, kernel_size), padding = 'same',
               # kernel_initializer = ki.Ones(),
               kernel_initializer = ki.he_normal(),                
               bias_initializer = ki.Zeros(),
               name = conv_name_base +'b', trainable = trainable)(x) 
    x = kl.BatchNormalization(name = conv_name_base + '_2_bn')(x) 
    x = kl.Activation('relu')(x)    
    
    #layer3    
    x = kl.Conv2D(filter3,(1, 1), name = conv_name_base +'c',
               padding = 'same',
               # kernel_initializer = ki.Ones(),
               kernel_initializer = ki.he_normal(),                
               bias_initializer=ki.Zeros(),
               trainable = trainable)(x) 
    x = kl.BatchNormalization(name = conv_name_base + '_3_bn')(x) 
    x = kl.Add()([x, input_tensor])
    x = kl.Activation('relu')(x)    
    return x

class BaseModel():
    def __init__(self, input_shapes, trainable = True):
        self.input_shapes = input_shapes
        self.trainable = trainable
        input_x = kl.Input(shape = self.input_shapes)
        x = kl.Conv2D(64, (3, 3), strides = (2, 2), name = 'conv1_conv',
                   padding = 'same',      
                   # kernel_initializer = ki.Ones(),
                   kernel_initializer = ki.he_normal(),                
                   bias_initializer = ki.Zeros(),
                   trainable = self.trainable)(input_x)
        
        x = kl.BatchNormalization(name = 'conv1_bn')(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPool2D((3, 3), padding = 'SAME', strides = (2, 2))(x)

        x = conv_block(x, 3, [64, 64, 256], stage = 2, num_block = '1', strides = (1, 1), trainable = self.trainable)
        x = identity_block(x, 3, [64, 64, 256], stage = 2, num_block = '2', trainable = self.trainable)
        x = identity_block(x, 3, [64, 64, 256], stage = 2, num_block = '3', trainable = self.trainable)

        x = conv_block(x, 3, [128, 128, 512], stage = 3, num_block = '1', trainable = self.trainable)
        x = identity_block(x, 3, [128, 128, 512], stage = 3, num_block = '2', trainable = self.trainable)
        x = identity_block(x, 3, [128, 128, 512], stage = 3, num_block = '3', trainable = self.trainable)
        x = identity_block(x, 3, [128, 128, 512], stage = 3, num_block = '4', trainable = self.trainable)  

        # x = conv_block(x, 3, [256, 256, 1024], stage=4, num_block='1', trainable=trainable)
        # x = identity_block(x, 3, [256, 256, 1024], stage=4, num_block='2', trainable=trainable)
        # x = identity_block(x, 3, [256, 256, 1024], stage=4, num_block='3', trainable=trainable)
        # x = identity_block(x, 3, [256, 256, 1024], stage=4, num_block='4', trainable=trainable)    
          
        x = kl.GlobalAveragePooling2D(name = 'GAP')(x)
        # x = kl.BatchNormalization(name = 'user_gap')(x)        
        # x = kl.Dense(512, activation = 'relu',
        #           kernel_initializer = ki.he_normal(),
        #           name = 'top_Dense')(x)    
        # x = kl.BatchNormalization(name = 'user_top_bn')(x)   
        self.model = km.Model(input_x, x, name = 'base_model')

    def __call__(self, input_tensor):
        return self.model(input_tensor)
    
