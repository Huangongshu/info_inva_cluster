# -*- coding: utf-8 -*-
#author: huan

import keras.layers as kl
import keras.models as km
import keras.initializers as ki
import tensorflow as tf
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from user_generate import Gen
import matplotlib.pyplot as plt
import keras.callbacks as kc
from mnist_draw import convex_combo
from test_model import BaseModel
import numpy as np
from PIL import Image
from data_enhancement import *
from scipy.optimize import linear_sum_assignment
import sys
import keras.backend as K

inputs_shapes = (28, 28, 1)
cluster_k = 10
lr = 0.005
BATCH_SIZE = 256
epoch_num = 1000
heat_a_num = 5

class Log(kc.Callback):
    def on_epoch_end(self, epoch, logs):
        with open('train_info.txt', 'a') as f:
            f.write('\nepoch: {}   train loss:{:.5f}   val_loss:{:.5f}  train acc:{:.5f}   val acc:{:.5f}'.format(
                        epoch, logs.get('loss'), logs.get('val_loss'), logs.get('accuracy'), logs.get('val_accuracy')))

def lr_ajust(model, decay_ratio = 0.1):
    lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, lr * decay_ratio)

def evaluate(y_true, y_pred, c = 10):
  y_pred = np.array(y_pred)
  y_true = np.array(y_true).astype(np.int64)
  w = np.zeros((c, c), dtype=np.int64)
    # Confusion matrix.
  for i in range(y_pred.size):
      w[y_pred[i], y_true[i]] += 1
  ind = linear_sum_assignment(-w) 
  w_ind , h_ind = ind
  print(w, '\n')
  acc = np.sum([w[i, j] for i, j in zip(w_ind, h_ind)]) / np.sum(w)
  return acc

def preprocess_image(im):
    #im
    imx = Image.fromarray(im.copy())
    # imx = random_flip(imx)     
    if np.random.randint(0, 2):    
        imx = random_crop(imx, crop_range = [20, 20])
    else:
        imx = random_centre_crop(imx, crop_range = 0.2) 
    imx = np.asarray(imx)
    imx = np.expand_dims(imx, axis = 0)        
    imx = np.concatenate([imx] * 3, axis = 0)
    #imgx
    # imgx = random_flip(imgx)
    # imgx = random_zoom(imgx, 0.1)    
    # imgx = random_shift(imgx, shift_range = 0.2)
    img = Image.fromarray(im.copy())
    imgx = []
    for s in [16, 20, 24]:
        temp = random_crop(img, crop_range = [s, s]) 
        temp = random_rotation(temp, rotate_range = 180)  
        temp = random_color(temp, color_range = 0.2,
                            contrast_range = 0.2,
                            brightness_range = 1.5,
                            Sharpness = 3)
        temp = np.expand_dims(np.asarray(temp), axis = 0)
        imgx.append(temp) 
    imgx = np.concatenate(imgx, axis = 0) 
    imx = (imx - imx.mean()) / (imx.std() + 1e-07)    
    imgx = (imgx - imgx.mean()) / (imgx.std() + 1e-07)    
    return np.expand_dims(imx, axis = -1), np.expand_dims(imgx, axis = -1)  

def test_preprocess(im):
    im = Image.fromarray(im)
    # im = random_centre_crop(im, crop_range = 0.2)
    im = np.asarray(im)     
    im = (im - im.mean()) / im.std()    
    return np.expand_dims(im, axis = -1) 

def load_mnist(path):
  with np.load(path) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    y_train = np.expand_dims(y_train, axis = 1)
    y_test = np.expand_dims(y_test, axis = 1)
    return (x_train, y_train), (x_test, y_test)

def iic_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
      k = x_out.shape[-1]
      # joint probability 通过矩阵乘，得到两个模型各类别预测值的乘积。同类别乘积为共同特征。
      p = tf.reduce_sum(tf.expand_dims(x_out, 2) * tf.expand_dims(x_tf_out, 1), 0)
      p = (p + tf.transpose(p)) / 2 # symmetry
      p = tf.clip_by_value(p, EPS, 1e9)
      p /= tf.reduce_sum(p) # normalize
      pi = tf.broadcast_to(tf.reshape(tf.reduce_sum(p, axis=0), (k, 1)), (k, k))
      pj = tf.broadcast_to(tf.reshape(tf.reduce_sum(p, axis=1), (1, k)), (k, k))
      iloss = -tf.reduce_sum(p * (tf.math.log(p) - lamb * tf.math.log(pi) - lamb * tf.math.log(pj)))
      return iloss

if __name__ == '__main__':
    input_x = kl.Input(shape = inputs_shapes, name = 'input1')
    input_gx = kl.Input(shape = inputs_shapes, name = 'input2')

    base_model = BaseModel(inputs_shapes)

    #head A
    head_a = [kl.Dense(cluster_k * 5,
                  activation = 'softmax',
                  kernel_initializer = ki.he_normal(),  
                  # bias_initializer = ki.Zeros(),          
                  name = 'a' + str(i)) for i in  range(heat_a_num)]

    #head B
    head_b = kl.Dense(cluster_k,
                  activation = 'softmax',
                  kernel_initializer = ki.he_normal(),
                  # bias_initializer = ki.Zeros(),
                  name = 'b')
             
    x = base_model(input_x)
    gx = base_model(input_gx)
    head_a_loss = []
    train_a_model = []
    for m_i, a in enumerate(head_a):
        head_a_loss.append(kl.Lambda(lambda x: iic_loss(*x), name = 'head_a_loss' + str(m_i))([a(x), a(gx)]))
        train_a_model.append(km.Model(inputs = [input_x, input_gx],
                             outputs = head_a_loss[m_i], name =''.join(('model_a', str(m_i), '_'))))
        train_a_model[m_i].add_loss(train_a_model[m_i].get_layer('head_a_loss' + str(m_i)).output)
        train_a_model[m_i].compile(optimizer = Adam(lr), loss = [None])

    # train_a_model = km.Model(inputs = [input_x, input_gx], outputs = [m for m in head_a_loss])
    head_b_loss = kl.Lambda(lambda x: iic_loss(*x), name = 'head_b_loss')([head_b(x), head_b(gx)])
    model_b = km.Model(inputs = input_x, outputs = head_b(x), name = 'model_b_')
    train_b_model = km.Model(inputs = [input_x, input_gx], outputs = head_b_loss)  

    head_b_loss_value = train_b_model.get_layer('head_b_loss').output
    train_b_model.add_loss(head_b_loss_value)

    train_b_model.compile(optimizer = Adam(lr), loss = [None])
    model_b.compile(optimizer = Adam(lr), loss = 'categorical_crossentropy')

    (x_train, y_train),(x_test, y_test) = load_mnist(path = 'mnist.npz')
    x_train = np.concatenate([x_train, x_test])
    y_train = np.concatenate([y_train, y_test])       

    best_acc = 0
    no_improve_epoch = 0
    for epoch in range(1, epoch_num):
        train_dataset = Gen(x_train, y_train, batch_size = BATCH_SIZE, preprocess_fun = preprocess_image, worker = 0,
                          balance = True,
                         shuffle = True, mode = 'categorical', state = 'train')
        train_b_model.fit_generator(train_dataset,
                      epochs = 2,
                      steps_per_epoch = len(train_dataset),
                      # workers = 4,
                      max_queue_size = 10,
                      # use_multiprocessing = True,
                      )
        # i = 0
        # for x, y in train_dataset:
        #     loss = train_a_model.train_on_batch(x = x, y = None)
            # print('batch a loss', loss)

        # print(train_a_model.losses)
        train_dataset = Gen(x_train, y_train, batch_size = BATCH_SIZE, preprocess_fun = preprocess_image, worker = 0,
                          balance = True,
                         shuffle = True, mode = 'categorical', state = 'train')
     
        # train_a_model.fit_generator(train_dataset,
        #               epochs = 1,
        #               steps_per_epoch = len(train_dataset),
        #               # workers = 8,
        #               max_queue_size = 10,
        #               # use_multiprocessing = True,
        #               )

        head_a_loss = []
        for x, y in train_dataset:
            loss = 0
            for i in range(5):
                loss += train_a_model[i].train_on_batch(x = x, y = None)
            head_a_loss.append(loss / 5)
            print('\rhead a loss: {:.4f}'.format(np.mean(head_a_loss)), end = '')
        #     # print('batch b loss', loss)
        # print(train_b_model.losses)
        print('\ncomplete {} epoch'.format(epoch))

        #eval        
        plt.figure(figsize=(3,3), dpi = 300)
        ax = plt.gca()
        
        test_dataset = Gen(x_test, y_test,
                           batch_size = BATCH_SIZE,
                           preprocess_fun = test_preprocess,
                           worker = 2,
                           # shuffle = True,
                           mode = 'categorical',
                           sample_ratio = 1,
                            state = 'val')    
        # y_pred = model_b.predict_generator(test_dataset, steps = len(test_dataset),
        #                                    verbose = 1,
        #                                   # workers = 4,
        #                                   # use_multiprocessing = True,
        #                                   max_queue_size = 1,
        #                                   )
        y_true = []
        y_pred = []
        for x, y in test_dataset:
            y_pred.append(model_b.predict(x))
            y_true.append(y)

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        y_true = np.argmax(y_true, axis = 1)
        convex_combo(y_pred, y_true, ax, 'pointcloud/{}.png'.format(epoch))   
        y_pred = np.argmax(y_pred, axis = 1)
        acc = evaluate(y_true, y_pred)
        with open('test_acc.txt', 'a') as f:
            f.write('\nepoch: {}    accuracy: {}'.format(epoch, acc))
        print('\nepoch: {}    accuracy: {}'.format(epoch, acc)) 
        if acc > best_acc:
            best_acc = acc
            model_b.save(''.join(('./model/model_b_', str(epoch), '.h5')))
            for m in train_a_model:
              m.save(''.join(('./model/', m.name, str(epoch), '.h5')))
            no_improve_epoch = 0
        else:
            no_improve_epoch += 1
            print('no improve_epoch:', no_improve_epoch) 
        if no_improve_epoch % 8 == 0 and no_improve_epoch != 0:
            map(lr_ajust, train_a_model)
            lr_ajust(train_b_model)
        elif no_improve_epoch >20:
            break
        if best_acc < 0.7:
            no_improve_epoch = 0 
        print('np improve epoch:', no_improve_epoch) 
