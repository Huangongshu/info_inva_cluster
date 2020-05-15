# -*- coding: utf-8 -*-
#author: huan

import keras.layers as kl
import keras.models as km
import keras.initializers as ki
import tensorflow as tf
from keras.optimizers import Adam
from seg_generate import Gen
import keras.callbacks as kc
from seg_draw import draw_result, mix_draw
from seg_model import BaseModel
import numpy as np
import pickle
from utils import *
import os
import pandas as pd
import time
from functools import partial

if __name__ == '__main__':

    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--lamb_b', default = 1.5,
                       help = 'the optional entropy coefficient of the b model', type = float)
    parse.add_argument('--lamb_a', default = 1.0,
                       help = 'the optional entropy coefficient of the a model', type = float)    
    parse.add_argument('--i_e', default = 1,
                       help = 'the initial epoch of the train', type = int, dest = 'init_epoch')
    parse.add_argument('--e_n', default = 1000,
                       help = 'the epoch number of the train', type = int, dest = 'epoch_num')    

    parse.add_argument('--od', default = './output',
                       help = 'the result output path of the train', type = str, dest = 'out_dir')  
    parse.add_argument('--t_b', default = 20,
                       help = 'the batch size of the train', type = int, dest = 'train_batch')  
    parse.add_argument('--test_b', default = 20, 
                       help = 'visiable batch', type = int, dest = 'test_batch')  
    parse.add_argument('--test_n', default = 100, 
                       help = 'visiable num', type = int, dest = 'test_num')        

    parse.add_argument('--config_save_path', default = './config',
                       help = 'the config output path of the train', type = str, dest = 'config_out')
    parse.add_argument('--bal', default = True,
                       help = 'balance sample in the model training', dest = 'balance', action = 'store_false')
    parse.add_argument('--lr', default = 1e-3, type = float, help = 'the learning rate')
    parse.add_argument('--mp', default = './model',
                       help = 'save model in the path', type = str, dest = 'm_s_path')

    parse.add_argument('--ha', default = 1,
                       help = 'the number of the head A', type = int, dest = 'head_a_num')
    parse.add_argument('--df', default = '/home/ouzhuang/kaggle_data/trainLabels.csv',
                       help = 'used the dataset csv', type = str, dest = 'df')
    parse.add_argument('--r', default = '/home/zhangyun/EyeImagepreprocessed/dataset_2/afterResize/',
                       help = 'the data save path', type = str, dest = 'root')
    parse.add_argument('--be', default = 2,
                       help = 'head B epoch', type = int, dest = 'b_epoch_num')

    parse.add_argument('--ae', default = 1,
                       help = 'heat A epoch', type = int, dest = 'a_epoch_num')
    parse.add_argument('--i_h' , default = 400,
                       help = 'the image hight', type = int, dest = 'h')
    parse.add_argument('--i_w' , default = 400,
                       help = 'the image weight', type = int, dest = 'w')  
    parse.add_argument('--s_u', default = True, 
                       help = 'use the sobel', action = 'store_false', dest = 'sobel') 

    parse.add_argument('--s_c', default = 2, 
                       help = 'the output channel  when used the sobel,i.e. 2 or 4',
                       type = int, dest = 'channel')  

    parse.add_argument('--is_g', default = False, 
                       help = 'whether the input data is a gray picture,i.e. 2 or 4',
                       action = 'store_true', dest = 'is_gray') 

    parse.add_argument('--a_k', default = 15, 
                       help = 'b cluster number', type = int, dest = 'cluster_a') 
    parse.add_argument('--b_k', default = 3, 
                       help = 'b cluster number', type = int, dest = 'cluster_b')   
    parse.add_argument('--f_t', default = False, 
                       help = 'model fine tune', action = 'store_true', dest = 'fine_tune')   
    parse.add_argument('--d', default = 'kaggle_data',
                       help = 'dataset name', type = str, dest = 'dataset')

    parse.add_argument('--adj_t_d', default = 10,
                       help = 'adjacent regions size', dest = 'adjacent_right_and_left')
    parse.add_argument('--adj_l_r', default = 10,
                       help = 'adjacent regions size', dest = 'adjacent_top_and_down')
    parse.add_argument('--train_w', default = 10, 
                       help = 'the multiprocessing number or the threading number', type = int, dest = 'train_worker')  
    parse.add_argument('--test_w', default = 5, 
                       help = 'the multiprocessing number or the threading number', type = int, dest = 'test_worker')

    parse.add_argument('--project', default = '0',
                       help = 'the test project name', type = str)
    parse.add_argument('--test_output', default = './result',
                       help = 'the test result output path', type = str)

    parse.add_argument('--aj_lr', default = False,
                       help = 'ajust the learning rate', action = 'store_true', dest = 'ajust_lr') 

    args = parse.parse_args()

    args.laster = args.init_epoch
    if args.fine_tune:
        lr = args.lr
        with open(os.path.join(args.config_out, args.project, "config.pickle"), 'rb') as f:
            config_args = pickle.load(f)  
        for k, v in vars(config_args).items():
            if k in fune_tune_args:
                args[k] = hasattr(args, k) and v
        args.lr = lr  
    args.input_sz = (args.h, args.w, args.channel)   

    # make directory   
    check_path(os.path.join(args.m_s_path, args.project))
    check_path(os.path.join(args.config_out, args.project))
    check_path(os.path.join(args.out_dir, args.project))    
    check_path(os.path.join(args.test_output, args.project))

    #preprocessing function
    train_prepro = partial(preprocess_image, is_sobel = args.sobel,
                           channel = args.channel, is_gray = args.is_gray)
    test_prepro = partial(test_preprocess, is_sobel = args.sobel,
                          channel = args.channel, is_gray = args.is_gray, orig = True)

    #build
    input_x = kl.Input(shape = args.input_sz, name = 'input1')
    input_gx = kl.Input(shape = args.input_sz, name = 'input2')

    base_model = BaseModel(args.input_sz)

    head_a = [kl.Conv2D(args.cluster_a,
                  kernel_size = 1,
                  strides = 1,
                  activation = 'softmax',
                  kernel_initializer = ki.he_normal(),  
                  # bias_initializer = ki.Zeros(),          
                  name = 'a' + str(i)) for i in  range(args.head_a_num)]

    head_b = kl.Conv2D(args.cluster_b,
                  kernel_size = 1,
                  strides = 1,
                  activation = 'softmax',
                  kernel_initializer = ki.he_normal(),
                  # bias_initializer = ki.Zeros(),
                  name = 'b')
             
    x = base_model(input_x)
    gx = base_model(input_gx)
    head_a_loss = []
    train_a_model = []
    
    for m_i, a in enumerate(head_a):
        head_a_loss.append(kl.Lambda(lambda x: seg_loss(*x,
                           paddings = (args.adjacent_top_and_down, args.adjacent_right_and_left),
                           lamb = args.lamb_a), name = 'head_a_loss' + str(m_i))([a(x), a(gx)]))
        train_a_model.append(km.Model(inputs = [input_x, input_gx], outputs = head_a_loss[m_i]))
        train_a_model[m_i].add_loss(train_a_model[m_i].get_layer('head_a_loss' + str(m_i)).output)
        train_a_model[m_i].compile(optimizer = Adam(args.lr), loss = [None])

    head_b_loss = kl.Lambda(lambda x: seg_loss(*x,
                                              paddings = (args.adjacent_top_and_down, args.adjacent_right_and_left),
                                              lamb = args.lamb_b), name = 'head_b_loss')([head_b(x), head_b(gx)])
                                              
    model_b = km.Model(inputs = input_x, outputs = head_b(x), name = 'output')
    train_b_model = km.Model(inputs = [input_x, input_gx], outputs = head_b_loss)  

    head_b_loss_value = train_b_model.get_layer('head_b_loss').output
    train_b_model.add_loss(head_b_loss_value)

    train_b_model.compile(optimizer = Adam(args.lr), loss = [None])
    model_b.compile(optimizer = Adam(args.lr), loss = 'categorical_crossentropy')

    # dataset
    if args.dataset == 'mnist':
        (x_train, y_train),(x_test, y_test) = load_mnist(path = 'mnist.npz')
        x_train = np.concatenate([x_train, x_test])
        y_train = np.concatenate([y_train, y_test])       
    elif args.dataset == 'kaggle_data':
        df = pd.read_csv(args.df)
        df = df.astype({'image': 'str', 'level': 'int'})        
        df = df[(df['level'] == 1) | (df['level'] == 4)]
        # df = df[df['level'] == 1]
        df['image'] = df['image'].apply(lambda x: ''.join((args.root, x, '.jpg')))
        tra = lambda x: np.array(list(x))
        x_train, y_train = tra(df['image']), tra(df['level'])
        x_test, y_test = tra(df['image']), tra(df['level'])

    if args.fine_tune:
        for i in range(args.head_a_num):
            train_a_model[i].load_weights('{}/{}/{}_{}_{}{}'.format(args.m_s_path, args.project,
                             'model_a', str(i), str(args.laster), '.h5'), by_name = True)
        train_b_model.load_weights('{}/{}/{}_{}'.format(args.m_s_path, args.project,
                             'model_b', str(args.laster), '.h5'), by_name = True)
        model_b.load_weights('{}/{}/{}_{}'.format(args.m_s_path, args.project,
                             'model_b', str(args.laster), '.h5'), by_name = True)
        args.init_epoch = args.laster + 1

        if args.ajust_lr:
            train_a_model = [lr_ajust(m, args.lr) for m in train_a_model]
            train_b_model = lr_ajust(train_b_model, args.lr)

    for epoch in range(args.init_epoch, args.epoch_num):
        print('start head b train\n')
        start_time = time.time()
        train_dataset = Gen(x_train, y_train,
                            batch_size = args.train_batch,
                            preprocess_fun = train_prepro,
                            worker = args.train_worker,
                            balance = args.balance,
                            shuffle = True,
                            mode = 'categorical',
                            state = 'train',
                            target_size = args.input_sz[:2],
                            dataset = args.dataset,
                            )

        # train_b_model.fit_generator(train_dataset,
        #               epochs = args.b_epoch_num,
        #               steps_per_epoch = len(train_dataset),
        #               # workers = 4,
        #               max_queue_size = 10,
        #               use_multiprocessing = True,
        #               )

        b_pro = ProgressBar(args.b_epoch_num, len(train_dataset))
        for i in range(args.b_epoch_num):
            for x, y in train_dataset:
                loss = train_b_model.train_on_batch(x = x, y = None)
                b_pro.update(float(loss))
        with open(args.out_dir + '/b_loss.txt', 'a') as f:
            f.write('epoch: {} loss {}\n'.format(epoch, b_pro.loss))

        train_dataset = Gen(x_train,
                            y_train,
                            batch_size = args.train_batch,
                            preprocess_fun = train_prepro,
                            worker = args.train_worker,
                            balance = args.balance,
                            shuffle = True,
                            mode = 'categorical',
                            target_size = args.input_sz[:2],
                            dataset = args.dataset,
                            state = 'train',
                            )

        print('start head a train\n')
        a_pro = ProgressBar(args.a_epoch_num, len(train_dataset))
        for i in range(args.a_epoch_num):        
            for x, y in train_dataset:
                loss = 0
                for num in range(args.head_a_num):
                    loss += train_a_model[num].train_on_batch(x = x, y = None)
                a_pro.update(loss / args.head_a_num)

        with open(args.out_dir + '/a_loss.txt', 'a') as f:
            f.write('epoch: {} loss {}\n'.format(epoch, a_pro.loss))

        #eval        
        test_dataset = Gen(x_test[: args.test_num], y_test[: args.test_num],
                           batch_size = args.test_batch,
                           preprocess_fun = test_prepro,
                           worker = args.test_worker,
                           # shuffle = True,
                           mode = 'None',
                           sample_ratio = 1,
                           target_size = args.input_sz[:2],
                           dataset = args.dataset,
                           state = 'train',
                           )    

        if epoch % 1 == 0:
            y_pred = []
            im = []
            for x, orig in test_dataset:
                y_pred.append(model_b.predict(x))
                im.append(orig)
            y_pred = np.concatenate(y_pred)
            im = np.concatenate(im)

            @mix_draw(orig = im, visible = 'merge')
            def draw(x, batch):
                for i, data in enumerate(x):
                    cv2.imwrite('{}/{}/result{}__{}.jpg'.format(args.test_output,
                                          args.project, batch, i), data)       
            draw(y_pred, 0)  

        print('complete {} epoch, and used time :{}\n'.format(epoch, time.time() - start_time))

        #save
        if epoch % 5 == 0:
            args.laster = epoch
            for i, m in enumerate(train_a_model):
                m.save('{}/{}/{}_{}_{}{}'.format(args.m_s_path, args.project,
                             'model_a', str(i), str(args.laster), '.h5'))
            model_b.save('{}/{}/{}_{}'.format(args.m_s_path, args.project, 'model_b', str(args.laster), '.h5'))

            with open(os.path.join(args.config_out, args.project, "config.pickle"), 'wb') as f:
                pickle.dump(args, f)
            with open(os.path.join(args.config_out, args.project, 'config.txt'), 'w') as f:
                confi = vars(args)
                for k,v in confi.items():
                    f.write('{}:{}\n'.format(k, v))
