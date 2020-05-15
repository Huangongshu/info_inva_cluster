# -*- coding: utf-8 -*-
#author: huan

import cv2
import numpy as np
import threading as th

paltte = [[0, 0, 255],   [255, 0, 0],     [0, 255, 0],
          [25, 25, 112],     [0, 255, 255],   [107, 142, 35],
          [250, 240, 230], [255, 255, 0],   [8, 46, 84],
          [255, 192, 203], [176, 23, 31],   [34, 139, 34],
          [255, 0, 255],   [192, 192, 192], [202, 235, 216],  
          [255, 127, 80],  [255, 99, 71],   [255, 227, 132], 
          [237, 145, 33],  [128, 42, 42],   [64, 224, 205],     
          [160, 32, 240],  [3, 168, 158],   [218, 112, 214],    
          [0, 0, 0], [255, 153, 18]]

def mix_draw(*args, **kwargs):

    orig =  kwargs.get('orig', 0)
    visible = kwargs.get('visible', 0)

    def _draw(f):    

        def _lie(x, y):
            y = np.stack([y[:, :, 0]] * 3, axis = -1)
            return np.hstack([x, y])

        def _merge(x, y):
            assert x.shape == y.shape
            return x * 0.3 + y * 0.7           
        def _handle(data):
            # data = np.transpose(data, (1, 2, 0))
            color = data.shape[-1]
            data = np.argmax(data, axis = -1)
            temp = np.zeros((data.shape[0], data.shape[1], 3))
            for i in range(color):
                temp[data == i] = paltte[i]
            return temp

        def _all(x, y):
            m1 = x * 0.3 + y * 0.7 
            return np.hstack([x, m1, y])   

        v_fun = {'all': _all, 'merge': _merge, 'lie': _lie, 0: 0}
        visible_fun = v_fun[visible]
        
        def _mix(*arg):
            temp = []          
            for i, x  in enumerate(arg[0]):
                x = _handle(x)
                if visible_fun and not isinstance(orig, bool):
                    x = visible_fun(x, orig[i])
                temp.append(x)
            return f(temp, *arg[1:])
        return _mix
    return _draw

@mix_draw()
def draw_result(x, batch, im_size = None):
    for i, data in enumerate(x):
        if im_size:
            data = cv2.resize(data, im_size)
        cv2.imwrite('./result/result{}__{}.jpg'.format(batch, i), data)
    print('\ncomplete a batch image draw')
