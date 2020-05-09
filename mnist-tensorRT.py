#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import time
import numpy

def mnist_model():
   mnist = tf.keras.datasets.mnist
   (x_train, y_train), (x_test, y_test) = mnist.load_data()
   x_train = x_train.astype('float32')
   x_test = x_test.astype('float32')
   x_train, x_test = x_train / 255.0, x_test / 255.0

   model = tf.keras.models.Sequential()
   model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
   model.add(tf.keras.layers.Dense(512, activation='relu'))
   model.add(tf.keras.layers.Dropout(0.2))
   model.add(tf.keras.layers.Dense(10, activation='softmax'))

   model.summary()
   model.compile(optimizer='adam',
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])
   model.fit(x_train, y_train, batch_size=64,epochs=10)
   score = model.evaluate(x_test,  y_test, verbose=2)
   print('loss:',score[0])
   print('accuracy:',score[1])
   #model.save('tf_model',save_format = 'tf')
   model.save('tf_model.h5')

def trt(trt_opt):
   converter = tf.experimental.tensorrt.Converter(input_saved_model_dir='tf_model')
   converter.convert()#完成转换,但是此时没有进行优化,优化在执行推理时完成
   if trt_opt == True:
      mnist = tf.keras.datasets.mnist
      (x_train, y_train), (x_test, y_test) = mnist.load_data()
      x_test = x_test.astype('float32')
      x_test = x_test / 255.0
      def input_fn():
         yield (x_test[:1])
      converter.build(input_fn) #优化后保存
      converter.save('trt_model_opt')
   else:
      converter.save('trt_model')

def opt(model_path):
   mnist = tf.keras.datasets.mnist
   (x_train, y_train), (x_test, y_test) = mnist.load_data()
   x_test = x_test.astype('float32')
   x_test /= 255

   model_loaded = tf.saved_model.load(model_path)#读取模型
   graph_func = model_loaded.signatures['serving_default']#获取推理函数
   t=time.time()
   #output = graph_func(tf.constant(x_test))
   output = model_loaded(x_test)
   print(output[0],'\n',time.time()-t)

if __name__ == '__main__':
   print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
   mnist_model()
   #trt(True)
   #opt("tf_model")
   #opt("trt_model")
   #opt("trt_model_opt")
