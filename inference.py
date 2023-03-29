#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Eleftherios Trivizakis

@github: http://github.com/trivizakis/loockme-model
"""
import gc
import cv2
import json
import numpy as np

from threading import Thread

#from math import floor
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

from tl import Transferable_Networks

def get_hypes(path="hypes"):        
        with open(path, encoding='utf-8') as file:
            hypes = json.load(file)
        return hypes
    
def standardize(X,mean,std):
    X=(X-mean)/std
    X[X>1]=1
    X[X<-1]=-1
    return X

def standardize_mch(X,mean,std,channels):
    shapes = len(X.shape)
    final_X=0
    for index in range(0, channels):
        if index == 0:
            if shapes == 3:
                final_X = np.expand_dims(standardize(X[:,:,index], mean[index], std[index]),axis=-1)
            elif shapes == 4: 
                final_X = np.expand_dims(standardize(X[:,:,:,index], mean[index], std[index]),axis=-1)
        else:
            if shapes == 3:
                temp_X = np.expand_dims(standardize(X[:,:,index], mean[index], std[index]),axis=-1)
            elif shapes == 4: 
                temp_X = np.expand_dims(standardize(X[:,:,:,index], mean[index], std[index]),axis=-1)
            final_X = np.concatenate((final_X,temp_X),axis=-1)
    return final_X

def readImage(image_path, hyperparameters):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(hyperparameters["input_shape"][0], hyperparameters["input_shape"][1]), interpolation=cv2.INTER_CUBIC)
    img = standardize_mch(img,hyperparameters["mean"],hyperparameters["std"],hyperparameters["input_shape"][2])
    img = np.expand_dims(img,axis=0)
    return img

def construct_model(hypes):
    tn = Transferable_Networks(hypes)
    weights = None
    hypes["freeze"]=-1
    
    model_input = tf.keras.layers.Input(shape=(hypes["input_shape"][0], hypes["input_shape"][1], hypes["input_shape"][2])) 
        
    model = tn.get_pretrained(input_shape=model_input,
                              model_name=hypes["model_name"], pooling="average",
                              classes=hypes["num_classes"], volumetric=False,
                              freeze_up_to=hypes["freeze"], include_top=False,
                              weights=weights)
    
    if hypes["pretrained"]:
        model_outut = model.output
        model_outut = tf.keras.layers.Flatten()(model_outut)
        for neurons in hypes["neurons"]:
            model_outut = tf.keras.layers.Dense(units=neurons,activation=hypes["activation"])(model_outut)
            model_outut = tf.keras.layers.BatchNormalization()(model_outut)
        classifier = tf.keras.layers.Dense(units=hypes["num_classes"],activation=hypes["classifier"])(model_outut)
        final_model = tf.keras.Model(inputs=model_input, outputs=classifier)
    else:
        final_model = model
    
    final_model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                        optimizer=Adam(learning_rate=hypes["learning_rate"]),
                        metrics=[hypes["metric"]],
                        run_eagerly=True)
    
    final_model.load_weights(hypes["chkp_dir"]+hypes["best_weights"])
    # print(model.summary())
    return final_model

def inference(hypes, image_path, classes):    
    #create network
    model = construct_model(hypes)
    
    # read image from storage
    image = readImage(image_path, hypes)     
    
    #predict
    prediction = model.predict(image)
    index = np.argmax(prediction)
    
    #clear session in every iteration        
    K.clear_session()
    
    #clean memmory
    del(model)
    gc.collect()
    
    return classes[index]
    


class DL_Greek_Locations:    
    def get_label(self, image_path):
        best_model = get_hypes("models/current_best_model")
        
        model_path = "models/"+best_model["architecture"]+"/"+str(best_model["neurons"][0])+"/"    
        
        hypes = get_hypes(model_path+"hypes0")
        classes = hypes["class_names"]
        
        hypes["architecture"] = best_model["architecture"]
        hypes["neurons"] = best_model["neurons"]
        hypes["best_weights"] = best_model["h5_name"]
        hypes["chkp_dir"] = model_path        
        
        return inference(hypes, image_path, classes)

    class CustomThread(Thread):
        def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
            Thread.__init__(self, group, target, name, args, kwargs)
            self._return = None

        def run(self):
            if self._target is not None:
                self._return = self._target(*self._args,
                                                    **self._kwargs)
        def join(self, *args):
            Thread.join(self, *args)
            return self._return   
    
