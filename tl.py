#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Eleftherios Trivizakis

@github: http://github.com/trivizakis/loockme-model
"""
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

#Xception: 22M, L126
from tensorflow.keras.applications.xception import Xception

#VGG: M138-143, L23-26
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19

#ResNet: M25-60
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet import ResNet152

from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import ResNet152V2

#INC3: M23
from tensorflow.keras.applications.inception_v3 import InceptionV3

#INCR2: M55
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

#Mobile: M4
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

#DenseNet: M8-20
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.applications.densenet import DenseNet201

#NASNet: M5-88
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.nasnet import NASNetLarge

def model_selection(input_shape, model_name, pooling, freeze_up_to, include_top, classes, volumetric=False, weights=None):
    #load pre-trained model
    print(input_shape)
    if not volumetric:
        if type(input_shape) == tuple:
            input_tensor = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
        else:
            input_tensor = input_shape
        
        if model_name == "xception":
            pretrained_model = Xception(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "inc3":
            pretrained_model = InceptionV3(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "vgg16":
            pretrained_model = VGG16(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes) 
        elif model_name == "vgg19":
            pretrained_model = VGG19(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes) 
        elif model_name == "incr2":
            pretrained_model = InceptionResNetV2(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "densenet201":
            pretrained_model = DenseNet201(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "densenet169":
            pretrained_model = DenseNet169(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "densenet121":
            pretrained_model = DenseNet121(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "nasnetm":
            pretrained_model = NASNetMobile(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "nasnetl":
            pretrained_model = NASNetLarge(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "mobile":
            pretrained_model = MobileNet(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "mobilev2":
            pretrained_model = MobileNetV2(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "resnet50":
            pretrained_model = ResNet50(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "resnet101":
            pretrained_model = ResNet101(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "resnet152":
            pretrained_model = ResNet152(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "resnetv250":
            pretrained_model = ResNet50V2(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "resnetv2101":
            pretrained_model = ResNet101V2(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "resnetv2152":
            pretrained_model = ResNet152V2(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        else:
            print("Model "+model_name+" does not exist.")

    else:
        print("3D models are not supported!")
    
    model = Model(inputs = pretrained_model.input,
                         outputs = pretrained_model.output)

    # i.e. freeze all convolutional InceptionV3 layers
    for layer in pretrained_model.layers[:freeze_up_to]:
        layer.trainable = False
    return model
class Transferable_Networks:    
    def __init__(self,hypes):
        self.hypes = hypes
    
    def get_pretrained(self,input_shape, model_name, pooling, freeze_up_to, include_top, classes, volumetric, weights):
        return model_selection(input_shape=input_shape,model_name=model_name, pooling=pooling, freeze_up_to=freeze_up_to, include_top=include_top, classes=classes, volumetric=volumetric, weights=weights)