#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Eleftherios Trivizakis

@github: http://github.com/trivizakis/loockme-model
"""
import gc
from threading import Thread
import DL_Greek_Locations

if __name__ == '__main__':
    #get image path from post
    image_path=""
    
    dlgl = DL_Greek_Locations()
    prediction_thread = Thread(target=dlgl.get_label, args=(image_path,))
    prediction_thread.start()
    prediction_thread.join()
    
    label = prediction_thread.value
    
    del(prediction_thread)
    gc.collect()
    
    #send response label 