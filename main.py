#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:20:38 2023

@author: eosforos
"""

import gc
from inference import DL_Greek_Locations

if __name__ == '__main__':
    #get image path from post
    image_path=""
    
    dlgl = DL_Greek_Locations()
    prediction_thread = dlgl.CustomThread(target=dlgl.get_label, args=(image_path, ))
    prediction_thread.start()
    label = prediction_thread.join()
    
    del(prediction_thread)
    gc.collect()
    
    #send response label 
