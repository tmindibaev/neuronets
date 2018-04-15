# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 13:23:19 2017

@author: timofey
"""
import numpy as np
from os import listdir

dt = np.dtype([('name', np.str_, 32), ('value', np.int64, (1,))])


def parse(file):
    a = np.genfromtxt(file, 
                      dtype = dt,
                      delimiter=':')
    
    return a


files = listdir()

#inp = parse()
inpList = []

for i in files:    
    inpList.append(parse(i))
    
print(inpList)