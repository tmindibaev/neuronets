# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 13:23:19 2017

@author: time_of_fate
"""
import numpy as np
import csv

from os import listdir

dt = np.dtype([('name', np.str_, 32), ('value', np.int64, (1,))])


def parse(file):
    a = np.genfromtxt(file, 
                      dtype = dt,
                      delimiter=':')
    
    return a


files = listdir(".")

#inp = parse()
inpList = []

for i in range(0, len(files)-1):    
    inpList.append(parse(files[i]))

globList = []

for iter in range(0,len(inpList)):
    newList = []
    for i in range(4,17):    
        newList.append(int(inpList[iter][i][1]))
    globList.append(newList)

    
'''print(globList)'''

arr = np.array(globList, dtype = np.int64)

fill = open('train.txt', 'w')

for iter in arr:
    iter.tofile(fill, sep = ',', format = "%s")
    print('', file = fill)
fill.close()