# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 20:11:57 2016

@author: Ravius
"""


import pandas as pd
import numpy as np

path = "data.dump.csv"


#check structure


dct = {}
 
def analysis(lst):
    for i in lst:
        if i in dct:
            dct[i] += 1
        else:
            dct[i] = 1  
 


check = open(path,'r', encoding="utf8");
#check.readline()

df =pd.DataFrame.from_records(list(map(lambda x: (int(x[0]),x[1][:-1],x[2], x[3], ) ,map( 
                    lambda x: x.split(', "', 4), check.readlines()))), columns=["timestamp", "ip address", "login", "user-agent"])





analysis(df['ip address'])

l = lambda x: x[1]
p = sorted(dct.items(), key=l, reverse=True) 

print(p[:20])

print(df[['ip address','user-agent', 'login']][df['ip address'] == '175.105.225.72"'])
#14.240.154.62"


with open('log.txt', 'a') as logfile:
    logfile.write( omg)
    
    
    'Mozilla/5.0 (X11 Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2690.0 Safari/537.36'
    
    
    
    