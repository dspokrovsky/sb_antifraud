# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 18:52:09 2016

@author: Ravius
"""

import pandas as pd
import numpy as np

path = "data.dump.csv"

#check structure
'''
check = open(path,'r');
print(check.readline())
'''
#print(check.readline())


df =pd.DataFrame.from_records(list(map(lambda x: (int(x[0]),x[1][1:-1],x[2][1:-1], x[3][1:-1],int(x[4][1:-2]), ) ,map( 
                    lambda x: x.split(", ", 6), check.readlines()))), columns=["timestamp", "username", "username phone num", "payment phone num", "amount"])

'''#dict about the number of transactions'''
l = {}
y = 0
for i in df['username']:
    try:
        if i not in l:
            l[i] = 1;
            print(i)
        else:
            l[i]= l[i]+1;
            print(l[i])
    except:
        y = y+1;
'''#find the biggest one'''
z2 =0;
z= 0;
s ='';
s2 ='';
for i in l:
    if z < l.get(i):
        z2 = z;
        s2 = s;
        z= l.get(i);
        s =i;
        
#check the values
print(df[['payment phone num', 'timestamp']][df['username']== 'SFer682'])
        
#result !
        #SFer682
        #+7 920 666 87 15