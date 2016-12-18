# -*- coding: utf-8 -*-
"""
@author: Pokrovsky
"""



import pandas as pd
import numpy as np


def validate(str):
    summ = 0;
    if (len(str)%2) == 0:
        st = 2;
    else:
        st = 1;
    for i in str:
        x = int(i);
        x = x*st;
        if x > 9:
            x = x-9;
        st = st%2 +1;
        summ = summ + x;
    res = summ%10;
    if res == 0 :
        return 0;
    else:
        return 1;
             
path = "data.dump.csv"

check = open(path,'r');
#check structure of the file 
'''
print(check.readline())
print(check.readline())
'''

df =pd.DataFrame.from_records(list(map(lambda x: (x[0][1:-1],x[1][1:-1],x[2][1:-1], x[3][1:-1],x[4][1:-1], x[5][1:-2],validate(x[5][1:-2])) ,map( 
                    lambda x: x.split("; ", 6), check.readlines()))), columns=["username", "name", "address", "password", "card type", "card number","validated"])

df.sort_values(["validated", "username", 'password'], ascending = [0,0,1], inplace  = True)
print(df[42:43])
