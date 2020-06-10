# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:57:37 2020

@author: maria
"""

import pandas as pd

def f (x) :
  return x**6/6 - 3*x**4 - 2*x**3/3 + 27*x**2/2 + 18*x - 30

def d_f (x) :
  return (x**5) - (12 * x ** 3) - (2 * x **2) + (27 * x) + 18

x = -4.0

d = {"x" : [x], "f(x)": [f(x)]}
for i in range(0, 20):
  x = x - f(x) / d_f(x)
  d["x"].append(x)
  d["f(x)"].append(f(x))

pd.DataFrame(d, columns=['x', 'f(x)'])
