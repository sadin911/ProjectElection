#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:07:04 2020

@author: trainai1
"""

start_x = [759, 1355]
start_y = [443, 1695]
x_count = 20
y_count = 57

crop_col = []
crop_row = []

for i in range(x_count):
    crop_col.append(int( ((x_count-1-i)/(x_count-1))*  start_x[0] + (i/(x_count-1))* start_x[1]) )

print(crop_col)

for i in range(y_count):
    crop_row.append(int( ((y_count-1-i)/(y_count-1))*  start_y[0] + (i/(y_count-1))* start_y[1] ))
print(crop_row)
