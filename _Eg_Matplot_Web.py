# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:56:54 2020

@author: maria
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

"""
Matplotlib Basics

1 - Creating Plots
    Figure
        fig = plt.figurec()
    Axes
        fig.add_axes()
        a = fig.add_subplot(222)
        fig, b = plt.subplots(nrows = 3, ncols = 2)
        ax = plt.subplots(2,2) 
2 - Plotting
    1D Data
        lines = plt.plot(x, y)
        plt.scatter(x, y)
        plt.bar(xvalue, data, width, color...)
        plt.barh(yvalue, data, width, color...)
        plt.hist(x, y)
        plt.boxplot -> box and whisker plot
        plt.violinplot -> Creates a violin plot
        ax.fill(x, y, color = 'lightblue')
        ax.fill_between(x, y, color = 'yellow')
    2D Data
        fig or ax = plt.subplots()
        im = ax.imshow(img, cmap, vmin)
    Saving plots
        plt.savefig('pic.png')
4 - Customization
    Color
        plt.color(x, y, color = 'lightblue')
        plt.color(x, y, apha = 0.4)
        plt.colorbar(mappable, orientation = 'horizontal')
    Markers
        plt.plot(x, y market = '*')
        plt.plot(x, y market = '.')
    Lines
        plt.plot(x, y, linewidth = 2)
        plt.plot(x, y, ls = 'solid')
        plt.plot(x, y, ls - '--')
        plt.plot(x, y, '--', x**2, y**2, '-')
        plt.setp(lines, color = 'red', linewidth = 2)
    Text
        plt.text(1, 1, 'Example Text', style = 'italic')
        ax.annotate('some annotation', xy = (10,10))
        plt.title(r'$delta_i = 20$', fontsize = 10)
    Limits
        plt.xlim(0,7)
        other = array.copy()
        plt.ylim(-0.5, 9)
        ax.set(xlim = [0, 7], ylim = [-0.5, 9])
        ax.set_xlim(0, 7)
        plt.margins(x = 1.0, y = 1.0)
        plt.axis('equal')
    Legends/Labels
        plt.title('just a title')
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        ax.set(title = 'axis', ylabel = 'Y axis', xlabel = 'X label')
        ax.legend(loc = 'best')
    Ticks
    
"""

# Basic plotting

x = np.linspace(1,50, 10000)
y = x * 2.4
plt.plot(x, y)
plt.title('Draw a line')
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.show()
plt.close()

x = [10, 20, 30]
y1 = [20, 40, 10]
y2 = [40, 10, 30]
plt.plot(x, y1, linewidth = 3, label = 'line 1 with width 3')
plt.plot(x, y2, linewidth = 5, label = 'line 2 with width 5')
plt.title('Two or more lines on same plot with suitable legends!')
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.legend(loc = 'best')
plt.show()
plt.close()

x = [10, 20, 30]
y1 = [20, 40, 10]
y2 = [40, 10, 30]
plt.plot(x, y1, linestyle = 'dotted', linewidth = 3, label = 'line 1 with width 3')
plt.plot(x, y2, linestyle = 'dashed', linewidth = 5, label = 'line 2 with width 5')
plt.title('Two or more lines on same plot with suitable legends!')
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.legend(loc = 'best')
plt.show()

x = [1, 4, 5, 6, 7]
y1 = [2, 6, 3, 6, 3]
plt.plot(x, y1, color = 'red', linestyle = 'dashdot', linewidth = 3, marker = 'o', markerfacecolor = 'blue', markersize = 12)
plt.title('Display marker')
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.ylim(1,8)
plt.xlim(1,8)
plt.show()

x = np.linspace(1,50, 10000)
y = x * 2.4
plt.plot(x, y)
#plt.xlim(0, 100)
#plt.ylim(0, 200)
plt.axis([0, 100, 0, 200])
plt.title('Draw a line')
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.show()
plt.close()

x1 = [2, 3, 5, 6, 8]
y1 = [1, 5, 10, 18, 20]
x2 = [3, 4, 6, 7, 9]
y2 = [2, 6, 11, 20, 22]
plt.axis([0,10, 0, 30])
plt.plot(x1, y1, 'b*', x2, y2, 'ro')
plt.show()

x = np.linspace(0, 5, 25)
plt.plot(x, x, 'g--', x, x**2, 'bs', x, x**3, 'r^', markersize = 3)
plt.show()

import datetime as DT
from matplotlib.dates import date2num

data = [(DT.datetime.strptime('2016-10-03', "%Y-%m-%d"), 772.559998),
        (DT.datetime.strptime('2016-10-04', "%Y-%m-%d"), 776.429993),
        (DT.datetime.strptime('2016-10-05', "%Y-%m-%d"), 776.469971),
        (DT.datetime.strptime('2016-10-06', "%Y-%m-%d"), 776.859985),
        (DT.datetime.strptime('2016-10-07', "%Y-%m-%d"), 775.080017 )]

x = [date2num(date) for (date, value) in data]
y = [value for (date, value) in data]

fig = plt.figure()
graph = fig.add_subplot(111)
graph.plot(x,y,'r-o')
graph.set_xticks(x) # sets the xtick locations
graph.set_xticklabels([date.strftime("%Y-%m-%d") for (date, value) in data]) # sets the xtick labels
plt.xlabel('Date')
plt.ylabel('Closing value')
plt.title('Closing stock value of Alphabet Inc')
plt.minorticks_on() #required for the minor grid
plt.grid(which = 'major', linestyle ='-', linewidth = '0.5', color = 'red')
plt.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'black')
plt.tick_params(which = 'both', top = 'off', left = 'off', right = 'off', bottom = 'off')
plt.show()

fig = plt.figure()
fig.subplots_adjust(bottom = 0.020, left = 0.020, top = 0.900, right = 0.800)
plt.subplot(2, 1, 1)
plt.xticks(()),plt.yticks(())
a = plt.subplot(2, 3, 4)
plt.xticks(())
plt.yticks(())
b = plt.subplot(2, 3, 5)
plt.xticks(())
plt.yticks(())
c = plt.subplot(2, 3, 6)
plt.xticks(())
plt.yticks(())
plt.show()

# Bar charts
plang = ['Java', 'Python', 'PHP', 'JavaScript', 'C#', 'C##']
popul = [22.2, 17.6, 8.8, 8,7.7, 6.7]

plt.bar([0,1,2,3,4,5], popul, color = ['red', 'black', 'green', 'blue', 'yellow', 'lightblue'], edgecolor = ('black')) # without h it goes verticle
plt.title('Popularitu of Programming Language Worldwide, Oct 2017 compared to a year ago')
plt.xlabel('Languages')
plt.ylabel('Populatity')
plt.xticks([0,1,2,3,4,5], plang)
plt.minorticks_on()
plt.grid(which = 'major', linestyle ='-', linewidth = '0.5', color = 'red')
plt.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'black')
plt.show()

width = 0.75
plt.bar([0,1,2,3,4,5], popul, width = [ 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], color = ['blue'], edgecolor = ('black'))
plt.title('Popularitu of Programming Language Worldwide, Oct 2017 compared to a year ago')
plt.xlabel('Languages')
plt.ylabel('Populatity')
plt.ylim(0,27)
plt.xticks([0,1,2,3,4,5], plang, rotation = 90)
# Customs the subplot layout
plt.subplots_adjust(bottom = 0.4, top = 1)
plt.minorticks_on()
plt.grid(which = 'major', linestyle ='-', linewidth = '0.5', color = 'red')
plt.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'black')
for i, v in enumerate(popul):
    plt.text(i - 0.1, popul[i] + 0.1, popul[i])
plt.show()

xall = ['G1', 'G2', 'G3', 'G4', 'G5']
menmeans = [22, 30, 35, 35, 26]
womenmeans = [25, 32, 30, 35, 29]
width = 0.35
plt.bar([0,1,2,3,4], menmeans, width, color = 'g',label = 'Men')
plt.bar([0 + width, 1 + width, 2 + width , 3 + width, 4 + width] , womenmeans, width, color = 'r', label = 'Woman')
plt.xlabel('Person')
plt.xticks([0 + width, 1 + width, 2 + width , 3 + width, 4 + width] , xall)
plt.ylabel('Scores')
plt.title('Scores by person')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()

base = ['a', 'b', 'c', 'd', 'e']
level1 = [2, 4, 8, 5, 7, 6]
level2 = [4, 2, 3, 4, 2, 6]
level3 = [6, 4, 7, 4, 7, 8]
level4 = [8, 2, 6, 4, 8, 6]
level5 = [10, 2, 4, 3, 3, 2]


w = 0.1
index = np.arange(len(base) +1)
plt.bar([0,1,2,3,4,5], level1, w, color = 'b', label = 'a')
plt.bar(index + w, level2, w, color = 'g', label = 'b')
plt.bar(index + (2 * w), level3, w, color = 'r', label = 'c')
plt.bar(index + (3 * w), level4, w, color = 'lightblue', label = 'd')
plt.bar(index + (4 * w), level5, w, color = 'purple', label = 'e')
plt.xticks(np.arange(len(base)), [2, 4, 6, 8, 10], rotation = 90)
plt.legend()
plt.ylim(0, 8, 1)
plt.minorticks_on()
plt.grid(which = 'major', linestyle ='-', linewidth = '0.5', color = 'black')
plt.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'black')
plt.show()

#OR

a = np.array([[4,8,5,7,6],[2,3,4,2,6],[4,7,4,7,8],[2,6,4,8,6],[2,4,3,3,2]])
df = DataFrame(a, columns=['a','b','c','d','e'], index=[2,4,6,8,10])
df.plot(kind='bar')
# Turn on the grid
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()

# Error bars

mmeans = [22, 30, 35, 35, 26]
wmeans = [25, 32, 30, 35, 29]
mstandarddev = [4, 3, 4, 1, 5]
wstandarddev = [3, 5, 2, 3, 3]
legendb = ['Group1', 'Group2', 'Group3', 'Group4', 'Group5']
index = np.arange(len(legendb))

plt.bar(np.arange(len(legendb)), mmeans, width = 0.2, color = 'red', edgecolor = 'black', label = 'Men', yerr = [4, 3, 4, 1, 5], ecolor = 'blue')
plt.bar(legendb, wmeans, width = 0.2, bottom = mmeans, color = 'green', edgecolor = 'black', label = 'Women', yerr = [3, 5, 2, 3, 3], ecolor = 'blue')
plt.xlabel('Groups')
plt.ylabel('Scores')
plt.legend()
plt.title('Scores by group and gender')
plt.show()

"""
DO IT
data = [[ 3.40022085, 7.70632498, 6.4097905, 10.51648577, 7.5330039, 7.1123587, 12.77792868, 3.44773477],
[ 11.24811149, 5.03778215, 6.65808464, 12.32220677, 7.45964195, 6.79685302, 7.24578743, 3.69371847],
[ 3.94253354, 4.74763549, 11.73529246, 4.6465543, 12.9952182, 4.63832778, 11.16849999, 8.56883433],
[ 4.24409799, 12.71746612, 11.3772169, 9.00514257, 10.47084185, 10.97567589, 3.98287652, 8.80552122]]

w = 0.8
xvalues = [0, 1, 2, 3, 4, 5, 6, 7]
leftsum = [0, 0, 0, 0, 0, 0, 0, 0]
percentages = (np.random.randint(5,20, (8, 4)))
color = ['red', 'green', 'white', 'purple']
for i in [0,1,2,3]:
    plt.barh(xvalues, data[i], left = leftsum, color = color[i], edgecolor = 'black')
    for j in list(range(8)):
        plt.text(leftsum[j] + (data[i][j] * 0.5), xvalues[j], percentages[i][j], ha='center')
    for j in list(range(8)):
        leftsum[j] = leftsum[j] + data[i][j]

plt.yticks(xvalues, ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8'])
plt.xlabel('Scores')
plt.xlim(0, 40, 5)
plt.show()

DO IT
"""




# From CHEAT SHEET

# 1 - 1D data
# x = np.linspace(1, 10, 100)
# y = np.cos(x)
# z = np.sin(x)

# 2 - 2D data
# data = 2 * np.random.random((10, 10))
# data2 = 3 * np.random.random((10, 10))
# Y, X = np.mgrid[-3:3:100j, -3:3:100j]
# V = 1 + X - Y ** 2

# from Data_Preparation_ELEXON_Exercise import data1 