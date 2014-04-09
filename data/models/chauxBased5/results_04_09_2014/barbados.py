#!/usr/bin/env python

"""

Manual tweaking of compare results to make graphs more presentable.  results are hand-copied from the various _comp.txt files in this folder.  
"""

import sys
import os
import matplotlib.cm as cm
from teHmm.common import runShellCommand, getLogLevelString, getLocalTempPath
from teHmm.parameterAnalysis import plotPoints2d

assert len(sys.argv) == 2

outFile = sys.argv[1]

# (results files below generated in command lines in ./compare.txt

iPoints = []
bPoints = []
stateNames = []

# from mcSegment1_flat_comp.txt
iPoints.append((0.6666666666666666, 0.9333333333333333))
bPoints.append((0.86000506551848244, 0.97487571335747714))
stateNames.append("HMM LTR")

iPoints.append((0.7647058823529411, 0.625))
bPoints.append((0.7220428578597835, 0.86557975374002327))
stateNames.append("HMM NON-LTR")

# from allChaux_flat_comp.txt
iPoints.append((0.07236842105263158, 0.36666666666666664))
bPoints.append((0.78475077606658028, 0.91806903638186099))
stateNames.append("RM-CHAUX LTR")

iPoints.append((0.5454545454545454, 0.5625))
bPoints.append((0.77846968008912942, 0.78468295030682234))
stateNames.append("RM-CHAUX NON-LTR")

# from ltrChaux_flat_comp.txt
iPoints.append((1.0, 0.5666666666666667))
bPoints.append((0.99981713115353665, 0.68847496385991247))
stateNames.append("LTRFINDER LTR")


titles = ["Base-level Accuracy", "Element-level Accuracy (th=0.8)"]
distList = [ bPoints, iPoints ]
markerList = [ "o" , "^" ]

rgbs = [cm.gist_rainbow_r(float(i) / float(len(stateNames))) for i in xrange(len(stateNames))]
for i in xrange(len(rgbs)):
    rgbs[i] = list(rgbs[i])
    rgbs[i][3] = 0.7

rgbs = [rgbs[0], rgbs[0], rgbs[2], rgbs[2], rgbs[1]]    

plotPoints2d(distList, titles, stateNames, outFile, xRange=(-0.1,1.1),
                 yRange=(-0.1, 1.4), ptSize=75, xLabel="Precision",
                 yLabel="Recall", cols=2, width=10, rowHeight=5,
                 markerList = markerList, rgbs = rgbs)
