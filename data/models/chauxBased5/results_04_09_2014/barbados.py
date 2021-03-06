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
iPoints.append((0.6136363636363636, 0.9333333333333333))
bPoints.append((0.82905799863794649, 0.97495630449345949))
stateNames.append("HMM LTR")

iPoints.append((0.8269230769230769, 0.671875))
bPoints.append((0.86083683162853031, 0.87590743191753906))
stateNames.append("HMM NON-LTR")

# from ltrChaux_flat_comp.txt
iPoints.append((1.0, 0.5666666666666667))
bPoints.append((0.99981713115353665, 0.68847496385991247))
stateNames.append("LTRFINDER LTR")

# from allChaux_flat_comp.txt
iPoints.append((0.5263157894736842, 0.36666666666666664))
bPoints.append((0.78475077606658028, 0.91806903638186099))
stateNames.append("RM-CHX LTR")

iPoints.append((0.8484848484848485, 0.5625))
bPoints.append((0.77846968008912942, 0.78468295030682234))
stateNames.append("RM-CHX NON-LTR")

# from hollister_flat_comp.txt
iPoints.append((0.3584905660377358, 0.4666666666666667))
bPoints.append((0.71012296853666157, 0.83568474762382072))
stateNames.append("RM-HOL LTR")

iPoints.append((0.6372549019607843, 0.765625))
bPoints.append((0.71192863715047638, 0.91547346889664305))
stateNames.append("RM-HOL NON-LTR")


titles = ["Base-level Accuracy", "Element-level Accuracy (th=0.8)"]
distList = [ bPoints, iPoints ]
markerList = [ "o" , "^" , "o", "o", "^", "o", "^" ]

rgbs = [cm.gist_rainbow_r(float(i) / float(len(stateNames))) for i in xrange(len(stateNames))]
for i in xrange(len(rgbs)):
    rgbs[i] = list(rgbs[i])
    rgbs[i][3] = 0.7

prgbs = [rgbs[0], rgbs[0], rgbs[5], rgbs[2], rgbs[2], rgbs[3], rgbs[3]]    

plotPoints2d(distList, titles, stateNames, outFile, xRange=(-0.1,1.1),
                 yRange=(-0.1, 1.2), ptSize=125, xLabel="Precision",
                 yLabel="Recall", cols=2, width=10, rowHeight=5,
                 markerList = markerList, rgbs = prgbs)

iPoints = []
bPoints = []
stateNames = []

# from mcSegment1_flat_comp.txt
iPoints.append((0.6136363636363636, 0.9333333333333333))
stateNames.append("HMM LTR")

iPoints.append((0.8269230769230769, 0.671875))
stateNames.append("HMM NON-LTR")

# from combined_flat_comp.txt
iPoints.append((0.23684210526315788, 0.8666666666666667))
stateNames.append("3way Union LTR")

iPoints.append((0.6213592233009708, 0.828125))
stateNames.append("3way Union NON-LTR")

prgbs = [rgbs[0], rgbs[0], rgbs[1], rgbs[1]]

titles = ["Element-level Accuracy (th=0.8)"]
distList = [ iPoints ]
assert len(distList[0]) == 4
markerList = [ "o" , "^" ]

plotPoints2d(distList, titles, stateNames, outFile + "com.pdf", xRange=(-0.1,1.1),
                 yRange=(-0.1, 1.2), ptSize=125, xLabel="Precision",
                 yLabel="Recall", cols=2, width=10, rowHeight=5,
                 markerList = markerList, rgbs = prgbs)
