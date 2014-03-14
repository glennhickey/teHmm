#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
#!/usr/bin/env python

import os
import sys
import logging
import argparse
import numpy as np
import scipy
import scipy.spatial
import scipy.cluster
import matplotlib
import math
#matplotlib.use('Agg')
import matplotlib.backends.backend_pdf as pltBack
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.pylab  as pylab
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, LogLocator
from matplotlib.ticker import LogFormatter
import matplotlib.mlab as mlab
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.mlab import PCA
import matplotlib.cm as cm



from teHmm.basehmm import normalize

"""
Methods to analyse parameter distributions of any graphical model.  In particular
we are interested in finding the tracks that provide most information when it
comes to distinguishing between states.  Note that this information is only
relevant in conjunction with some measure of accuracy.  IE one track can show
amazing performance for state selection but if it results in wrong predictions
then it's not very useful.
"""

def hierarchicalCluster(points, normalizeDistances=False):
    """ Perform simple hiearchical clustering using euclidian distance"""
    assert points is not None and len(points) > 0
    distanceMatrix = scipy.spatial.distance.pdist(points, "euclidean")
    if normalizeDistances is True:
        distanceMatrix /= math.pow(len(points[0]), 0.5)
    hc = scipy.cluster.hierarchy.linkage(distanceMatrix, method='average')
    return hc

def rankHierarchies(hcList, rankStat = "branch_length"):
    """ produce a ranking of hierarchical clusterings using one of a variety
    of statistics. Each element of hcList is the output of
    hierarchichalCluster()
    Return list of integer indexes in increasing order"""

    inputRanking = [x for x in xrange(len(hcList))]
        
    if rankStat == "branch_length":
        totalLengths = [np.sum([x[2] for x in hc]) for hc in hcList]
        sortedRanking = sorted(zip(totalLengths, inputRanking), reverse=True)
        ranks = list(zip(*sortedRanking)[1])
        return ranks
    else:
        raise RuntimeError("Rankstat %s not recognized" % rankStat)

def plotHierarchicalClusters(hcList, titles, leafNames, outFile):
    """ print out a bunch of dendrograms to a PDF file.  Each element of
    hcList is the output of hierarchichalCluster()"""
    cols = 4
    rows = int(np.ceil(float(len(hcList)) / float(cols)))
    width=10
    height=5 * rows

    pdf = pltBack.PdfPages(outFile)
    fig = plt.figure(figsize=(width, height))
    plt.clf()
    for i, hc in enumerate(hcList):
        # +1 below is to prevent 1st element from being put last
        # (ie sublot seems to behave as if indices are 1-base)
        plt.subplot(rows, cols, (i + 1) % len(hcList))
        dgram = scipy.cluster.hierarchy.dendrogram(
            hc, color_threshold=None, labels=leafNames, show_leaf_counts=False)
#            p=6,
#            truncate_mode='lastp')
        plt.title(titles[i])
        plt.setp(plt.xticks()[1], rotation=-90, fontsize=10)
    fig.tight_layout()
    fig.savefig(pdf, format = 'pdf')
    pdf.close()

def pcaFlatten(points, outDim = 2):
    """ flatten points to given dimensionality using PCA """
    assert outDim == 2
    
    # will get LinAlgError: SVD did not converge exception if all points
    # lie on some plane (ie all values equal for some dimension so we
    # have to check for that first
    dims = []
    for dim in xrange(len(points[0])):
        vals = set()
        for point in points:
            vals.add(point[dim])
        if len(vals) > 1:
            dims.append(dim)
    assert len(dims) > 0
    cleanPoints = np.array([[point[i] for i in dims] for point in points])
    assert len(cleanPoints) > 0
    
    pca = PCA(cleanPoints)
    return pca.Y, np.sum(pca.fracs[:2])

colorList = ['#1f77b4', # dark blue
             '#aec7e8', # light blue
            '#ff7f0e', # bright orange
            '#ffbb78', # light orange
            '#4B4C5E', # dark slate gray
            '#9edae5', # light blue 
            '#7F80AB', # purple-ish slate blue
            '#c7c7c7', # light gray
            '#9467bd', # dark purple
            '#c5b0d5', # light purple
            '#d62728', # dark red
            '#ff9896', # light red
                 ]
def plotPoints2d(distList, titles, stateNames, outFile):
    """ plot some points to a pdf file """
    cols = 2
    rows = int(np.ceil(float(len(distList)) / float(cols)))
    width=10
    height=5 * rows
    alpha = 0.7

    # pallettes are here : cm.datad.keys()
    rgbs = [cm.gist_rainbow_r(float(i) / float(len(stateNames)))
            for i in xrange(len(stateNames))]
    for i in xrange(len(rgbs)):
        rgbs[i] = list(rgbs[i])
        rgbs[i][3] = alpha

    pdf = pltBack.PdfPages(outFile)
    fig = plt.figure(figsize=(width, height))
    plt.clf()
    for i,  dist in enumerate(distList):
        # +1 below is to prevent 1st element from being put last
        # (ie sublot seems to behave as if indices are 1-base)
        plt.subplot(rows, cols, (i + 1) % len(distList))
        plotList = []
        for j in xrange(len(dist)):
            plotList.append(plt.scatter(dist[j, 0], dist[j, 1], c=rgbs[j],
                                        s=100))
        plt.axis('equal')
        plt.grid(True)
        plt.title(titles[i])
        if i % cols == 0:
            # write legend
            plt.legend(plotList, stateNames, 
            scatterpoints=1,
            loc='upper left',
            ncol=3,
            fontsize=8)

    fig.tight_layout()
    fig.savefig(pdf, format = 'pdf')
    pdf.close()


### Crappy sandbox for testing ###

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = parser.parse_args()

    linkageMat = hierarchicalCluster([(0,1), (1,2), (10,10), (10, 15)],
                                     normalizeDistances = True)
    plotHierarchicalClusters([linkageMat, linkageMat, linkageMat], ["yon", "Title", "Blin"], ["A", "B", "C", "D"],
                             "blin.pdf")

if __name__ == "__main__":
    sys.exit(main())
