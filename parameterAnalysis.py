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
        distanceMatrix /= len(points[0])
    hc = scipy.cluster.hierarchy.linkage(distanceMatrix, method='average')
    return hc

def rankHierarchies(hcList, rankStat):
    """ produce a ranking of hierarchical clusterings using one of a variety
    of statistics. Each element of hcList is the output of
    hierarchichalCluster() """
    
    pass

def plotHierarchicalClusters(hcList, titles, leafNames, outFile):
    """ print out a bunch of dendrograms to a PDF file.  Each element of
    hcList is the output of hierarchichalCluster()"""
    cols = 3
    rows = int(np.ceil(float(len(hcList)) / float(cols)))
    width=10
    height=5 * rows

    pdf = pltBack.PdfPages(outFile)
    fig = plt.figure(figsize=(width, height))
    plt.clf()
    for i, hc in enumerate(hcList):
        plt.subplot(rows, cols, i)
        dgram = scipy.cluster.hierarchy.dendrogram(
            hc, color_threshold=1, labels=leafNames, show_leaf_counts=False)
#            p=6,
#            truncate_mode='lastp')
        plt.title(titles[i])
        plt.setp(plt.xticks()[1], rotation=-90, fontsize=10)
    fig.tight_layout()
    fig.savefig(pdf, format = 'pdf')
    pdf.close()

def pcaFlatten(points, outDim = 2):
    """ flatten points to given dimensionality using PCA """
    pass

def plotPoints2d(points, label, outFile):
    """ plot some points to a pdf file """
    pass


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
