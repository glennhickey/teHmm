#!/usr/bin/env python

#Copyright (C) 2015 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse
import numpy as np
from pybedtools import BedTool, Interval
# give up for now on drawing the venn diagram... just print out all the values to be
# hand pasted into a picture...
#from matplotlib import pyplot as plt
#from matplotlib_venn import venn3, venn3_circles
#from teHmm.scripts.venn_maker import venn_maker

from teHmm.common import intersectSize, initBedTool, cleanBedTool
from teHmm.common import logger, getLocalTempPath, runShellCommand
from teHmm.trackIO import readBedIntervals

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Venn diagram from some BED files.")

    parser.add_argument("inputFiles", help="BED files (should already be filtered)",
                        nargs='+')
#    parser.add_argument("outTiff", help="Output TIFF file")
    parser.add_argument("--names", help="Names (corresponding to each file) for chart",
                        nargs="+", default=None)
    
    args = parser.parse_args()
    tempBedToolPath = initBedTool()

    if args.names is None:
        args.names = [os.path.splitext(os.path.basename(i))[0] for i in args.inputFiles]
    assert len(args.names) == len(args.inputFiles)
    assert len(args.names) < 5

    ## runVennMaker(args)
        
    if len(args.inputFiles) == 1:
        venn1set(args)
    elif len(args.inputFiles) == 2:
        venn2set(args)
    elif len(args.inputFiles) == 3:
        venn3set(args)
    elif len(args.inputFiles) == 4:
        venn4set(args)

    
    cleanBedTool(tempBedToolPath)

def minus(x, y, z1=None, z2=None):
    """ ((x - y) - z1) - z2 """
    t = x.subtract(y).sort()
    if z1 is not None:
        t = t.subtract(z1).sort()
    if z2 is not None:
        t = t.subtract(z2).sort()
    return t

def inter(x, y, z1=None, z2=None):
    """ intersection of two or three bedtools """
    t = x.intersect(y).sort()
    if z1 is not None:
        t = t.intersect(z1).sort()
    if z2 is not None:
        t = t.intersect(z2).sort()
    return t

def cov(x):
    return "%.2f" % (float(x.total_coverage()) / 1000000.)
    
def venn1set(args):
    """ 1 set """
    a = BedTool(args.inputFiles[0]).merge()
    print "A: %s (%s)" % (args.names[0], cov(a))
    print "--------------------"
    print "A: %s" % cov(a)

def venn2set(args):
    """ 2 set """
    a = BedTool(args.inputFiles[0]).sort().merge()
    b = BedTool(args.inputFiles[1]).sort().merge()
    print "A: %s (%s)" % (args.names[0], cov(a))
    print "B: %s (%s)" % (args.names[1], cov(b))
    print "--------------------"
    print "A: %s" % cov(minus(a,b))
    print "B: %s" % cov(minus(b,a))
    print "AB: %s" % cov(inter(a,b))

def venn3set(args):
    """ 3 set """    
    a = BedTool(args.inputFiles[0]).sort().merge()
    b = BedTool(args.inputFiles[1]).sort().merge()
    c = BedTool(args.inputFiles[2]).sort().merge()
    print "A: %s (%s)" % (args.names[0], cov(a))
    print "B: %s (%s)" % (args.names[1], cov(b))
    print "C: %s (%s)" % (args.names[2], cov(c))
    print "--------------------"
    print "A:   %s" % cov(minus(a, b, c))
    print "B:   %s" % cov(minus(b, a, c))
    print "C:   %s" % cov(minus(c, a, b))
    print "AB:  %s" % cov(minus(inter(a,b), c))
    print "AC:  %s" % cov(minus(inter(a,c), b))
    print "BC:  %s" % cov(minus(inter(b,c), a))
    print "ABC: %s" % cov(inter(a, b, c))
    
def venn4set(args):
    """ 4 set """
    a = BedTool(args.inputFiles[0]).sort().merge()
    b = BedTool(args.inputFiles[1]).sort().merge()
    c = BedTool(args.inputFiles[2]).sort().merge()
    d = BedTool(args.inputFiles[3]).sort().merge()
    print "A: %s (%s)" % (args.names[0], cov(a))
    print "B: %s (%s)" % (args.names[1], cov(b))
    print "C: %s (%s)" % (args.names[2], cov(c))
    print "D: %s (%s)" % (args.names[3], cov(d))
    print "--------------------"

    print "A:    %s" % cov(minus(a, b, c, d))
    print "B:    %s" % cov(minus(b, a, c, d))
    print "C:    %s" % cov(minus(c, a, b, d))
    print "D:    %s" % cov(minus(d, a, b, c))
    print "AB:   %s" % cov(minus(inter(a,b), c, d))
    print "AC:   %s" % cov(minus(inter(a,b), b, d))
    print "AD:   %s" % cov(minus(inter(a,d), b, c))        
    print "BC:   %s" % cov(minus(inter(b,c), a, d))
    print "BD:   %s" % cov(minus(inter(b,d), a, c))
    print "CD:   %s" % cov(minus(inter(c,d), a, b))
    print "ABC:  %s" % cov(minus(inter(a,b,c), d))
    print "BCD:  %s" % cov(minus(inter(b,c,d), a))
    print "ACD:  %s" % cov(minus(inter(a,c,d), b))
    print "ABD:  %s" % cov(minus(inter(a,b,d), c))
    print "ABCD: %s" % cov(inter(a, b, c, d))

def runVennMaker(args0):                     
    # venn_maker seems designed to run on intervals (and looks pretty broken doing this).
    # try converting to base intervals.
    todie = []
    for i, f in enumerate(args.inputFiles):
        tempFile = getLocalTempPath("Temp_%d" % i, ".bed")
        todie.append(tempFile)
        baserize(f, tempFile)
        args.inputFiles[i] = tempFile
    
    venn_maker(args.inputFiles, args.names, args.outTiff, "venn.R",
               additional_args=None, run=True)

    for f in todie:
        runShellCommand("rm -f %s" % f)

def baserize(inBed, outBed):
    outFile = open(outBed, "w")
    for interval in readBedIntervals(inBed):
        for i in xrange(interval[2] - interval[1]):
            outFile.write("%s\t%d\t%d\n" % (interval[0], interval[1] + i, interval[1] + i + 1))
    outFile.close()
    


if __name__ == "__main__":
    sys.exit(main())
