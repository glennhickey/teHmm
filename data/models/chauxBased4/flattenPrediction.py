#!/usr/bin/env python

"""

1) flatten LTR elements from 5 states to 1 (inside)
2) flatten non-LTR elements from 3 states to 1 (non-LTR)

"""

import sys
import os
from teHmm.common import runShellCommand, getLogLevelString, getLocalTempPath

assert len(sys.argv) == 3

infile = sys.argv[1]
outfile = sys.argv[2]
tempfile = outfile + "_temp"

# merge up TSD elements
runShellCommand("filterPredictions.py %s --mergeBefore \"TSD|left,LTR|left,non-LTR\" --mergeAfter \"TSD|right,LTR|right,non-LTR\" > %s" % (infile, tempfile))

# merge up LTR elements
runShellCommand("filterPredictions.py %s --mergeBefore \"LTR|left,inside\" --mergeAfter \"LTR|right,inside\" > %s" % (tempfile, outfile))

runShellCommand("rm -f %s" % tempfile)
