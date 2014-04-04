#!/usr/bin/env python

"""
1) merge TIR termini (and their tsds) into DNA elements (since truth only has DNA)

2) get rid of LTR and TSD elements that dont flank an inside or non-LTR

left with dna elements that look like

DNA

ltr elements that look like

TSD|left 
LTR|left
inside
LTR|right
TSD|right

and non-ltr elements that look like

TSD|left 
non-LTR
TSD|right

"""

import sys
import os
from teHmm.common import runShellCommand, getLogLevelString, getLocalTempPath

assert len(sys.argv) == 3

infile = sys.argv[1]
outfile = sys.argv[2]
tempfile = outfile + "_temp"

# get rid of any orphan TIR termini (not next to DNA)
runShellCommand("filterPredictions.py %s --mustBefore \"TIR|left,DNA\" --mustAfter \"TIR|right,DNA\" > %s" % (infile, outfile))

# merge up DNA elements into one
runShellCommand("filterPredictions.py %s --mergeBefore \"TSD|left,TIR|left\" --mergeAfter \"TSD|right,TIR|right\" > %s" % (outfile, tempfile))

runShellCommand("filterPredictions.py %s --mergeBefore \"TIR|left,DNA\" --mergeAfter \"TIR|right,DNA\" > %s" % (tempfile, outfile))

# get rid of orphan LTR termini (not next to inside)
runShellCommand("filterPredictions.py %s --mustBefore \"LTR|left,inside\" --mustAfter \"LTR|right,inside\" > %s" % (outfile, tempfile))

# get rid of orphan TSD termini (not next to ltr or non-ltr)
runShellCommand("filterPredictions.py %s --mustBefore \"TSD|left,LTR|left\" --mustAfter \"TSD|right,LTR|right\" > %s" % (tempfile, outfile))

runShellCommand("rm -f %s" % tempfile)
