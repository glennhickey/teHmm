#!/usr/bin/env python

import sys

prec = float(sys.argv[1])
rec = float(sys.argv[2])

if prec + rec > 0:
    print (2. * prec * rec) / (prec + rec)
else:
    print 0


