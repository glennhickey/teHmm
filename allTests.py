#!/usr/bin/env python

#Copyright (C) 2012 by Glenn Hickey (hickey@soe.ucsc.edu)
#
#Released under the MIT license, see LICENSE.txtimport unittest

import unittest
import sys
import os
import argparse
from teHmm.tests.bedTrackTest import TestCase as bedTrackTest
from teHmm.tests.emissionTest import TestCase as emissionTest
from teHmm.tests.hmmTest import TestCase as hmmTest
from teHmm.tests.compareTest import TestCase as compareTest
from teHmm.tests.cfgTest import TestCase as cfgTest
from teHmm.tests.kmerTest import TestCase as kmerTest
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger

def allSuites(): 
    allTests = unittest.TestSuite((unittest.makeSuite(bedTrackTest, 'test'),
                                   unittest.makeSuite(emissionTest, 'test'),
                                   unittest.makeSuite(hmmTest, 'test'),
                                   unittest.makeSuite(compareTest, 'test'),
                                   unittest.makeSuite(cfgTest, 'test'),
                                   unittest.makeSuite(kmerTest, 'test')))
    return allTests
        
def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run unit tests")
    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)

    suite = allSuites()
    runner = unittest.TextTestRunner()
    i = runner.run(suite)
    return len(i.failures) + len(i.errors)
        
if __name__ == '__main__':
    import sys
    sys.exit(main())
                
