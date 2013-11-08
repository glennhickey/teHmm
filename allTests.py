#!/usr/bin/env python

#Copyright (C) 2012 by Glenn Hickey (hickey@soe.ucsc.edu)
#
#Released under the MIT license, see LICENSE.txtimport unittest

import unittest
import sys
import os
from teHmm.tests.bedTrackTest import TestCase as bedTrackTest
from teHmm.tests.emissionTest import TestCase as emissionTest
from teHmm.tests.hmmTest import TestCase as hmmTest

def allSuites(): 
    allTests = unittest.TestSuite((unittest.makeSuite(bedTrackTest, 'test'),
                                   unittest.makeSuite(emissionTest, 'test'),
                                   unittest.makeSuite(hmmTest, 'test')))
    return allTests
        
def main():    
    suite = allSuites()
    runner = unittest.TextTestRunner()
    i = runner.run(suite)
    return len(i.failures) + len(i.errors)
        
if __name__ == '__main__':
    import sys
    sys.exit(main())
                
