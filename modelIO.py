#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt

import os
import sys
import numpy as np
import pickle
import logging
from numpy.testing import assert_array_equal, assert_array_almost_equal

from .hmm import MultitrackHmm
from .cfg import MultitrackCfg

def loadModel(path):
    """ load an hmm or cfg from a pickle file """
    f = open(path, "rb")
    model = pickle.load(f)
    f.close()
    assert isinstance(model, MultitrackCfg) or isinstance(model, MultitrackHmm)
    model.validate()
    return model

def saveModel(path, model):
    """ save an hmm or cfg to a pickle file """
    assert isinstance(model, MultitrackCfg) or isinstance(model, MultitrackHmm)
    model.validate()  
    f = open(path, "wb")
    pickle.dump(model, f, 2)
    f.close()

