#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
#!/usr/bin/env python

import os
import sys
import numpy as np
import pickle
from teModel import TEModel

"""
HMM Model for a transposable element.  Gets trained by a set of tracks.
Prototype implementation using scikit-learn
"""
class TEHMMModel(TEModel):
    def __init__(self):
        super(TEHMMModel, self).__init__()


