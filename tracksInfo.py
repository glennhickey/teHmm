#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
#!/usr/bin/env python

import os
import sys
import xml.etree.ElementTree as ET

""" Read some tracks info.  For now just a map between track name
and track path.  
"""
class TracksInfo(object):
    def __init__(self, inputPath = None):
        #: Store list of paths as mapped to track names
        #: (spaces not supported for now)
        self.pathMap = dict()
        if inputPath is not None:
            self.loadFile(inputPath)

    def loadFile(self, inputPath):
        """Load in an xml file that contains a list of track elements right
        below its root node.  Will extend to contain more options..."""
        root = ET.parse(inputPath).getroot()
        for child in root.findall("track"):
            self.pathMap[child.attrib["name"]] = child.attrib["path"]

    def getNumTracks(self):
        """Number of tracks for model"""
        return len(self.pathMap)

    def getPath(self, trackName):
        """Retrieve path of annoation track.  Return None if not found"""
        if trackName not in self.pathMap:
            return None
        else:
            return self.pathMap[trackName]
    
    def save(self, path):
        root = ET.Element("teModelConfig")
        for (trackName, trackPath) in self.pathMap.items():
            track = ET.SubElement(root, "track")
            track.set("name", trackName)
            track.set("path", path)
        ET.ElementTree(root).write(path)
        
