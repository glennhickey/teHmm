#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
#!/usr/bin/env python

import os
import sys
import logging
import numpy as np
import xml.etree.ElementTree as ET

def hashDNA(s):
    """ crappy DNA hashing function
    IMPORTANT:  return values only unique across fixed string length.  
     """
    
    # 3bits / char for up to 64 bits
    assert len(s) < 22
    val = 0
    for i, c in enumerate(s):
        if c == "A" or c == "a":
            val |= 0 << (i * 3)
        elif c == "C" or c == "c":
            val |= 1 << (i * 3)
        elif c == "G" or c == "g":
            val |= 2 << (i * 3)
        elif c == "T" or c == "t":
            val |= 3 << (i * 3)
        else:
            assert c == "N" or c == "n"
            val |= 4 << (i * 3)
    return val

class KmerTable:
    """ Hash table of kmers for quick seed alignment.  Easier to 
    just reimplement here for now than go through some big library
    that's tuned to look for longer matches etc..  Each string is 
    mapped to a list (which should be kept sorted) of positions"""
    def __init__(self, kmerLen = 3, hashFn = None):
        if hashFn is None:
            # very inefficent space-wise: for testing purposes only
            # should use with hashDNA in practice (at least)
            self.hashFn = lambda x: x
        else:
            self.hashFn = hashFn
        self.table = dict()
        self.kmerLen = kmerLen
        # for debugging
        self.useClosed = True

    def addKmer(self, kmer, position):
        """ insert position into the kmer table."""
        assert len(kmer) == self.kmerLen
        key = self.hashFn(kmer)
        if key not in self.table:
            self.table[key] = [position]
        else:
            assert self.table[key][-1] <= position
            self.table[key].append(position)

    def getKmer(self, kmer):
        """ return list of positions of kmer in table."""
        assert len(kmer) == self.kmerLen
        key = self.hashFn(kmer)
        if key not in self.table:
            return []
        else:
            return self.table[key]

    def loadString(self, targetText):
        """ create a kmer hash from given string """
        self.table = dict()
        for i in xrange(max(0, len(targetText) - self.kmerLen) + 1):
            self.addKmer(targetText[i:i+self.kmerLen], i)

    def exactMatches(self, queryText, minMatchLen = 3, maxMatchLen = 10):
        """ find all words in the table of length matchLength that
        are exactly present in queryText.  Each patch is a pair (i,j)
        where i is the position of the character of the match in 
        queryText, and j is the first position in the table (corpus)"""
        assert minMatchLen >= self.kmerLen
        N = len(queryText)
        closedMatches = []
        activeMatches = []
        for i in xrange(max(0, N - self.kmerLen + 1)):
            key = self.hashFn(queryText[i:i+self.kmerLen])
            if key in self.table:
                matchPositions = self.table[key]
                matchPairs = [[i, i+self.kmerLen, j, j+ self.kmerLen] 
                              for j in matchPositions]
                #print "M", matchPairs
                activeMatches = self.resolveOverlaps(matchPairs, 
                                                     activeMatches, 
                                                     closedMatches,
                                                     maxMatchLen)
                #print "A", activeMatches
                #print "C", closedMatches
                #print "777"
        return closedMatches + activeMatches

    def resolveOverlaps(self, matches, activeMatches, closedMatches, 
                        maxMatchLen):
        """merge pairwise alignments in matches list that overlap with pairs in 
        activeMatches.   any matches that 
        have no hope of being merged in the future (during the left-right scan)
        get added to closedMatches"""

        assert len(matches) > 0
        if len(activeMatches) == 0:
            return matches
        
        outList = []

        # do all against all comparison of new candidates with active list,
        # merging with the active list when possible.  Note sorted order 
        # should be used to speed up but list sizes are so small right now
        # we don't bother.  
        for i in xrange(len(matches)):
            found = False
            for j in xrange(len(activeMatches)):
                mergedMatch = None
                if activeMatches[j][1] - activeMatches[j][0] + \
                  matches[i][1] - matches[i][0] < maxMatchLen:
                    mergedMatch = self.getMerge(activeMatches[j], matches[i])
                if mergedMatch is not None:
                    activeMatches[j] = mergedMatch
                    found = True
                    break
            if not found:
                outList.append(matches[i])
        
        # filter out matches from the output that have no hope of being merged
        # in the future.  
        outList = activeMatches + outList
        cleanList = []
        for match in outList:
            if self.useClosed is True and match[0] > matches[0][1]:
                closedMatches.append(match)
            else:
                cleanList.append(match)
        return cleanList

    def getMerge(self, leftMatch, rightMatch):
        """ return the merge of two pairwise alignments, by extending the left one
        by the length of the overhand.  if no valid merge possible, return None"""        
        assert leftMatch[0] <= rightMatch[0]
        if rightMatch[0] >= leftMatch[0] and rightMatch[0] <= leftMatch[1] and\
          rightMatch[2] >= leftMatch[2] and rightMatch[2] <= leftMatch[3] and\
          rightMatch[1] >= leftMatch[1]:
            ovl1 = rightMatch[1] - leftMatch[1]
            ovl2 = rightMatch[3] - leftMatch[3]
            if ovl1 == ovl2:
              return [leftMatch[0], leftMatch[1] + ovl1, leftMatch[2], leftMatch[3] + ovl1]
        return None

          
             
        
                
