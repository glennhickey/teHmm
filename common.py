#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse
import subprocess
from multiprocessing import Pool
import logging
import resource
import logging.handlers
import collections
import numpy as np
import string
import random
from argparse import ArgumentParser
from optparse import OptionParser, OptionContainer, OptionGroup
import pybedtools

LOGZERO = -1e100
EPSILON = np.finfo(float).eps

def __myLogFloat(x):
    if np.abs(x) < EPSILON:
        return LOGZERO
    return np.log(x)

""" Replace np.log to accept zero """
myLog = np.vectorize(__myLogFloat)
    
def runShellCommand(command):
    try:
        logger.info("Running %s" % command)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                                   stderr=sys.stderr, bufsize=-1)
        output, nothing = process.communicate()
        sts = process.wait()
        if sts != 0:
            raise RuntimeError("Command: %s exited with non-zero status %i" %
                               (command, sts))
        return output
    except KeyboardInterrupt:
        raise RuntimeError("Aborting %s" % command)

def runParallelShellCommands(cmdList, numProc):
    if numProc == 1 or len(cmdList) == 1:
        map(runShellCommand, cmdList)
    elif len(cmdList) > 0:
        mpPool = Pool(processes=min(numProc, len(cmdList)))
        result = mpPool.map_async(runShellCommand, cmdList)
        # specifying a timeout allows keyboard interrupts to work?!
        # http://stackoverflow.com/questions/1408356/keyboard-interrupts-with-pythons-multiprocessing-pool
        try:
            result.get(sys.maxint)
        except KeyboardInterrupt:
            mpPool.terminate()
            raise RuntimeError("Keyboard interrupt")
        if not result.successful():
            raise "One or more of commands %s failed" % str(cmdList)

def getLocalTempPath(prefix="", extension="", tagLen=5):
    S = string.ascii_uppercase + string.digits
    tag = ''.join(random.choice(S) for x in range(tagLen))
    tempPath = os.path.join(os.getcwd(), "%s%s%s" % (prefix, tag, extension))
    return tempPath
        
def initBedTool(tempPrefix=""):
    # keep temporary files in current directory, to make it a little harder to
    # lose track of them and clog up the system....
    S = string.ascii_uppercase + string.digits
    tag = ''.join(random.choice(S) for x in range(5))
    tempPath = os.path.join(os.getcwd(), "%sTempBedTool_%s" % (tempPrefix, tag))
    logger.info("Temporary directory for BedTools (you may need to manually"
                 " erase in event of crash): %s" % tempPath)
    try:
        os.makedirs(tempPath)
    except:
        pass
    pybedtools.set_tempdir(tempPath)
    return tempPath

def cleanBedTool(tempPath):
    # do best to erase temporary bedtool files if necessary
    # (tempPath argument must have been created with initBedTool())
    assert "TempBedTool_" in tempPath
    pybedtools.cleanup(remove_all=True)
    runShellCommand("rm -rf %s" % tempPath)

        
#########################################################
#########################################################
#########################################################  
#global logging settings / log functions
#########################################################
#########################################################
#########################################################

# Logging stuff copied from bioio.py from https://github.com/benedictpaten/sonLib
# 
#Copyright (C) 2006-2012 by Benedict Paten (benedictpaten@gmail.com)
#
#Released under the MIT license, see LICENSE.txt

loggingFormatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def __setDefaultLogger():
    l = logging.getLogger()
    for handler in l.handlers: #Do not add a duplicate handler unless needed
        if handler.stream == sys.stderr:
            return l
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(loggingFormatter)
    l.addHandler(handler) 
    l.setLevel(logging.CRITICAL)
    return l

logger = __setDefaultLogger()
logLevelString = "CRITICAL"

def redirectLoggerStreamHandlers(oldStream, newStream):
    """Redirect the stream of a stream handler to a different stream
    """
    for handler in list(logger.handlers): #Remove old handlers
        if handler.stream == oldStream:
            handler.close()
            logger.removeHandler(handler)
    for handler in logger.handlers: #Do not add a duplicate handler 
        if handler.stream == newStream:
           return
    logger.addHandler(logging.StreamHandler(newStream))

def getLogLevelString():
    return logLevelString

__loggingFiles = []
def addLoggingFileHandler(fileName, rotatingLogging=False):
    if fileName in __loggingFiles:
        return
    __loggingFiles.append(fileName)
    if rotatingLogging:
        handler = logging.handlers.RotatingFileHandler(fileName, maxBytes=1000000, backupCount=1)
    else:
        handler = logging.FileHandler(fileName)
    handler.setFormatter(loggingFormatter)
    logger.addHandler(handler)
    return handler
    
def setLogLevel(logLevel):
    logLevel = logLevel.upper()
    assert logLevel in [ "OFF", "CRITICAL", "INFO", "DEBUG" ] #Log level must be one of these strings.
    global logLevelString
    logLevelString = logLevel
    if logLevel == "OFF":
        logger.setLevel(logging.FATAL)
    elif logLevel == "INFO":
        logger.setLevel(logging.INFO)
    elif logLevel == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif logLevel == "CRITICAL":
        logger.setLevel(logging.CRITICAL)

def logFile(fileName, printFunction=logger.info):
    """Writes out a formatted version of the given log file
    """
    printFunction("Reporting file: %s" % fileName)
    shortName = fileName.split("/")[-1]
    fileHandle = open(fileName, 'r')
    line = fileHandle.readline()
    while line != '':
        if line[-1] == '\n':
            line = line[:-1]
        printFunction("%s:\t%s" % (shortName, line))
        line = fileHandle.readline()
    fileHandle.close()

def addLoggingOptions(parser):
    ##################################################
    # BEFORE YOU ADD OR REMOVE OPTIONS TO THIS FUNCTION, BE SURE TO MAKE THE SAME CHANGES TO 
    # addLoggingOptions_argparse() OTHERWISE YOU WILL BREAK THINGS
    ##################################################
    parser.add_argument("--logOff", action="store_true", default=False,
                        help="Turn off logging. (default is CRITICAL)")
    parser.add_argument("--logInfo", action="store_true", default=False,
                     help="Turn on logging at INFO level. (default is CRITICAL)")
    parser.add_argument("--logDebug", action="store_true", default=False,
                     help="Turn on logging at DEBUG level. (default is CRITICAL)")
    parser.add_argument("--logLevel", default='CRITICAL',
                      help="Log at level (may be either OFF/INFO/DEBUG/CRITICAL). default=CRITICAL")
    parser.add_argument("--logFile", help="File to log in")
    parser.add_argument("--rotatingLogging", action="store_true", default=False,
                     help="Turn on rotating logging, which prevents log files getting too big. default=False")
  
def setLoggingFromOptions(options):
    """Sets the logging from a dictionary of name/value options.
    """
    #We can now set up the logging info.
    if options.logLevel is not None:
        setLogLevel(options.logLevel) #Use log level, unless flags are set..   
    
    if options.logOff:
        setLogLevel("OFF")
    elif options.logInfo:
        setLogLevel("INFO")
    elif options.logDebug:
        setLogLevel("DEBUG")
        
    logger.info("Logging set at level: %s" % logLevelString)  
    
    if options.logFile is not None:
        addLoggingFileHandler(options.logFile, options.rotatingLogging)
    
    logger.info("Logging to file: %s" % options.logFile)  
