from libc.math cimport exp, log
import numpy as np
cimport numpy as np
cimport cython
from .track import TrackTable
np.import_array()

ctypedef np.float64_t dtype_t
ctypedef np.int32_t itype_t
ctypedef np.uint8_t atype_t

cdef dtype_t _NINF = -np.inf
        
@cython.boundscheck(False)
def fastCykTable(cfg, np.ndarray[atype_t, ndim=2] obs, alignmentTrack):
    """ Do the CYK dynamic programming algorithm (like viterbi) to
    compute the maximum likelihood CFG derivation of the observations."""
    cdef itype_t nObs = len(obs)
    cdef itype_t M = cfg.M
    cdef itype_t baseMatch = alignmentTrack is not None
    cdef np.ndarray[itype_t, ndim=1] emittingStates = cfg.emittingStates
    cdef np.ndarray[itype_t, ndim=1] helperDim1 = cfg.helperDim1
    cdef np.ndarray[itype_t, ndim=1] helperDim2 = cfg.helperDim2
    cdef np.ndarray[np.int32_t, ndim=3] helper1 = cfg.helper1
    cdef np.ndarray[np.int32_t, ndim=2] helper2 = cfg.helper2
    cdef np.ndarray[np.int64_t, ndim=4] tb = cfg.tb
    cdef np.ndarray[np.float_t, ndim=3] dp = cfg.dp
    cdef np.ndarray[np.float_t, ndim=3] logProbs1 = cfg.logProbs1
    cdef np.ndarray[np.float_t, ndim=2] logProbs2 = cfg.logProbs2
    cdef np.ndarray[np.float_t, ndim=2] emLogProbs = cfg.emLogProbs
    cdef itype_t size
    cdef itype_t match
    cdef itype_t i
    cdef itype_t j
    cdef itype_t k
    cdef itype_t q
    cdef itype_t lState
    cdef itype_t r1State
    cdef itype_t r2State
    cdef itype_t PAIRFLAG = cfg.PAIRFLAG
    cdef itype_t defAlignmentSymbol = cfg.defAlignmentSymbol
    for size in xrange(2, len(obs) + 1):
        for i in xrange(len(obs) + 1 - size):
            j = i + size - 1
            match = baseMatch and\
                    alignmentTrack[i,0] != defAlignmentSymbol and\
                    alignmentTrack[i,0] == alignmentTrack[j,0]
            for lState in emittingStates:
                for q in xrange(helperDim1[lState]):
                    r1State = helper1[lState, q, 0]
                    r2State = helper1[lState, q, 1]
                    for k in xrange(i, i + size - 1):
                        lp = logProbs1[lState, r1State, r2State] + \
                             dp[i, k, r1State] +\
                             dp[k+1, j, r2State]
                        if lp > dp[i, j, lState]:
                            dp[i, j, lState] = lp
                            tb[i, j, lState] = [k, r1State, r2State]
                if size > 2:
                    for q in xrange(helperDim2[lState]):
                        rState = helper2[lState, q]
                        lp = logProbs2[lState, rState] +\
                             dp[i+1, j-1, rState] +\
                             cfg.pairEmissionModel.pairLogProb(\
                            lState, \
                            emLogProbs[i, lState], \
                            emLogProbs[j, lState], \
                            match)
                        assert lp <= 0
                        if lp > dp[i, j, lState]:
                            dp[i, j, lState] = lp
                            tb[i, j, lState] = [PAIRFLAG,
                                                rState, rState]
