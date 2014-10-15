from libc.math cimport exp, log
import numpy as np
cimport numpy as np
cimport cython
np.import_array()

ctypedef np.float64_t dtype_t
ctypedef np.int32_t itype_t

cdef dtype_t _NINF = -np.inf
cdef dtype_t _MINDBL = -1e20
        
@cython.boundscheck(False)
def runSum(np.ndarray[np.uint8_t, ndim=1] mask,
           np.ndarray[itype_t, ndim=1] outSum):
    cdef itype_t runningSum = 0
    cdef itype_t N = len(mask)
    cdef itype_t i = 0
    assert N == len(outSum)
    
    with nogil:    
        for i in xrange(N):
            outSum[i] = runningSum
            if mask[i] == 0:
                runningSum += 1
    
