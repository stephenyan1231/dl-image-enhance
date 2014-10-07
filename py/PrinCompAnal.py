import numpy as np
from numpy import linalg as LA

def pca(data):
    # data: n*d
    n = data.shape[0]
    mean = np.mean(data, 0)
    cData = data - mean
    covMat = np.cov(cData, rowvar=0)
    evals, evecs = LA.eigh(covMat)
    evals = evals.real
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    cumSumEval = np.cumsum(evals)
    evalSum = np.sum(evals)
    cumFracEval = cumSumEval / evalSum
    pcaCoef = np.dot(data - mean, evecs)
    return pcaCoef, cumFracEval, evecs, evals, mean


# if __name__ == "__main__":
#     pass
