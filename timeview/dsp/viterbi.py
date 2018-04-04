"""Viterbi"""

import numpy as np
import numba


# @numba.jit((numba.float64[:, :], numba.float64), nopython=True, cache=True)  # eager compilation through function signature
def search_smooth(ftr, smooth):  # 100% function-based
    """P is the number of candidates at each time
    T is the number of available time points"""
    T, P = ftr.shape
    assert T > 1
    assert P < 65536  # could adjust dtype automatically, currently set at uint16
    trans = np.empty(P)
    zeta = np.empty(P)
    score = np.empty((2, P))  # one row for previous, and one for current score (no tracking for all t, saving memory)
    path = np.zeros((T, P), np.uint16) - 1  # often this matrix has less than %1 of meaningful entries, could be made sparse
    seq = np.zeros(T, np.uint16) - 1  # the - 1 (really: 65535) helps catch bugs
    # forward
    for t in range(T):
        # print(t, T)
        current = t % 2  # 2-cell ring buffer pointer
        previous = (t - 1) % 2
        # OBSERVATION t
        # observe(t, ftr, score[score_cur])  # score is the output
        score[current, :] = ftr[t]
        # OBSERVATION t
        if t == 0:
            jindex = np.where(score[0])[0]  # active FROM nodes
            assert len(jindex), 'no observations for target[0]'
        else:
            iindex = np.where(score[current] > -np.inf)[0]  # possible TO nodes
            # assert len(iindex), 'no observation probabilities above pruning threshold for target[%d]' % t
            for i in iindex:  # TO this node - TODO: this may be parallelizable
                # TRANSITION jindex -> i @ t
                # transition(t, ftr, jindex, i, trans)  # trans is the output
                trans[jindex] = -np.abs(jindex - i) * smooth
                # TRANSITION jindex -> i @ t
                # zeta[jindex] = score[score_prv, jindex] + trans[jindex]
                # zeta[jindex] = score[score_prv][jindex] + trans[jindex]
                zeta = score[previous] + trans  # really only needed over jindex, but indexing is slow
                path[t, i] = zindex = jindex[zeta[jindex].argmax()]
                score[current, i] += zeta[zindex]
            assert np.any(score[current] > -np.inf), 'score/prob[t] must not be all below pruning threshold'
            jindex = iindex  # new active FROM nodes
    # backward
    assert current == (T - 1) % 2
    # seq[-1] = iindex[score[score_cur, iindex].argmax()]
    seq[-1] = iindex[score[current][iindex].argmax()]
    for t in range(T - 1, 0, -1):
        seq[t - 1] = path[t, seq[t]]
    return seq, score[current, seq[-1]]