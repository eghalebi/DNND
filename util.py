import numpy as np

def get_binary_mtx(mtx, all_elements):
    size = len(all_elements)
    y_ = np.zeros(size*2)
    for i, [s,r] in enumerate(all_elements):
        if (s,r) in mtx or (r,s) in mtx:
            y_[i] = 1
            y_[i+size] = 1
    return y_

def get_f1_score(actual, prediction, all_elements):
    tp, fp, tn, fn = 0, 0,0,0
    for edge in all_elements:
        if edge in prediction or (edge[1],edge[0]) in prediction:
            if edge in actual or [edge[1],edge[0]] in actual:
                tp+=1
            else:
                fp+=1
        else:
            if edge in actual:
                fn+=1
            else:
                tn+=1
    if tp+fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if tp+fn == 0:
        recall = 0
    else:
        recall = tp/ (tp+fn)
    if precision+recall ==0:
        return 0

    return 2*precision*recall/(precision+recall)


def calc_f1(predicted, pairs_next_t, train_pairs, num_links):
    next_links = pairs_next_t[:num_links, :]
    train_nodes = np.unique(train_pairs)
    test_nodes = np.unique(next_links)
    Nt = len(test_nodes)
    adj_actual = np.zeros((Nt + 1, Nt + 1))
    adj_predic = np.zeros((Nt + 1, Nt + 1))
    for _, [s, r] in enumerate(next_links):
        if s in train_nodes:
            si = np.argwhere(test_nodes == s)[0]
        else:
            si = Nt
        if r in train_nodes:
            ri = np.argwhere(test_nodes == r)[0]
        else:
            ri = Nt
        adj_actual[si, ri] = 1
        adj_actual[ri, si] = 1
    for _, [s, r] in enumerate(predicted):
        if s in train_nodes:
            si = np.argwhere(test_nodes == s)[0]
        else:
            si = Nt
        if r in train_nodes:
            ri = np.argwhere(test_nodes == r)[0]
        else:
            ri = Nt
        adj_predic[si, ri] = 1
        adj_predic[ri, si] = 1

    tp = np.sum(np.logical_and(adj_predic, adj_actual))
    fp = np.sum(np.less(adj_actual, adj_predic))
    fn = np.sum(np.less(adj_predic, adj_actual))
    if tp + fp == 0 or (tp + fn) == 0:
        F = 0
    else:
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        if p + r == 0:
            F = 0
        else:
            F = (2 * (p * r) / (p + r))


    return F
def _retype(y_pred, y):
    y_pred = np.array(y_pred)
    y = np.array(y)

    return y_pred, y

#
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def mapk(y_pred, y, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(y, y_pred)])

def hits_k(y_pred, y, k=10):
    acc = 0
    for p_, y_ in zip(y_prob, y):
        top_k = p_.argsort()[-k:][::-1]
        acc += [1. if y_ in top_k else 0.]
    return sum(acc) / len(acc)


def portfolio(y_pred, y, k_list=None,other=False):
    y_pred, y = _retype(y_pred, y)
    scores = {}
    hits = []
    aps=[]
    for k in k_list:
        hits.append(hits_k(y_pred, y, k=k))
        aps.append(apk(y_pred,y, k=k))
        scores['hits@' + str(k)] = hits[-1]
        scores['ap@' + str(k)] = aps[-1]
    return scores, hits, aps
