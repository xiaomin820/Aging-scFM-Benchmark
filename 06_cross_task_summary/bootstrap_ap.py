"""TF-cluster weighted AP with exact score-tie handling.

Only thresholds with reference positives contribute to the AP sum. Store
cumulative positive and total edge counts by TF at those thresholds, then
multiply by TF multiplicities. This is algebraically equivalent to weighted
average_precision_score without sorting the edges again for each replicate.
"""
import numpy as np


def weighted_bootstrap_ap(scores, labels, edge_tf, multiplicities):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    edge_tf = np.asarray(edge_tf)
    multiplicities = np.asarray(multiplicities, dtype=float)
    if scores.ndim != 1 or scores.shape != labels.shape or scores.shape != edge_tf.shape:
        raise ValueError('Scores, labels and edge TF indices must be equal-length vectors')
    if not len(scores) or not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected finite scores and binary labels on a nonempty candidate universe')
    if multiplicities.ndim != 2 or not np.isfinite(multiplicities).all() or (multiplicities < 0).any():
        raise ValueError('TF multiplicities must be a finite nonnegative matrix')
    if not np.issubdtype(edge_tf.dtype, np.integer) or edge_tf.min() < 0 or edge_tf.max() >= multiplicities.shape[1]:
        raise ValueError('Edge TF indices are outside the multiplicity columns')
    order = np.argsort(-scores, kind='stable')
    sorted_scores, sorted_labels, sorted_tf = scores[order], labels[order], edge_tf[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores)-1]
    starts = np.r_[0, ends[:-1]+1]
    positive_ends = ends[np.add.reduceat(sorted_labels, starts) > 0]
    total = np.empty((len(positive_ends), multiplicities.shape[1]), dtype=float)
    positive = np.empty_like(total)
    for tf in range(multiplicities.shape[1]):
        mask = sorted_tf == tf
        total[:, tf] = np.cumsum(mask)[positive_ends]
        positive[:, tf] = np.cumsum(mask & (sorted_labels == 1))[positive_ends]
    increments = np.diff(positive, axis=0, prepend=np.zeros((1, positive.shape[1])))
    numerator = positive @ multiplicities.T
    denominator = total @ multiplicities.T
    delta = increments @ multiplicities.T
    precision = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    positive_weights = np.bincount(edge_tf, weights=labels, minlength=multiplicities.shape[1]) @ multiplicities.T
    return np.divide((precision*delta).sum(axis=0), positive_weights,
                     out=np.full(multiplicities.shape[0], np.nan), where=positive_weights > 0)
