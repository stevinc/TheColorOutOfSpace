from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, fbeta_score, hamming_loss


def precision(labels, pred, average):
    return precision_score(labels, pred, average=average)


def recall(labels, pred, average):
    return recall_score(labels, pred, average=average)


def hamming_loss(labels, pred):
    return hamming_loss(labels, pred)


def f1(labels, pred, average):
    return f1_score(labels, pred, average=average)


def f2(labels, pred, average):
    return fbeta_score(labels, pred, average=average, beta=2)


def average_precision(labels, score, average):
    return average_precision_score(labels, score, average=average)


metrics_def = {
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'f2': f2
}

