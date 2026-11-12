
def confusion_matrix(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Input lengths must match.")

    if len(y_true) == 0:
        raise ValueError("Inputs cannot be empty.")

    tp = tn = fp = fn = 0

    for actual, predicted in zip(y_true, y_pred):
        if actual not in (0, 1) or predicted not in (0, 1):
            raise ValueError("Labels must be 0 or 1.")

        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        else:
            fn += 1

    return tp, tn, fp, fn


def roc_auc(y_true, y_score):
    if len(y_true) != len(y_score):
        raise ValueError("Input lengths must match.")

    if not y_true:
        raise ValueError("Inputs cannot be empty.")

    if any(label not in (0, 1) for label in y_true):
        raise ValueError("Labels must be 0 or 1.")

    if any(not 0 <= score <= 1 for score in y_score):
        raise ValueError("Scores must be between 0 and 1.")

    positives = sum(y_true)
    negatives = len(y_true) - positives

    # ROC AUC is undefined when only one class is present.
    if positives == 0 or negatives == 0:
        return None

    # Pairwise definition of AUC.
    # A tied positive-negative pair contributes 0.5.
    positive_scores = [
        score for label, score in zip(y_true, y_score)
        if label == 1
    ]
    negative_scores = [
        score for label, score in zip(y_true, y_score)
        if label == 0
    ]

    correct = 0.0

    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                correct += 1
            elif positive == negative:
                correct += 0.5

    return correct / (positives * negatives)


def evaluate(y_true, y_pred, y_score=None):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    specificity = tn / (tn + fp) if tn + fp else 0.0

    balanced_accuracy = (recall + specificity) / 2

    # F1 for the negative class
    precision_0 = tn / (tn + fn) if tn + fn else 0.0
    recall_0 = specificity

    f1_0 = (
        2 * precision_0 * recall_0 / (precision_0 + recall_0)
        if precision_0 + recall_0 else 0.0
    )

    macro_f1 = (f1 + f1_0) / 2

    auc = None
    if y_score is not None:
        auc = roc_auc(y_true, y_score)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "roc_auc": auc,
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }
    }


def evaluate_by_cluster(y_true, y_pred, y_score, cluster_ids):
    if len(y_true) != len(cluster_ids):
        raise ValueError("Cluster IDs must match input lengths.")

    if y_score is not None and len(y_true) != len(y_score):
        raise ValueError("Scores must match input lengths.")

    overall = evaluate(y_true, y_pred, y_score)

    clusters = {}

    for cluster in dict.fromkeys(cluster_ids):
        indices = [
            i for i, value in enumerate(cluster_ids)
            if value == cluster
        ]

        true = [y_true[i] for i in indices]
        pred = [y_pred[i] for i in indices]

        scores = None
        if y_score is not None:
            scores = [y_score[i] for i in indices]

        clusters[cluster] = evaluate(true, pred, scores)

    return {
        "overall": overall,
        "clusters": clusters
    }
