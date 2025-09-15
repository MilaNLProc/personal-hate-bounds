import numpy as np
import evaluate
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
    precision_score, recall_score, f1_score)

metrics = evaluate.load('f1')
# metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def compute_metrics(eval_pred):
    """
    Note: this works if we use a single metric, or a combination of metrics
          whose `compute` method takes the same inputs. This is NOT the case
          for `accuracy` and `f1` as the latter requires the additional
          `average` input w.r.t. the former. If a combination of metrics
          is needed, use the `compute_metrics_sklearn` function instead.
    """
    predictions = np.argmax(eval_pred.predictions, axis=1)
    
    return metrics.compute(
        predictions=predictions, references=eval_pred.label_ids
    )


def compute_metrics_sklearn(eval_pred):
    """
    Computes a combination of metrics using the SKlearn functions directly.
    """
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    labels = eval_pred.label_ids
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='macro',
        zero_division=np.nan
    )
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def compute_metrics_sklearn_training(predicted_labels, targets):
    """
    Computes the needed metrics given the predictions and targets. To be used
    inside the custom training routine.
    """
    accuracy = accuracy_score(targets, predicted_labels)
    precision = precision_score(targets, predicted_labels, average='macro')
    recall = recall_score(targets, predicted_labels, average='macro')
    f1 = f1_score(targets, predicted_labels, average='macro')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
