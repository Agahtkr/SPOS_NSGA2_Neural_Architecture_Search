from sklearn.metrics import f1_score


class MetricTracker:
    """Computes metrics used by the NAS search loop."""

    def __init__(self, average_method='macro'):
        self.average_method = average_method

    def calculate_f1(self, targets, predictions):
        """Multi-class F1. FIX: sklearn's f1_score raises ValueError on
        empty input (e.g. a subnet's validation pass happens to produce
        zero labeled grid cells) - guarded to return 0.0 instead of
        crashing the whole NSGA-II run."""
        if len(targets) == 0:
            return 0.0
        return f1_score(targets, predictions, average=self.average_method, zero_division=0)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def compute_accuracy(self, targets, predictions):
        if len(targets) == 0:
            return 0.0
        correct = sum(1 for t, p in zip(targets, predictions) if t == p)
        return correct / len(targets)