class PolicyBase:
    def evaluate(self, history, verification, started_at=None):
        raise NotImplementedError
