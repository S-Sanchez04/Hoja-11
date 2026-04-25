from ._base import PolicyImprover
import numpy as np


class StandardPolicyImprover(PolicyImprover):

    def __init__(self, min_advantage=10**-15):
        """
            :param min_advantage: minimum improvement that a q-value must offer over the current state value to trigger a change in policy
        """
        super().__init__()
        self.q = {}
        self.min_advantage = min_advantage

        def max_q_policy(s):
            options = self.q[s]
            return max(options, key=options.get)
        self._policy = max_q_policy

    def improve(self, q):
        has_improved = False
        for s, q_values in q.items():
            if s not in self.q or any([q_1 - self.q[s][a] > self.min_advantage for a, q_1 in q_values.items()]):
                has_improved = True
                break
        self.q = q
        return has_improved
    
    @property
    def policy(self):
        return self._policy