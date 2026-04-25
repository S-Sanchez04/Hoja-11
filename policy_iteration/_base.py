from abc import ABC


class PolicyIteration(ABC):

    def __init__(self, policy_evaluator, policy_improver):
        self.policy_evaluator = policy_evaluator
        self.policy_improver = policy_improver

    def step(self):
        """
            executes one iteration of the policy iteration algorithm
        """
        raise NotImplementedError
    
    def run(self, max_iter=10**6):
        """
            :param max_iter: maximum number of iterations before the algorithm stops
            tun
        """
        for _ in range(max_iter):
            improved = self.step()
            if not improved:
                break
        return self.policy_improver.policy
