from ._base import PolicyIteration


class StandardPolicyIteration(PolicyIteration):

    def __init__(self, init_policy, policy_evaluator, policy_improver):
        """
            :param init_policy: policy with which the algorithm is initialized
            :param policy_evaluator: the policy evaluator
            :param policy_improver: the policy improver
        """
        super().__init__(policy_evaluator, policy_improver)
        self.policy = init_policy
    
    def step(self):
        self.policy_evaluator.reset(self.policy)
        has_improved = self.policy_improver.improve(self.policy_evaluator.q)
        self.policy = self.policy_improver.policy
        return has_improved