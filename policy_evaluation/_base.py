from abc import ABC, abstractmethod


class PolicyEvaluator(ABC):

    def __init__(self, gamma):
        self.gamma = gamma
        self.policy = None
        self._v_values = None
        self._q_values = None

    def reset(self, policy):
        """
            :param policy: the policy that is subject to evaluation
        """
        self.policy = policy
        self._after_reset()
    
    def _after_reset(self):
        """
            Hook function that is called after the policy has been resetted. Typical usage is to update q-values here.
        """
        pass

    @property
    @abstractmethod
    def provides_state_values(self):
        """
            True if this evaluator gives access to state values through the property `v`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def v(self):
        r"""
            :return: a dictionary v where v[s] estimates v^\pi(s) for the current policy \pi
        """
        raise NotImplementedError


    @property
    @abstractmethod
    def q(self):
        r"""
            :return: a 2-depth dictionary q where q[s][a] estimates q^\pi(s,a) for the current policy \pi
        """
        raise NotImplementedError
