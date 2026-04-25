from abc import ABC, abstractmethod


class PolicyImprover(ABC):

    @abstractmethod
    def improve(self, q):
        """
            :param q: a 2-depth dictionary where q[s][a] = q(s,a)
            :return: True if the policy has been changed, False if not
            
            improves the policy based on current estimates of q accessible
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def policy(self):
        """
            :return: function that maps a state to an action
        """
        raise NotImplementedError