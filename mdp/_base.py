import numpy as np

from abc import ABC


class MDP(ABC):
    """
        Abstract class to be used as a common interface by algorithms based on MDPs
    """

    @property
    def init_states(self) -> list:
        """

        :return: list of states in which the agent might start (uniformly distributed)
        """
        raise NotImplementedError

    @property
    def states(self) -> list:
        """

        :return: list of all possible states of the MDP (might not be implemented if this set is too large or infinite).
        """
        raise NotImplementedError

    def is_terminal_state(self, s) -> bool:
        """

        :param s: state to be checked for being terminal
        :return: True iff `s` is terminal (False otherwise)
        """
        return len(self.get_actions_in_state(s)) == 0

    @property
    def actions(self) -> list:
        """

        :return: list of all actions the agent could ever execute in any state.
        """
        actions = set()
        for s in self.get_states():
            actions |= set(self.get_actions_in_state(s))
        return list(actions)

    def get_actions_in_state(self, s) -> list:
        """

        :param s: state for which the set of applicable actions is queried
        :return: list of actions applicable for the agent in state `s`
        """
        raise NotImplementedError

    def get_reward(self, s) -> float:
        """

        :param s: state `s` for which the reward is being queried
        :return:  reward (float) the agent receives when entering state `s`
        """
        raise NotImplementedError

    def get_transition_distribution(self, s, a) -> dict:
        """

        :param s: state from which the agent departs
        :param a: action executed by the agent
        :return: dictionary describing the distribution among states in which the agent will end up
        """
        raise NotImplementedError


class ClosedFormMDP(MDP):

    def __init__(self, states, actions, prob_matrix, rewards):
        super().__init__()
        self._states = states
        self._terminal_states = set(s for i, s in enumerate(states) if np.sum(prob_matrix[i]) == 0)
        self._actions = actions
        self.prob_matrix = prob_matrix
        self.rewards = rewards
    
    @property
    def states(self):
        return self._states
    
    @property
    def actions(self):
        return self._actions

    def is_terminal_state(self, s):
        return s in self._terminal_states

    def get_reward(self, s):
        return self.rewards[self.states.index(s)]
    
    def get_transition_distribution(self, s, a):
        prob_vector = self.prob_matrix[self.states.index(s)][self.actions.index(a)]
        return {
            sp: prob
            for sp, prob in zip(self.states, prob_vector)
            if prob > 0
        }
    
    def get_actions_in_state(self, s):
        return self.actions

    @classmethod
    def from_mdp(cls, mdp : MDP):
        """
        :param mdp: the MDP object
        :return: triple (states, probs, rewards), where `states` and `rewards` are lists, and probs[s][a][s'] = P(s'|s,a)
        """

        # get closed form representation of the MDP
        states = list(mdp.states)
        actions = list(mdp.actions)
        size = len(mdp.states)
        prob_matrix = np.zeros((size, len(actions), size))
        for i, s in enumerate(states):
            for a in mdp.get_actions_in_state(s):
                j = actions.index(a)
                for ss, p in mdp.get_transition_distribution(s, a).items():
                    k = states.index(ss)
                    prob_matrix[i, j, k] = p
        rewards = np.array([mdp.get_reward(s) for s in states])
        return ClosedFormMDP(states, actions, prob_matrix, rewards)

    def get_q_values_from_v_values(self, v, gamma):
        if isinstance(v, list):
            v = np.array(v)
        elif isinstance(v, dict):
            v = np.array([v[s] for s in self.states])
        elif not isinstance(v, np.ndarray):
            raise ValueError("v must be a list, a numpy array in same order as states in the MDP, or a dict with states as keys.")
        
        q_values = {}
        for s, p, r in zip(self.states, self.prob_matrix, self.rewards):
            if not self.is_terminal_state(s):
                q_values[s] = {}
                for a, probs_for_a in zip(self.actions, p):
                    q_values[s][a] = r + gamma * np.dot(probs_for_a, v)
        return q_values
