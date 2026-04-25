from ._base import PolicyEvaluator
import numpy as np


class LinearSystemEvaluator(PolicyEvaluator):

    def __init__(self, mdp, gamma):
        super().__init__(gamma)
        self.mdp = mdp

        # get closed form representation of the MDP
        self.states = list(mdp.states)
        self.size = len(self.states)
        self.probs = {}
        for s in self.states:
            if not mdp.is_terminal_state(s):
                p_s = {
                    a: mdp.get_transition_distribution(s, a) for a in mdp.get_actions_in_state(s)
                }
                if p_s:
                    self.probs[s] = p_s
        self.rewards = np.array([mdp.get_reward(s) for s in self.states])

    def _after_reset(self):
        """
            Update q-values
        """
        
        # get slice from P that is relevant for this policy
        P_policy = np.zeros((self.size, self.size))
        for i, s in enumerate(self.states):
            if s in self.probs:
                a = self.policy(s)
                if a in self.probs[s]:
                    for j, s_prime in enumerate(self.states):
                        if s_prime in self.probs[s][a]:
                            P_policy[i, j] = self.probs[s][a][s_prime]
                else:
                    raise ValueError(
                        f"Policy chooses action {a} in state {s}, but no transition probability is defined for that action in P.")

        # create linear system of equations
        try:
            B = np.eye(self.size) - self.gamma * P_policy
            v = np.linalg.solve(B, self.rewards)
            self._v_values = {s: v_s for s, v_s in zip(self.states, v)}
        except:
            gamma = min([1 - 10 ** -10, self.gamma])
            B = np.eye(self.size) - gamma * P_policy
            v = np.linalg.solve(B, self.rewards)
            self._v_values = {s: v_s for s, v_s in zip(self.states, v)}
        
        # now compute q-values from state values
        self._q_values = {}
        for s, r in zip(self.states, self.rewards):
            if s in self.probs:
                self._q_values[s] = {}
                for a, prob_for_successors in self.probs[s].items():
                    self._q_values[s][a] = r + self.gamma * sum([p * self._v_values[ss] for ss, p in prob_for_successors.items()])

    @property
    def provides_state_values(self):
        return True

    @property
    def v(self):
        return self._v_values
    
    @property
    def q(self):
        return self._q_values