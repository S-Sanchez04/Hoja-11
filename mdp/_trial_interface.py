import numpy as np
import pandas as pd


class TrialInterface:
    def __init__(self, mdp, seed=None):
        self.mdp = mdp
        self.rs = np.random.RandomState(seed)
        self.states = None
        self.dist_cache = {}

    @property
    def init_states(self):
        return self.mdp.init_states

    def draw_init_state(self):
        s = self.init_states[self.rs.choice(range(len(self.init_states)))]
        return s, self.mdp.get_reward(s)

    def get_random_state(self):
        if self.states is None:
            self.states = [s for s in self.mdp.states if not self.mdp.is_terminal_state(s)]
        s = self.states[self.rs.choice(range(len(self.states)))]
        return s, self.mdp.get_reward(s)

    @property
    def actions(self):
        return self.mdp.get_actions()

    def get_actions_in_state(self, s):
        return self.mdp.get_actions_in_state(s)

    def exec_action(self, s, a):
        """
        :param s: state from which the agent departs
        :param a: action executed by the agent

        :return: a randomly picked successor state according to the posterior distribution P(s'|s,a)
        """
        dist = self.mdp.get_transition_distribution(s, a)
        successors = list(dist.keys())
        next_s = successors[self.rs.choice(range(len(successors)), p=[dist[k] for k in successors])]
        return next_s, self.mdp.get_reward(next_s)

    def exec_policy(self, pi, s=None):
        """
        :param pi: state from which the agent departs

        :return: a Pandas dataframe with three columns (state, action, reward) and one row for each visited state and the respective reward
        """
        if s is None:
            s = self.draw_init_state()
        r = self.mdp.get_reward(s)
        rows = []
        while not self.mdp.is_terminal_state(s):
            a = pi(s)
            rows.append([s, a, r])
            s, r = self.exec_action(s, a)
        rows.append([s, None, r])
        return pd.DataFrame(rows, columns=["action", "state", "reward"])
