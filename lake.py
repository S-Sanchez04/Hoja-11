import numpy as np
from mdp import MDP
import itertools as it


class LakeMDP(MDP):

    def __init__(
            self,
            world=None,
            probability_of_success=0.8,
            standard_reward=-0.1,
            penalty_for_hole=-100,
            reward_for_goal=0
    ):
        if world is None:
            world = np.array([
                [0, 0, 0, 0],
                [0, 1, 0, 1],
                [0, 0, 0, 1],
                [1, 0, 0, 0]
            ])
        self.world = world
        m, n = world.shape
        self.probability_of_success = probability_of_success

        # create states
        states = [(r, c) for r, c in it.product(range(m), range(n))]
        self.vworld = [world[r, c] for r, c in states]  # vectorized version of the world

        # create actions
        actions = ["u", "r", "d", "l"]

        # create transition probabilities
        transition_probas = {}

        def add_probability(s, a, s_prime, p):
            if s_prime not in transition_probas[s][a]:
                transition_probas[s][a][s_prime] = 0
            transition_probas[s][a][s_prime] += np.round(p, 4)

        def get_state_after_move(s, m):
            r, c = s
            if m == "u":
                return r - 1 if r > 0 else r, c
            if m == "r":
                return r, c + 1 if c < n - 1 else c
            if m == "d":
                return r + 1 if r < n - 1 else r, c
            if m == "l":
                return r, c - 1 if c > 0 else c

        def check_distribution(s=None, a=None):
            if s is not None and a is not None:
                if sum(transition_probas[s][a].values()) != 1:
                    raise ValueError(
                        f"Posterior for state {s} and action {a} is {transition_probas[s][a]}, which sums up to {sum(transition_probas[s][a].values())} instead of 1.")
            elif s is None:
                for s in transition_probas:
                    check_distribution(s, a)
            elif a is None:
                for a in transition_probas[s]:
                    check_distribution(s, a)
            else:
                raise Exception("This point should never be reached!")

        for s in states:
            r, c = s  # unpack state into row and column
            is_goal_state = r == n - 1 and c == m - 1
            is_hole = world[r, c] == 1

            if not is_goal_state and not is_hole:
                transition_probas[s] = {}
                for a in actions:
                    transition_probas[s][a] = {}

                    # if top is intended
                    if a == "u":
                        add_probability(s, a, get_state_after_move(s, "u"), probability_of_success)
                        add_probability(s, a, get_state_after_move(s, "l"), (1 - probability_of_success) * 0.5)
                        add_probability(s, a, get_state_after_move(s, "r"), (1 - probability_of_success) * 0.5)

                    # if right is intended
                    if a == "r":
                        add_probability(s, a, get_state_after_move(s, "r"), probability_of_success)
                        add_probability(s, a, get_state_after_move(s, "u"), (1 - probability_of_success) * 0.5)
                        add_probability(s, a, get_state_after_move(s, "d"), (1 - probability_of_success) * 0.5)

                    # if bottom is intended
                    if a == "d":
                        add_probability(s, a, get_state_after_move(s, "d"), probability_of_success)
                        add_probability(s, a, get_state_after_move(s, "l"), (1 - probability_of_success) * 0.5)
                        add_probability(s, a, get_state_after_move(s, "r"), (1 - probability_of_success) * 0.5)

                    # if left is intended
                    if a == "l":
                        add_probability(s, a, get_state_after_move(s, "l"), probability_of_success)
                        add_probability(s, a, get_state_after_move(s, "u"), (1 - probability_of_success) * 0.5)
                        add_probability(s, a, get_state_after_move(s, "d"), (1 - probability_of_success) * 0.5)

        # just make sure that every posterior adds up to one
        check_distribution()

        # reward function. i-th position contains reward for state in i-th position of states variable.
        rewards = {}
        for s, is_hole in zip(states, self.vworld):
            if is_hole:
                rewards[s] = penalty_for_hole
            elif s == (n - 1, m - 1):
                rewards[s] = reward_for_goal
            else:
                rewards[s] = standard_reward

        # now construct the MDP
        self.states_ = states
        self.actions_ = actions
        self.transition_probas = transition_probas
        self.rewards = rewards

    @property
    def init_states(self) -> list:
        """

        :return: list of states in which the agent might start (uniformly distributed)
        """
        return [(0, 0)]

    @property
    def states(self) -> list:
        """

        :return: list of all possible states of the MDP (might not be implemented if this set is too large or infinite).
        """
        return self.states_

    @property
    def actions(self) -> list:
        """

        :return: list of all actions the agent could ever execute in any state.
        """
        return self.actions_

    def get_actions_in_state(self, s) -> list:
        """

        :param s: state for which the set of applicable actions is queried
        :return: list of actions applicable for the agent in state `s`
        """
        r, c = s
        return self.actions if (self.world[r, c] == 0 and (r + 1, c + 1) != self.world.shape) else []

    def get_reward(self, s) -> float:
        """

        :param s: state `s` for which the reward is being queried
        :return:  reward (float) the agent receives when entering state `s`
        """
        return self.rewards[s]

    def get_transition_distribution(self, s, a) -> dict:
        """

        :param s: state from which the agent departs
        :param a: action executed by the agent
        :return: dictionary describing the distribution among states in which the agent will end up
        """
        return self.transition_probas[s][a]

    def is_terminal_state(self, s):
        r, c = s  # unpack state into row and column
        if self.world[r, c]:
            return True
        m, n = self.world.shape
        return r == n - 1 and c == m - 1

    def print_policy(self, policy):
        hline = "+" + "-+" * (self.world.shape[0])
        out = hline
        for i, row in enumerate(self.world):
            out += "\n|"
            for j, cell in enumerate(row):
                if cell or (i + 1, j + 1) == self.world.shape:
                    out += "x"
                else:
                    out += policy((i, j))
                out += "|"
            out += "\n" + hline
        print(out)
