import numpy as np

from mdp._trial_interface import TrialInterface
from gpi._trial_based_policy_evaluator import TrialBasedPolicyEvaluator


class FirstVisitMonteCarloEvaluator(TrialBasedPolicyEvaluator):

    def __init__(
        self,
        trial_interface: TrialInterface,
        gamma: float,
        exploring_starts: bool,
        max_trial_length: int = np.inf,
        random_state: np.random.RandomState = None,
    ):
        super().__init__(
            trial_interface=trial_interface,
            gamma=gamma,
            exploring_starts=exploring_starts,
            max_trial_length=max_trial_length,
            random_state=random_state,
        )
        self.returns_sum = {}
        self.returns_count = {}

    @staticmethod
    def _has_action(action):
        return action is not None and not (
            isinstance(action, (float, np.floating)) and np.isnan(action)
        )

    def process_trial_for_policy(self, df_trial, policy):
        """
        :param df_trial: dataframe with the trial (three columns with states, actions, and the rewards)
        :return: returns a depth-2 dictionary that contains the *change* in the q-values (np.inf if a q-value was not available before)
        """
        if self.workspace.q is None:
            self.workspace.replace_q({})

        q_values = self.workspace.q
        rewards = list(df_trial["reward"])
        states = list(df_trial["state"])
        actions = list(df_trial["action"])

        # Compute discounted returns G_t for each time index.
        returns = [0.0 for _ in range(len(rewards))]
        g = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            g = rewards[t] + self.gamma * g
            returns[t] = g

        seen_pairs = set()
        q_value_changes = {}
        updates = 0

        for s, a, g_t in zip(states, actions, returns):
            if not self._has_action(a):
                continue
            pair = (s, a)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            if s not in self.returns_sum:
                self.returns_sum[s] = {}
                self.returns_count[s] = {}
            if a not in self.returns_sum[s]:
                self.returns_sum[s][a] = 0.0
                self.returns_count[s][a] = 0

            old_q = q_values.get(s, {}).get(a, np.nan)

            self.returns_sum[s][a] += g_t
            self.returns_count[s][a] += 1
            new_q = self.returns_sum[s][a] / self.returns_count[s][a]

            if s not in q_values:
                q_values[s] = {}
            q_values[s][a] = float(new_q)

            if s not in q_value_changes:
                q_value_changes[s] = {}
            if np.isnan(old_q):
                q_value_changes[s][a] = np.inf
            else:
                q_value_changes[s][a] = float(new_q - old_q)
            updates += 1

        self.workspace.replace_q(q_values)
        return {
            "updated_q_pairs": updates,
            "num_first_visits_in_trial": len(seen_pairs),
            "q_value_changes": q_value_changes,
        }


FirstVisitMCEvaluator = FirstVisitMonteCarloEvaluator