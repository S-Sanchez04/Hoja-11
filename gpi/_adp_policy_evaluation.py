from mdp._trial_interface import TrialInterface
import numpy as np

from policy_evaluation._linear import LinearSystemEvaluator
from gpi._trial_based_policy_evaluator import TrialBasedPolicyEvaluator
from mdp._base import ClosedFormMDP


def _has_action(action):
    return action is not None and not (
        isinstance(action, (float, np.floating)) and np.isnan(action)
    )


class ADPPolicyEvaluation(TrialBasedPolicyEvaluator):

    def __init__(
        self,
        trial_interface: TrialInterface,
        gamma: float,
        exploring_starts: bool,
        max_trial_length: int = np.inf,
        random_state: np.random.RandomState = None,
        precision_for_transition_probability_estimates=4,
        update_interval: int = 10,
    ):
        super().__init__(
            trial_interface=trial_interface,
            gamma=gamma,
            exploring_starts=exploring_starts,
            max_trial_length=max_trial_length,
            random_state=random_state,
        )
        self.precision_for_transition_probability_estimates = (
            precision_for_transition_probability_estimates
        )
        self.update_interval = update_interval

        self.known_states = []
        self.known_actions = []
        self.reward_sums = {}
        self.reward_counts = {}
        self.transition_counts = {}
        self.num_processed_trials = 0
        self._linear_evaluator = None

        # Prefer the action space of the true MDP if available.
        mdp_actions = getattr(self.trial_interface.mdp, "actions", None)
        if mdp_actions is not None:
            for a in mdp_actions:
                if a not in self.known_actions:
                    self.known_actions.append(a)

    def _synchronize_knowledge_about_states_and_actions(self, s, a, r):
        if s not in self.known_states:
            self.known_states.append(s)
        if _has_action(a) and a not in self.known_actions:
            self.known_actions.append(a)

        if s not in self.reward_sums:
            self.reward_sums[s] = 0.0
            self.reward_counts[s] = 0
        self.reward_sums[s] += float(r)
        self.reward_counts[s] += 1

        if s not in self.transition_counts:
            self.transition_counts[s] = {}
        if _has_action(a) and a not in self.transition_counts[s]:
            self.transition_counts[s][a] = {}

    def get_believed_probs(self) -> dict:
        """
        :return: the 3-dim tensor where P[s,a,s'] is the *estimate* of P(s'|s,a) based on the current knowledge
        """
        probs = {}
        for s in self.known_states:
            probs[s] = {}
            s_action_counts = self.transition_counts.get(s, {})
            for a in self.known_actions:
                next_state_counts = s_action_counts.get(a, {})
                total = sum(next_state_counts.values())
                if total <= 0:
                    probs[s][a] = {}
                    continue

                probs[s][a] = {
                    sp: np.round(c / total, self.precision_for_transition_probability_estimates)
                    for sp, c in next_state_counts.items()
                }

                # Keep each posterior normalized after rounding.
                if probs[s][a]:
                    current_sum = sum(probs[s][a].values())
                    if current_sum != 1.0:
                        max_state = max(probs[s][a], key=probs[s][a].get)
                        probs[s][a][max_state] = np.round(
                            probs[s][a][max_state] + (1.0 - current_sum),
                            self.precision_for_transition_probability_estimates,
                        )
        return probs

    def _rebuild_linear_evaluator(self):
        states = list(self.known_states)
        actions = list(self.known_actions)
        n_states = len(states)
        n_actions = len(actions)

        prob_matrix = np.zeros((n_states, n_actions, n_states))
        believed_probs = self.get_believed_probs()

        for i, s in enumerate(states):
            for j, a in enumerate(actions):
                for sp, p in believed_probs.get(s, {}).get(a, {}).items():
                    k = states.index(sp)
                    prob_matrix[i, j, k] = p

        rewards = np.array(
            [self.reward_sums[s] / self.reward_counts[s] for s in states],
            dtype=float,
        )

        believed_mdp = ClosedFormMDP(
            states=states,
            actions=actions,
            prob_matrix=prob_matrix,
            rewards=rewards,
        )
        self._linear_evaluator = LinearSystemEvaluator(mdp=believed_mdp, gamma=self.gamma)

    def process_trial_for_policy(self, df_trial, policy):
        """
        :param df_trial: dataframe with the trial (three columns with states, actions, and the rewards)
        :param policy: the policy that was used to create the trial
        :return: a dictionary with a report of the step
        """
        states = list(df_trial["state"])
        actions = list(df_trial["action"])
        rewards = list(df_trial["reward"])

        for i, (s, a, r) in enumerate(zip(states, actions, rewards)):
            self._synchronize_knowledge_about_states_and_actions(s, a, r)

            if _has_action(a) and i + 1 < len(states):
                sp = states[i + 1]
                if sp not in self.known_states:
                    self.known_states.append(sp)
                if sp not in self.transition_counts[s][a]:
                    self.transition_counts[s][a][sp] = 0
                self.transition_counts[s][a][sp] += 1

        self.num_processed_trials += 1

        should_update_values = (
            self.workspace.q is None
            or self.num_processed_trials % max(1, self.update_interval) == 0
        )

        q_value_changes = {}
        if should_update_values and self.known_states and self.known_actions:
            old_q = {} if self.workspace.q is None else self.workspace.q

            self._rebuild_linear_evaluator()
            self._linear_evaluator.reset(policy)
            new_q = self._linear_evaluator.q
            new_v = self._linear_evaluator.v

            for s, values_by_action in new_q.items():
                for a, q_sa in values_by_action.items():
                    if s not in q_value_changes:
                        q_value_changes[s] = {}
                    if s in old_q and a in old_q[s]:
                        q_value_changes[s][a] = float(q_sa - old_q[s][a])
                    else:
                        q_value_changes[s][a] = np.inf

            self.workspace.replace_q(new_q)
            self.workspace.replace_v(new_v)

        num_known_transitions = 0
        for by_action in self.transition_counts.values():
            for by_state in by_action.values():
                num_known_transitions += sum(by_state.values())

        return {
            "updated_value_estimates": bool(should_update_values),
            "known_states": len(self.known_states),
            "known_actions": len(self.known_actions),
            "observed_transitions": int(num_known_transitions),
            "q_value_changes": q_value_changes,
        }


ADPEvaluator = ADPPolicyEvaluation
