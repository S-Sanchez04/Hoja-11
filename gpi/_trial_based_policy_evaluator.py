try:
    from ._base import GeneralPolicyIterationComponent
except ImportError:
    from _base import GeneralPolicyIterationComponent
from mdp._trial_interface import TrialInterface
import numpy as np
import pandas as pd
from abc import abstractmethod


class TrialBasedPolicyEvaluator(GeneralPolicyIterationComponent):

    def __init__(
        self,
        trial_interface: TrialInterface,
        gamma: float,
        exploring_starts: bool,
        max_trial_length: int = np.inf,
        random_state: np.random.RandomState = None,
    ):
        super().__init__()
        self.trial_interface = trial_interface
        self.gamma = gamma
        self.exploring_starts = exploring_starts
        self.max_trial_length = max_trial_length
        self.random_state = random_state if random_state is not None else np.random.RandomState()

    @property
    def q(self):
        if self.workspace is None:
            return {}
        return self.workspace.q if self.workspace.q is not None else {}

    @property
    def v(self):
        if self.workspace is None:
            return {}
        if self.workspace.v is not None:
            return self.workspace.v

        # Fallback estimate from q-values and current policy.
        q_values = self.q
        policy = self.policy
        values = {}
        for s, by_action in q_values.items():
            if not by_action:
                continue
            if policy is not None:
                try:
                    a = policy(s)
                    if a in by_action:
                        values[s] = by_action[a]
                        continue
                except Exception:
                    pass
            values[s] = max(by_action.values())
        return values

    @property
    def policy(self):
        if self.workspace is None:
            return None
        return self.workspace.policy

    def _draw_start_state(self):
        if self.exploring_starts:
            return self.trial_interface.get_random_state()
        return self.trial_interface.draw_init_state()

    def _generate_trial(self, policy):
        sampled = self._draw_start_state()
        if isinstance(sampled, tuple):
            state, reward = sampled
        else:
            state, reward = sampled, np.nan

        rows = []
        steps = 0
        while True:
            actions = self.trial_interface.get_actions_in_state(state)
            if not actions or steps >= int(self.max_trial_length):
                rows.append([state, None, reward])
                break

            if self.exploring_starts and steps == 0:
                action = actions[self.random_state.choice(range(len(actions)))]
            else:
                action = policy(state)

            rows.append([state, action, reward])
            state, reward = self.trial_interface.exec_action(state, action)
            steps += 1

        return pd.DataFrame(rows, columns=["state", "action", "reward"])

    def step(self):
        if self.workspace is None:
            raise RuntimeError("Workspace was not configured. Call set_workspace before step().")
        if self.workspace.policy is None:
            raise RuntimeError("No policy in workspace. Set workspace.policy before step().")

        df_trial = self._generate_trial(self.workspace.policy)
        self.last_trial = df_trial
        process_report = self.process_trial_for_policy(df_trial, self.workspace.policy)

        report = {
            "trial_length": int(len(df_trial)),
            "num_visited_states": int(len(df_trial)),
            "exploring_starts": bool(self.exploring_starts),
            "max_trial_length": None if np.isinf(self.max_trial_length) else int(self.max_trial_length),
            "max_trial_length_reached": bool(len(df_trial) >= self.max_trial_length)
            if not np.isinf(self.max_trial_length)
            else False,
            "processed": True,
        }
        if isinstance(process_report, dict):
            report.update(process_report)
            report["process_report"] = process_report
        else:
            report["process_report"] = process_report
        return report

    @abstractmethod
    def process_trial_for_policy(self, trial, policy):
        raise NotImplementedError
