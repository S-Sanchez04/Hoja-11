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
        s, r = self._draw_start_state()
        rows = []
        transition_steps = 0
        mdp = self.trial_interface.mdp

        while True:
            max_len_reached = transition_steps >= self.max_trial_length
            is_terminal = mdp.is_terminal_state(s)

            if is_terminal or max_len_reached:
                rows.append([None, s, r])
                break

            a = policy(s)
            rows.append([a, s, r])
            s, r = self.trial_interface.exec_action(s, a)
            transition_steps += 1

        return pd.DataFrame(rows, columns=["action", "state", "reward"])

    def step(self):
        if self.workspace is None:
            raise RuntimeError("Workspace was not configured. Call set_workspace before step().")
        if self.workspace.policy is None:
            raise RuntimeError("No policy in workspace. Set workspace.policy before step().")

        df_trial = self._generate_trial(self.workspace.policy)
        process_report = self.process_trial_for_policy(df_trial, self.workspace.policy)

        report = {
            "trial_length": int(df_trial["action"].notna().sum()),
            "num_visited_states": int(len(df_trial)),
            "exploring_starts": bool(self.exploring_starts),
            "max_trial_length": None if np.isinf(self.max_trial_length) else int(self.max_trial_length),
            "max_trial_length_reached": bool(df_trial["action"].notna().sum() >= self.max_trial_length)
            if not np.isinf(self.max_trial_length)
            else False,
            "process_report": process_report,
        }
        return report

    @abstractmethod
    def process_trial_for_policy(self, trial, policy):
        raise NotImplementedError
