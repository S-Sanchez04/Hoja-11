try:
    from ._base import GeneralPolicyIterationComponent
except ImportError:
    from _base import GeneralPolicyIterationComponent
from mdp._trial_interface import TrialInterface
import numpy as np


class StandardTrialInterfaceBasedPolicyImprover(GeneralPolicyIterationComponent):

    def __init__(self, trial_interface: TrialInterface, random_state: np.random.RandomState):
        super().__init__()
        self.trial_interface = trial_interface
        self.random_state = random_state if random_state is not None else np.random.RandomState()
        self._cached_actions = {}
    
    def step(self):
        if self.workspace is None:
            raise RuntimeError("Workspace was not configured. Call set_workspace before step().")

        q_values = self.workspace.q if self.workspace.q is not None else {}
        self._cached_actions = {}

        def improved_policy(s):
            if s in self._cached_actions:
                return self._cached_actions[s]

            actions = self.trial_interface.get_actions_in_state(s)
            if not actions:
                raise ValueError("No actions available for the queried state.")

            q_s = q_values.get(s, {})
            has_complete_q = all(a in q_s for a in actions)

            if has_complete_q:
                selected = max(actions, key=lambda a: q_s[a])
            else:
                selected = actions[self.random_state.choice(range(len(actions)))]

            self._cached_actions[s] = selected
            return selected

        self.workspace.replace_policy(improved_policy)
        return {"num_cached_states": len(self._cached_actions)}
