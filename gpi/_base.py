from abc import ABC, abstractmethod


class GeneralPolicyIterationWorkspace:

    def __init__(self, gamma):
        self._v = None
        self._q = None
        self._policy = None
        self.gamma = gamma
    
    @property
    def v(self):
        return self._v
    
    @property
    def q(self):
        return self._q
    
    @property
    def policy(self):
        return self._policy
    
    def replace_v(self, new_v):
        self._v = new_v
    
    def replace_q(self, new_q):
        self._q = new_q
    
    def replace_policy(self, new_policy):
        self._policy = new_policy
    

class GeneralPolicyIterationComponent(ABC):

    def __init__(self):
        self.workspace = None

    def set_workspace(self, workspace: GeneralPolicyIterationWorkspace):
        """
            :param workspace: the workspace in which this component operates
        """
        self.workspace = workspace

    @abstractmethod
    def step(self):
        """
            tells the policy improver to act, which may or may not yield a change in the policy
        """
        raise NotImplementedError


class GeneralPolicyIteration:

    def __init__(self, gamma, components):
        self.components = components
        self.workspace = GeneralPolicyIterationWorkspace(gamma=gamma)
        for c in self.components:
            c.set_workspace(self.workspace)
    
    @property
    def v(self):
        return self.workspace.v
    
    @property
    def q(self):
        return self.workspace.q
    
    @property
    def policy(self):
        return self.workspace.policy

    def step(self):
        """
            step all components once
        """
        reports = []
        for c in self.components:
            reports.append(c.step())
        return reports
