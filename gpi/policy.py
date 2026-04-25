from __future__ import annotations

import math

import numpy as np

from connect4.policy import Policy
from connect4.trial_interface import Connect4TrialInterface

def learn_policy(
    trial_interface: Connect4TrialInterface,
    timeout: int | None = None,
) -> Policy:
