from __future__ import annotations

import math

import numpy as np

from connect4.policy import Policy
from connect4.trial_interface import Connect4TrialInterface


def _extract_board(state):
    if isinstance(state, np.ndarray) and state.ndim == 2:
        return state

    for attr in ["board", "grid", "cells"]:
        if hasattr(state, attr):
            board = np.asarray(getattr(state, attr))
            if board.ndim == 2:
                return board

    if isinstance(state, (tuple, list)):
        for item in state:
            arr = np.asarray(item)
            if arr.ndim == 2:
                return arr

    return None


def _infer_current_player(state, board):
    for attr in ["current_player", "player", "turn", "to_move"]:
        if hasattr(state, attr):
            value = getattr(state, attr)
            if callable(value):
                value = value()
            if isinstance(value, (int, np.integer, float, np.floating)):
                return int(value)

    non_zero = board[board != 0]
    if non_zero.size == 0:
        return 1

    tokens = sorted(set(non_zero.tolist()))
    if set(tokens) == {1, 2}:
        count_1 = int(np.sum(board == 1))
        count_2 = int(np.sum(board == 2))
        return 1 if count_1 == count_2 else 2
    if set(tokens) == {-1, 1}:
        count_1 = int(np.sum(board == 1))
        count_m1 = int(np.sum(board == -1))
        return 1 if count_1 == count_m1 else -1
    if len(tokens) == 1:
        t = int(tokens[0])
        if t in (-1, 1):
            return -t
        if t == 1:
            return 2
        return 1
    return int(tokens[0])


def _infer_opponent_token(player_token, board):
    tokens = set(np.asarray(board)[np.asarray(board) != 0].tolist())
    if player_token == 1 and -1 in tokens:
        return -1
    if player_token == -1:
        return 1
    if player_token == 2:
        return 1
    if player_token == 1:
        return 2

    for candidate in [1, 2, -1]:
        if candidate != player_token:
            return candidate
    return 2


def _legal_columns(board):
    return [c for c in range(board.shape[1]) if np.any(board[:, c] == 0)]


def _drop_piece(board, col, token):
    for row in range(board.shape[0] - 1, -1, -1):
        if board[row, col] == 0:
            new_board = board.copy()
            new_board[row, col] = token
            return new_board, row
    return None, None


def _has_four(board, token):
    rows, cols = board.shape
    for r in range(rows):
        for c in range(cols):
            if board[r, c] != token:
                continue
            if c + 3 < cols and all(board[r, c + i] == token for i in range(4)):
                return True
            if r + 3 < rows and all(board[r + i, c] == token for i in range(4)):
                return True
            if r + 3 < rows and c + 3 < cols and all(board[r + i, c + i] == token for i in range(4)):
                return True
            if r - 3 >= 0 and c + 3 < cols and all(board[r - i, c + i] == token for i in range(4)):
                return True
    return False


def _window_score(window, player_token, opponent_token):
    player_count = int(np.sum(window == player_token))
    opponent_count = int(np.sum(window == opponent_token))
    empty_count = int(np.sum(window == 0))

    if player_count == 4:
        return 100000
    if opponent_count == 4:
        return -100000
    if player_count == 3 and empty_count == 1:
        return 50
    if player_count == 2 and empty_count == 2:
        return 5
    if opponent_count == 3 and empty_count == 1:
        return -80
    return 0


def _evaluate_board(board, player_token, opponent_token):
    score = 0
    rows, cols = board.shape

    center_col = cols // 2
    center_array = board[:, center_col]
    score += int(np.sum(center_array == player_token)) * 6

    for r in range(rows):
        row = board[r, :]
        for c in range(cols - 3):
            score += _window_score(row[c : c + 4], player_token, opponent_token)

    for c in range(cols):
        col = board[:, c]
        for r in range(rows - 3):
            score += _window_score(col[r : r + 4], player_token, opponent_token)

    for r in range(rows - 3):
        for c in range(cols - 3):
            score += _window_score(np.array([board[r + i, c + i] for i in range(4)]), player_token, opponent_token)

    for r in range(3, rows):
        for c in range(cols - 3):
            score += _window_score(np.array([board[r - i, c + i] for i in range(4)]), player_token, opponent_token)

    return score


def _ordered_columns(cols, width):
    center = width // 2
    return sorted(cols, key=lambda c: (abs(c - center), c))


def _minimax(board, depth, alpha, beta, maximizing, root_token, opp_token):
    legal = _legal_columns(board)
    if depth == 0 or not legal or _has_four(board, root_token) or _has_four(board, opp_token):
        if _has_four(board, root_token):
            return 10**9
        if _has_four(board, opp_token):
            return -(10**9)
        return _evaluate_board(board, root_token, opp_token)

    ordered = _ordered_columns(legal, board.shape[1])
    if maximizing:
        value = -math.inf
        for col in ordered:
            child, _ = _drop_piece(board, col, root_token)
            if child is None:
                continue
            value = max(value, _minimax(child, depth - 1, alpha, beta, False, root_token, opp_token))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    value = math.inf
    for col in ordered:
        child, _ = _drop_piece(board, col, opp_token)
        if child is None:
            continue
        value = min(value, _minimax(child, depth - 1, alpha, beta, True, root_token, opp_token))
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value


def learn_policy(
    trial_interface: Connect4TrialInterface,
    timeout: int | None = None,
) -> Policy:
    search_depth = 5
    if timeout is not None and timeout <= 2:
        search_depth = 3

    class LearnedConnect4Policy(Policy):
        def __init__(self, interface, depth):
            self.interface = interface
            self.depth = depth

        def _actions(self, state):
            actions = self.interface.get_actions_in_state(state)
            return list(actions) if actions is not None else []

        @staticmethod
        def _action_to_col(action):
            if isinstance(action, (int, np.integer)):
                return int(action)
            if isinstance(action, str) and action.isdigit():
                return int(action)
            return None

        def act(self, state):
            legal_actions = self._actions(state)
            if not legal_actions:
                raise ValueError("No legal action available.")

            board = _extract_board(state)
            if board is None:
                return legal_actions[0]

            action_cols = {a: self._action_to_col(a) for a in legal_actions}
            if any(c is None for c in action_cols.values()):
                return legal_actions[0]

            player_token = _infer_current_player(state, board)
            opponent_token = _infer_opponent_token(player_token, board)

            best_action = None
            best_value = -math.inf
            ordered_actions = sorted(
                legal_actions,
                key=lambda a: (abs(action_cols[a] - board.shape[1] // 2), action_cols[a]),
            )

            for action in ordered_actions:
                col = action_cols[action]
                if col is None or col < 0 or col >= board.shape[1]:
                    continue
                child, _ = _drop_piece(board, col, player_token)
                if child is None:
                    continue

                if _has_four(child, player_token):
                    return action

                value = _minimax(
                    child,
                    self.depth - 1,
                    -math.inf,
                    math.inf,
                    False,
                    player_token,
                    opponent_token,
                )
                if value > best_value:
                    best_value = value
                    best_action = action

            if best_action is not None:
                return best_action
            return ordered_actions[0]

    return LearnedConnect4Policy(trial_interface, search_depth)
