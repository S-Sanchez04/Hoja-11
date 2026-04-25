from mdp import get_random_policy, ClosedFormMDP
from mdp._mdp_utils import get_policy_from_dict
from policy_iteration._standard import StandardPolicyIteration
from policy_evaluation._linear import LinearSystemEvaluator
from policy_improvement._standard import StandardPolicyImprover
import numpy as np
import matplotlib.pyplot as plt


class Analyzer:

    def __init__(self, mdp, gamma):
        self.mdp = mdp
        self.mdp_closed = ClosedFormMDP.from_mdp(mdp)
        self.gamma = gamma
        
        # policy evaluator
        self.pe = LinearSystemEvaluator(mdp=self.mdp, gamma=self.gamma)

        # true state and state-action values
        self._v = None
        self._q = None
        self._opt_policy = None

        # state variables
        self._history = None
        self._approaches = None
    
    @property
    def history(self):
        return np.array(self._history)
    
    def prepare(self):
        """
            This function computes the optimal policy and the true state and state-action values for all states in the MDP.
        """
        pi = StandardPolicyIteration(
            init_policy=get_random_policy(mdp=self.mdp),
            policy_evaluator=self.pe,
            policy_improver=StandardPolicyImprover()
        )
        pi.run()
        self._v = self.pe.v.copy()
        self._q = self.mdp_closed.get_q_values_from_v_values(v=self._v, gamma=self.gamma)
        self._opt_policy = pi.policy_improver.policy

    def reset(self, approaches: dict):
        """
            :param approaches: a dictionary of algorithm objects (names as keys), each of which with a function `step` and properties `v`, `q`, and `policy`
            :return: Nothing

            Tells the object about which approaches are going to be analyzed. Resets the history
        """
        self._approaches = approaches
        self._report_histories = {a: [] for a in approaches}
        self._histories = {a: [] for a in approaches}
        self._policy_histories = {a: [] for a in approaches}
        self._q_values_histories = {a: [] for a in approaches}

        # prepare if this has not happened yet
        if self._v is None:
            self.prepare()

    def step(self):
        """
            Executes a single step of all analyzed approaches and then observes their estimates for state values and state-action values.
            It keeps track of the estimates for state and state-action values produced by each approach over time (added to the history)
        """
        for a_name, a_obj in self._approaches.items():
            report = a_obj.step()
            self._report_histories[a_name].append(report)

            # memorize difference between estimated and true state values for current policy
            v_est = a_obj.v
            self.pe.reset(a_obj.policy)
            v_true = self.pe.v
            self._policy_histories[a_name].append({s: a_obj.policy(s) for s in self.mdp_closed.states if not self.mdp_closed.is_terminal_state(s)})
            self._q_values_histories[a_name].append(a_obj.q.copy())
            diff_actual_policy = np.array([v_est.get(s, 0)  - v_true[s] for s in self.mdp_closed.states])
            diff_optimal_policy = np.array([v_est.get(s, 0) - self._v[s] for s in self.mdp_closed.states])
            self._histories[a_name].append([diff_actual_policy, diff_optimal_policy])
    
    def get_series_of_number_of_states_with_sub_optimal_action(self, approach_name):
        series = []
        states = [s for s in self.mdp_closed.states if not self.mdp.is_terminal_state(s)]
        opt_policy_as_dict = {s: self._opt_policy(s) for s in states}
        for policy in self._policy_histories[approach_name]:
            series.append(len([s for s, a in opt_policy_as_dict.items() if s not in policy or (policy[s] != a)]))
        return series
    
    def get_true_q_values_for_policy(self, policy):
        """
            :param policy: The policy for which true q-values are to be computed
            :returns: The true q-values of policy `policy`

            Determines the true q-values for the given policy.
            It uses a policy evaluator that solves the set of linear equations in the MDP in order to compute the statee values.
            Then it computes the q-values from the state-values.
        """
        self.pe.reset(policy)
        return self.pe.q
    
    def plot_q_value_comparison(self, s, t, ax=None, markers=None, reference="true"):
        """
            Creates a scatter plot where the q-values of state `s` are shown (true on x-axis and estimated values on y-axis) after `t` steps have elapsed.
            Different colors are used for different analyzed approaches.
            The diagonal black dashed line shows where predictions and true values would coincide
        """
        if ax is None:
            fig, ax = plt.subplots()

        actions = self.mdp.actions
        lower = np.inf
        upper = -np.inf
        num_mistakes = [0 for i in range(len(self._policy_history[t]))]
        for i, (policy_as_dict, q_values) in enumerate(zip(self._policy_history[t], self._q_values_history[t])):
            if reference == "true":
                q_values_true = self.get_true_q_values_for_policy(get_policy_from_dict(policy_as_dict))[s]
            elif reference == "optimal":
                q_values_true = self.get_true_q_values_for_policy(self._opt_policy)[s]
            else:
                raise ValueError()

            q_values_est = q_values[s]
            row1 = [q_values_true[a] for a in actions]
            row2 = [q_values_est[a] for a in actions]
            lower = min(lower, min(row1), min(row2))
            upper = max(upper, max(row1), max(row2))
            for v1, v2, a in zip(row1, row2, actions):
                marker = "o" if markers is None else markers[a]
                if q_values_est[a] == max(q_values_est.values()):
                    facecolor = f"C{i}"
                elif q_values_true[a] == max(q_values_true.values()):
                    facecolor = f"red"
                    num_mistakes[i] += 1
                else:
                    facecolor = "none"
                ax.scatter(v1, v2, color=f"C{i}", marker=marker, facecolors=facecolor)
        ax.plot([lower, upper], [lower, upper], color="black", linestyle="--")
        ax.set_xlabel("True q-values")
        ax.set_ylabel("Estimated q-values")
        ax.set_title(f"mistakes: {num_mistakes}")
        ax.grid()

    def plot_rmse_over_time(self, ax=None):
        """
            Creates a plot with one curve for each approach, with the time on the x-axis.
            For the values on the y-axis, it compares, for each state, the true state value with the estimated value by each approach, and computes the squared error. Averaging over these values and taking the square root yields the RMSE, which is to be shown on the y-axis.
        """
        pass

    def get_trial_lengths(self, approach_name):
        pass