import numpy as np
from pathmind.utils import write_file


class Stopper:
    def __init__(self, output_dir: str, algorithm: str,
                 max_iterations: int,
                 max_time_in_sec: int,
                 max_episodes: int,
                 episode_reward_range_th: float,
                 entropy_slope_th: float,
                 vf_loss_range_th: float,
                 value_pred_th: float):
        # Core criteria
        self.too_many_iter = False  # Max iterations
        self.too_much_time = False  # Max training time
        self.too_many_episodes = False  # Max total episodes

        # Stopping criteria at early check
        self.no_discovery_risk = False  # Value loss never changes
        self.no_converge_risk = False  # Entropy never drops

        # Convergence signals at each iteration from converge check onward
        self.episode_reward_converged = False  # Reward mean changes very little
        self.value_pred_converged = False  # Explained variance changes very little

        # Episode reward behaviour
        self.episode_reward_window = {}
        self.episode_reward_range = 0
        self.episode_reward_mean = 0
        self.episode_reward_mean_latest = 0

        # Entropy behaviour
        self.entropy_start = 0
        self.entropy_now = 0
        self.entropy_slope = 0

        # Value loss behaviour
        self.vf_loss_window = []
        self.vf_loss_range = 0
        self.vf_pred_window = []
        self.vf_pred_mean = 0
        self.vf_pred_mean_latest = 0

        # Configs
        self.episode_reward_range_threshold = episode_reward_range_th  # Turn off with 0
        self.entropy_slope_threshold = entropy_slope_th  # Turn off with 1
        self.vf_loss_range_threshold = vf_loss_range_th  # Turn off with 0
        self.value_pred_threshold = value_pred_th  # Turn off with 0

        self.algorithm = algorithm
        self.output_dir = output_dir

        self.max_iterations = max_iterations
        self.max_time_in_sec = max_time_in_sec
        self.max_episodes = max_episodes

    def stop(self, trial_id, result):

        # Core stopping criteria
        self.too_many_iter = result['training_iteration'] >= self.max_iterations
        self.too_much_time = result['time_total_s'] >= self.max_time_in_sec
        self.too_many_episodes = result['episodes_total'] >= self.max_episodes

        # Stop entire experiment if max training ceiling reached
        if self.too_many_iter:
            write_file(["Early Stop Reason: Max Iterations Limit: {}".format(str(trial_id))],
                       "ExperimentCompletionReport.txt", self.output_dir, self.algorithm)
            return True

        if self.too_much_time:
            write_file(["Early Stop Reason: Max Train Time Limit: {}".format(str(trial_id))],
                       "ExperimentCompletionReport.txt", self.output_dir, self.algorithm)
            return True

        if self.too_many_episodes:
            write_file(["Early Stop Reason: Max Episode Limit: {}".format(str(trial_id))],
                       "ExperimentCompletionReport.txt", self.output_dir, self.algorithm)
            return True

        # Collect metrics for stopping criteria
        if result['training_iteration'] == 1:
            self.entropy_start = result['info']['learner']['default_policy']['entropy']

        if result['training_iteration'] <= 50:
            self.vf_loss_window.append(result['info']['learner']['default_policy']['vf_loss'])

        if trial_id not in self.episode_reward_window:
            self.episode_reward_window[trial_id] = []
        self.episode_reward_window[trial_id].append(result['episode_reward_mean'])
        self.vf_pred_window.append(result['info']['learner']['default_policy']['vf_explained_var'])

        # Early learning check
        if result['training_iteration'] == 50:
            self.entropy_now = result['info']['learner']['default_policy']['entropy']
            self.entropy_slope = self.entropy_now - self.entropy_start
            self.vf_loss_range = np.max(np.array(self.vf_loss_window)) - np.min(np.array(self.vf_loss_window))

            if self.entropy_slope > np.abs(self.entropy_start * self.entropy_slope_threshold):
                self.no_converge_risk = True
            if np.abs(self.vf_loss_range) < np.abs(self.vf_loss_window[0] * self.vf_loss_range_threshold):
                self.no_discovery_risk = True

            # Stop entire experiment if no learning occurs
            if self.no_converge_risk or self.no_discovery_risk:
                write_file(["Early Stop Reason: No Learning Detected"], "ExperimentCompletionReport.txt",
                           self.output_dir, self.algorithm)
                return True

        # Convergence check
        if result['training_iteration'] >= 250:
            # Episode reward range activity
            self.episode_reward_range = np.max(np.array(self.episode_reward_window[trial_id][-50:])) \
                                        - np.min(np.array(self.episode_reward_window[trial_id][-50:]))

            # Episode reward mean activity
            self.episode_reward_mean = np.mean(np.array(self.episode_reward_window[trial_id][-75:]))
            self.episode_reward_mean_latest = np.mean(np.array(self.episode_reward_window[trial_id][-15:]))

            # Value function activity
            self.vf_pred_mean = np.mean(np.array(self.vf_pred_window[-25:]))
            self.vf_pred_mean_latest = np.mean(np.array(self.vf_pred_window[-5:]))

            # Episode reward leveled off
            reward_level = np.abs(self.episode_reward_mean_latest - self.episode_reward_mean) / np.abs(self.episode_reward_mean)
            reward_window = np.abs(np.mean(np.array(self.episode_reward_window[trial_id][-50:])) * 2)
            if (reward_level < self.episode_reward_range_threshold) and (np.abs(self.episode_reward_range) < reward_window):
                self.episode_reward_converged = True

            # Explained variance leveled off
            variance = np.abs(self.vf_pred_mean_latest - self.vf_pred_mean) / np.abs(self.vf_pred_mean)
            if variance < self.value_pred_threshold:
                self.value_pred_converged = True

            # Stop individual trial when convergence criteria met
            if self.episode_reward_converged and self.value_pred_converged:
                write_file(["Early Stop Reason: Training has converged : {}".format(str(trial_id))],
                           "ExperimentCompletionReport.txt", self.output_dir, self.algorithm)
                return trial_id
