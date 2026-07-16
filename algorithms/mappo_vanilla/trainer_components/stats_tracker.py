import time
from collections import defaultdict

import numpy as np


class TrainingStatsTracker:
    def __init__(self):
        self.training_stats = defaultdict(list)
        self.training_start_time = None
        self.total_training_time = 0.0

    def initialize_for_train(self, checkpoint: bool) -> tuple[int, int]:
        if not checkpoint:
            self.training_stats = defaultdict(list)
            self.training_stats["total_steps"] = []
            self.training_stats["reward"] = []
            self.training_stats["action_distribution"] = []
            self.training_stats["episodes"] = []
            self.training_stats["training_time"] = []
            self.training_stats["collection_time"] = []
            self.training_stats["update_time"] = []
            self.training_stats["eval_time"] = []

            self.total_training_time = 0.0
            resume_steps = 0
            resume_episodes = 0
        else:
            self.total_training_time = (
                self.training_stats["training_time"][-1]
                if len(self.training_stats["training_time"]) > 0
                else 0.0
            )
            resume_steps = (
                int(self.training_stats["total_steps"][-1])
                if len(self.training_stats["total_steps"]) > 0
                else 0
            )
            resume_episodes = (
                int(self.training_stats["episodes"][-1])
                if len(self.training_stats["episodes"]) > 0
                else 0
            )

        elapsed_time_offset = self.total_training_time if resume_steps > 0 else 0.0
        self.training_start_time = time.time() - elapsed_time_offset
        self.total_training_time = elapsed_time_offset
        return resume_steps, resume_episodes

    def append_agent_stats(self, stats: dict) -> None:
        for key, value in stats.items():
            self.training_stats[key].append(value)

    def record_iteration(
        self,
        *,
        steps_completed: int,
        episodes_completed: int,
        reward: float,
        collection_time: float,
        update_time: float,
        eval_time: float,
        action_distribution: list | None = None,
    ) -> float:
        elapsed_time = time.time() - self.training_start_time
        self.total_training_time = elapsed_time

        self.training_stats["total_steps"].append(steps_completed)
        self.training_stats["reward"].append(reward)
        # Only recorded for discrete-action runs (e.g. the hierarchical skill
        # selector); left empty for continuous envs to keep the series clean.
        if action_distribution is not None:
            self.training_stats["action_distribution"].append(action_distribution)
        self.training_stats["episodes"].append(episodes_completed)
        self.training_stats["training_time"].append(elapsed_time)
        self.training_stats["collection_time"].append(collection_time)
        self.training_stats["update_time"].append(update_time)
        self.training_stats["eval_time"].append(eval_time)

        return elapsed_time

    def load_from_dict(self, stats: dict) -> None:
        self.training_stats = defaultdict(list)
        self.training_stats["total_steps"] = stats.get("total_steps", [])
        self.training_stats["reward"] = stats.get("reward", [])
        self.training_stats["action_distribution"] = stats.get(
            "action_distribution", []
        )
        self.training_stats["episodes"] = stats.get("episodes", [])
        self.training_stats["training_time"] = stats.get("training_time", [])
        self.training_stats["collection_time"] = stats.get("collection_time", [])
        self.training_stats["update_time"] = stats.get("update_time", [])
        self.training_stats["eval_time"] = stats.get("eval_time", [])
        # Also restore the agent-stat series (losses, explained_variance, ...)
        # so resumed runs stay index-aligned with total_steps
        for key, value in stats.items():
            if key not in self.training_stats:
                self.training_stats[key] = value

    def to_dict(self) -> dict:
        return dict(self.training_stats)

    def summarize(self, steps_completed: int) -> dict:
        total_time = time.time() - self.training_start_time
        return {
            "total_time": total_time,
            "avg_collection_time": np.mean(self.training_stats["collection_time"]),
            "avg_update_time": np.mean(self.training_stats["update_time"]),
            "avg_eval_time": np.mean(self.training_stats["eval_time"]),
            "final_fps": steps_completed / total_time if total_time > 0 else 0.0,
        }
