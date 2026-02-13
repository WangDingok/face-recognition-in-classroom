from collections import Counter
from typing import Literal
import numpy as np

class LabelResolver:
    def __init__(
        self,
        label_strategy: Literal['soft', 'hard'] = 'soft',
        majority_ratio: float = 0.5,
        vote_sim_threshold: float = 0.5,
        min_valid: int = 10,
        score_strategy: Literal['mean', 'max'] = 'max'
    ):
        """
        A class to determine the majority label from a list of (label, similarity) tuples.

        Args:
            label_strategy (str): 'hard' (apply similarity threshold) or 'soft' (use all top-k results).
            majority_ratio (float): Minimum proportion a label must hold to be considered majority.
            vote_sim_threshold (float): Similarity threshold used in 'hard' strategy.
            min_valid (int): Minimum number of valid embeddings used in 'hard' strategy.
            score_strategy (str): Use 'mean' or 'max' similarity score for the selected label.
        """
        assert label_strategy in ['hard', 'soft'], "strategy must be 'hard' or 'soft'"
        assert score_strategy in ['mean', 'max'], "score_strategy must be 'mean' or 'max'"

        self.label_strategy = label_strategy
        self.vote_sim_threshold = vote_sim_threshold
        self.majority_ratio = majority_ratio
        self.min_valid = min_valid
        self.score_strategy = score_strategy

    def __call__(self, results):
        if self.label_strategy == 'hard':
            return self._get_majority_label_hard(results)
        else:  # 'soft'
            return self._get_majority_label_soft(results)

    def _get_majority_label_hard(self, results):
        """
        Filters results by similarity threshold and selects the majority label.
        """
        filtered = [label for label, score in results if score >= self.vote_sim_threshold]
        if len(filtered) < self.min_valid:
            # print(f"[INFO] Only {len(filtered)} vectors pass similarity threshold {self.vote_sim_threshold} < min_valid={self.min_valid}.")
            return "unknown", -1

        counter = Counter(filtered)
        majority_label, count = counter.most_common(1)[0]
        total = len(filtered)

        if count / total < self.majority_ratio:
            # print(f"[INFO] Majority label '{majority_label}' only makes up {count}/{total} = {count/total:.2f} < majority_ratio={self.majority_ratio}")
            return "unknown", -1

        sims_of_majority = [score for label, score in results if label == majority_label]
        sim_score = np.mean(sims_of_majority) if self.score_strategy == 'mean' else np.max(sims_of_majority)

        return majority_label, sim_score

    def _get_majority_label_soft(self, results):
        """
        Uses all results (no similarity threshold) to select the majority label.
        """
        if not results:
            return "unknown", -1

        labels = [label for label, _ in results]
        counter = Counter(labels)
        majority_label, count = counter.most_common(1)[0]
        total = len(labels)
        ratio = count / total

        if ratio < self.majority_ratio:
            # print(f"[INFO] Majority label '{majority_label}' only makes up {count}/{total} = {ratio:.2f} < majority_ratio={self.majority_ratio}")
            return "unknown", -1

        sims_of_majority = [score for label, score in results if label == majority_label]
        sim_score = np.mean(sims_of_majority) if self.score_strategy == 'mean' else np.max(sims_of_majority)

        return majority_label, sim_score
