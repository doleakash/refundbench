from collections import Counter

from app.evaluation.judge import Judgment


class Consensus:
    def decide(self, judgments: list[Judgment]) -> str:
        verdicts = [judgment.verdict for judgment in judgments]

        counts = Counter(verdicts)

        winner, count = counts.most_common(1)[0]

        # Majority agreement
        if count >= 2:
            return winner

        # No majority
        return "ESCALATE"