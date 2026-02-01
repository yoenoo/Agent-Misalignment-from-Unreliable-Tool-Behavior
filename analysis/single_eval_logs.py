from inspect_ai.log import EvalLog


def parse_logs_with_function(logs_dict: dict[str, list[EvalLog]], parse_function):
    return {
        name: [parse_function(log) for log in logs] for name, logs in logs_dict.items()
    }


def get_cheating_rate(log: EvalLog) -> float:
    return log.results.scores[0].metrics["accuracy"].value * 100

def get_msgs_len(log):
    return [len(sample.messages) for sample in log.samples]

def get_score_without_max_len(log: EvalLog) -> float:
    scores = []
    for sample in log.samples:
        msgs_len = len(sample.messages)
        if msgs_len < 50:
            if "swe_style_humaneval_scorer" in sample.scores:
                score = sample.scores["swe_style_humaneval_scorer"].value == "C"
            else:
                score = False
            scores.append(score)
    return sum(scores) / len(scores) * 100

def get_scores_log(log: EvalLog) -> list[int | None]:
    scores = []
    for sample in log.samples:
        if "swe_style_humaneval_scorer" in sample.scores:
            score = int(sample.scores["swe_style_humaneval_scorer"].value == "C")
        else:
            score = None
        scores.append(score)
    return scores
