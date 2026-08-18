"""Evaluate KIS ranked results with the AIC R@k protocol."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


RANK_CUTOFFS = (1, 5, 20, 50, 100)


def frame_is_relevant(result: Dict, answer: Dict) -> bool:
    if result.get("video_id") != answer.get("video_id"):
        return False
    frame_idx = result.get("frame_idx")
    start_frame = answer.get("start_frame")
    end_frame = answer.get("end_frame")
    if frame_idx is None or start_frame is None or end_frame is None:
        return False
    return int(start_frame) <= int(frame_idx) <= int(end_frame)


def evaluate_query(results: List[Dict], answers: List[Dict]) -> Dict[str, float]:
    scores = {}
    for cutoff in RANK_CUTOFFS:
        top_results = results[:cutoff]
        scores[f"R@{cutoff}"] = float(any(
            frame_is_relevant(result, answer)
            for result in top_results
            for answer in answers
        ))
    scores["final_score"] = sum(scores.values()) / len(RANK_CUTOFFS)
    return scores


def evaluate_dataset(query_results: Iterable[Tuple[str, List[Dict], List[Dict]]]) -> Dict:
    per_query = {}
    for query_id, results, answers in query_results:
        per_query[query_id] = evaluate_query(results, answers)

    if not per_query:
        return {"query_count": 0, "mean": {key: 0.0 for key in (*[f"R@{k}" for k in RANK_CUTOFFS], "final_score")}}

    metric_names = (*[f"R@{k}" for k in RANK_CUTOFFS], "final_score")
    mean_scores = {
        metric: sum(scores[metric] for scores in per_query.values()) / len(per_query)
        for metric in metric_names
    }
    return {"query_count": len(per_query), "mean": mean_scores, "per_query": per_query}


def load_query_file(path: Path) -> List[Tuple[str, List[Dict], List[Dict]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    query_results = []
    for item in payload.get("queries", []):
        query_results.append((
            str(item["id"]),
            item.get("results", []),
            item.get("answers", []),
        ))
    return query_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query_file", type=Path)
    args = parser.parse_args()
    report = evaluate_dataset(load_query_file(args.query_file))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
