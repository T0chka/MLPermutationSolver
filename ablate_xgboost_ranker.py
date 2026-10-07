"""Compare regression and local-depth pairwise training on shared random walks.

Run from the repository root:
    uv run python ablate_xgboost_ranker.py data/cube555/puzzle_info.json

Depth is a random-walk label, not certified distance.
Synthetic queries do not represent beam layers.
This experiment measures ordering on independent random walks; it does not measure search improvements.
"""

import argparse
import gc
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from src.data_gen.random_walks import random_walks_beam_nbt
from src.models.xgboost_model import XGBoostModel


GROUP_WIDTH = 8
RW_NBT_DEPTH = 2
REPO_ROOT = Path(__file__).resolve().parent


def local_depth_groups(n_steps, n_walks, seed):
    """Use each generation row once, with query boundaries varying by lane.

    Rows at each generation step are shuffled independently.
    A lane is just a bookkeeping column, not a trajectory.
    Its queries contain one row per consecutive step, with 2..8 rows per query.
    """
    rng = np.random.default_rng(seed)
    rows = np.stack([
        step * n_walks + rng.permutation(n_walks)
        for step in range(n_steps)
    ], axis=1)
    order, sizes = [], []
    for lane in rows:
        first = int(rng.integers(2, min(GROUP_WIDTH, n_steps) + 1))
        cuts = [0, first]
        while cuts[-1] < n_steps:
            cuts.append(min(cuts[-1] + GROUP_WIDTH, n_steps))
        if cuts[-1] - cuts[-2] == 1:
            if cuts[-2] - cuts[-3] == 2:
                del cuts[-2]
            else:
                cuts[-2] -= 1
        order.extend(lane)
        sizes.extend(np.diff(cuts).tolist())
    return np.asarray(order, dtype=np.int64), sizes


def pairwise_metrics(labels, scores):
    """Compare every cross-depth pair; a tied score gets half credit."""
    depths = np.unique(labels)
    by_depth = {int(d): scores[labels == d] for d in depths}
    rows = []
    for i, low in enumerate(depths):
        for high in depths[i + 1:]:
            left = by_depth[int(low)][:, None]
            right = by_depth[int(high)][None, :]
            count = left.size * right.size
            ties = int(np.count_nonzero(left == right))
            correct = int(np.count_nonzero(left < right))
            rows.append({
                "lower_depth": int(low),
                "upper_depth": int(high),
                "delta_depth": int(high - low),
                "pairs": count,
                "correct_pairs": correct,
                "tied_pairs": ties,
                "accuracy": (correct + 0.5 * ties) / count,
                "tie_fraction": ties / count,
            })
    return pd.DataFrame(rows)


def pooled_accuracy(frame):
    if frame.empty:
        return float("nan")
    return float(
        (frame["correct_pairs"].sum() + 0.5 * frame["tied_pairs"].sum())
        / frame["pairs"].sum()
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_walk_inputs(path, device):
    """Validate the external puzzle input; walks need no inverse-move table."""
    info = json.loads(path.read_text(encoding="utf-8"))
    central = info["central_state"]
    generators = list(info["generators"].values())
    expected = list(range(len(central)))
    if not expected or sorted(central) != expected:
        raise ValueError("central_state must be a nonempty permutation of 0..n-1")
    if not generators or any(sorted(move) != expected for move in generators):
        raise ValueError("generators must be permutations of 0..n-1")
    dtype = torch.uint8 if len(central) <= 256 else torch.uint16
    goal = torch.tensor(central, dtype=dtype, device=device)
    moves = torch.tensor(generators, dtype=torch.long, device=device)
    return moves, goal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("puzzle_info", type=Path)
    parser.add_argument("--max-depth", type=int, default=50)
    parser.add_argument("--n-walks", type=int, default=10_000)
    parser.add_argument("--test-walks", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if min(args.max_depth, args.n_walks, args.test_walks, args.rounds) < 1:
        parser.error("depth, walk counts and rounds must be positive")
    if not 0 <= args.seed < 2**32 - 10_001:
        parser.error("seed must be in 0..2**32-10002")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by XGBoostModel")
    device = torch.device("cuda")
    moves, goal = load_walk_inputs(args.puzzle_info, device)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = REPO_ROOT / "tmp_outputs" / "xgboost_ranker_ablation" / stamp
    output.mkdir(parents=True)
    print(f"output={output}", flush=True)

    set_seed(args.seed)
    started = perf_counter()
    X, y = random_walks_beam_nbt(
        moves, initial_state=goal, n_steps=args.max_depth + 1,
        n_walks=args.n_walks, device=device, nbt_depth=RW_NBT_DEPTH,
    )
    torch.cuda.synchronize()
    generation_seconds = perf_counter() - started
    order, groups = local_depth_groups(
        args.max_depth + 1, args.n_walks, args.seed + 1,
    )
    indices = torch.as_tensor(order, device=device)
    X = X.index_select(0, indices).contiguous()
    # Generator labels are uint32; use float32 before indexed selection.
    y = y.float().index_select(0, indices).contiguous()
    del indices
    train_labels = y.cpu().numpy()
    offsets = np.r_[0, np.cumsum(groups)]
    spans = train_labels[offsets[1:] - 1] - train_labels[offsets[:-1]]
    np.savez_compressed(
        output / "training_groups.npz", row_order=order,
        group_sizes=groups, depth_labels=train_labels,
    )
    metadata = {
        "config": vars(args) | {"puzzle_info": str(args.puzzle_info.resolve())},
        "group_width": GROUP_WIDTH,
        "rw_nbt_depth": RW_NBT_DEPTH,
        "group_seed": args.seed + 1,
        "test_seed": args.seed + 10_000,
        "train_rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "queries": len(groups),
        "query_size_min": min(groups),
        "query_size_max": max(groups),
        "depth_span_max": int(spans.max()),
        "queries_with_label_variation": int(np.count_nonzero(spans)),
        "generation_seconds": generation_seconds,
        "training_states_sha256": hashlib.sha256(
            X.cpu().numpy().tobytes()
        ).hexdigest(),
        "versions": {"torch": torch.__version__, "xgboost": xgb.__version__},
        "gpu": torch.cuda.get_device_name(device),
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (
                Path(__file__), REPO_ROOT / "src/models/xgboost_model.py",
                REPO_ROOT / "src/data_gen/random_walks.py",
            )
        },
        "puzzle_sha256": hashlib.sha256(
            args.puzzle_info.read_bytes()
        ).hexdigest(),
        "models": {},
    }
    print(
        f"shared training rows={len(y):,}, queries={len(groups):,}, "
        f"query sizes={min(groups)}..{max(groups)}, "
        f"maximum depth span={int(spans.max())}",
        flush=True,
    )

    # Generate held-out rows once.
    # An independent seed does not imply disjoint states in this finite graph.
    set_seed(args.seed + 10_000)
    X_test, y_test = random_walks_beam_nbt(
        moves, initial_state=goal, n_steps=args.max_depth + 1,
        n_walks=args.test_walks, device=device, nbt_depth=RW_NBT_DEPTH,
    )
    labels = y_test.cpu().numpy().astype(np.int64)
    predictions = pd.DataFrame({"depth": labels})
    summary, detailed = [], []
    for name, objective in (
        ("regression", "reg:squarederror"), ("ranker", "rank:pairwise"),
    ):
        model = XGBoostModel(n_estimators=args.rounds, objective=objective)
        model.params["seed"] = args.seed
        torch.cuda.synchronize()
        started = perf_counter()
        model.train(X, y, group=groups if name == "ranker" else None)
        torch.cuda.synchronize()
        train_seconds = perf_counter() - started
        scores = model.predict(X_test).detach().cpu().numpy()
        model.save(str(output / f"{name}.ubj"))
        (output / f"{name}_config.json").write_text(
            model.booster.save_config(), encoding="utf-8",
        )
        predictions[name] = scores
        pairs = pairwise_metrics(labels, scores)
        pairs.insert(0, "model", name)
        detailed.append(pairs)
        local = pairs["delta_depth"] < GROUP_WIDTH
        summary.append({
            "model": name,
            "train_seconds": train_seconds,
            "spearman": pd.Series(labels).corr(
                pd.Series(scores), method="spearman",
            ),
            "pair_accuracy_all": pooled_accuracy(pairs),
            "pair_accuracy_delta1": pooled_accuracy(
                pairs[pairs["delta_depth"] == 1],
            ),
            "pair_accuracy_delta1_7": pooled_accuracy(pairs[local]),
            "pair_accuracy_delta1_7_depth10plus": pooled_accuracy(
                pairs[local & (pairs["lower_depth"] >= 10)],
            ),
            "pair_tie_fraction": (
                pairs["tied_pairs"].sum() / pairs["pairs"].sum()
            ),
            "rmse": (
                float(np.sqrt(np.mean((scores - labels) ** 2)))
                if name == "regression" else float("nan")
            ),
        })
        metadata["models"][name] = {
            "params": model.params,
            "num_boost_round": model.num_boost_round,
        }
        predictions.to_csv(output / "test_predictions.csv", index=False)
        pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
        pd.concat(detailed, ignore_index=True).to_csv(
            output / "pairwise_by_depth.csv", index=False,
        )
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(pd.DataFrame(summary).to_string(index=False), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"complete: {output}", flush=True)


if __name__ == "__main__":
    main()
