"""EmbedPlan on your own data, end to end. Runs on a laptop CPU in about a minute.

Replace `my_trajectories()` with your data: each trajectory is the list of states a run went
through and the actions between them, all as text. Everything else stays the same.

    python examples/your_own_data.py
    python examples/your_own_data.py --encoder BAAI/bge-m3      # pip install -e ".[encoders]"
"""

import argparse

from embedplan import EmbedPlan, split_transitions, transitions_from_trajectories
from embedplan.datasets import load_toy_ferry


def my_trajectories():
    """Your data goes here: a list of (states, actions) and one group id (e.g. problem) each.

    This demo rebuilds trajectories from the toy ferry world shipped with the package.
    """
    frame = load_toy_ferry(n_problems=12, n_plans=3).frame
    trajectories, groups = [], []
    for (problem, _), run in frame.groupby(["problem", "plan"], sort=False):
        states = run["state"].tolist() + [run["next_state"].iloc[-1]]
        trajectories.append((states, run["action"].tolist()))
        groups.append(problem)
    return trajectories, groups


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoder", default="hashing", help='"hashing" (no download) or a model name')
    args = ap.parse_args()

    # 1. Trajectories -> transitions: (state, action) -> next state, with a group per transition.
    trajectories, problems = my_trajectories()
    X, y, groups = transitions_from_trajectories(trajectories, groups=problems)
    print(f"{len(X)} transitions from {len(set(groups))} problems")

    # 2. Test on problems the model never saw (Extrapolation), then on held-out transitions of
    #    problems it did see (Interpolation). Whole problems are held out in the first case.
    for protocol in ("extrapolation", "interpolation"):
        X_tr, X_te, y_tr, y_te, g_tr, g_te = split_transitions(X, y, groups, protocol=protocol)
        use_groups = protocol == "extrapolation"
        model = EmbedPlan(encoder=args.encoder).fit(X_tr, y_tr, groups=g_tr if use_groups else None)
        r = model.evaluate(X_te, y_te, groups=g_te if use_groups else None)
        print(f"{protocol:>13}: Hit@1 {r['hit@1']:.2f}  Hit@5 {r['hit@5']:.2f}  "
              f"(chance@5 {r['chance@5']:.2f}, {r['mean_pool_size']:.0f} candidates, {r['n_queries']} queries)")

    # 3. Predict: the most likely next state of a (state, action) pair, among the known states.
    state, action = X_te[0]
    print("\nstate :", state, "\naction:", action)
    print("pred  :", model.predict([(state, action)])[0])
    print("truth :", y_te[0])

    # 4. Roll out a plan: each prediction is fed back as the next state.
    states, actions = trajectories[0]
    rolled = model.rollout(states[0], actions[:4])
    exact = sum(p == s for p, s in zip(rolled, states[1:5]))
    print(f"\nrollout: {exact} of 4 predicted states match the true trajectory")

    # 5. Save it and load it back.
    model.save("embedplan_model.pt")
    print("saved to embedplan_model.pt; load with EmbedPlan.load('embedplan_model.pt')")


if __name__ == "__main__":
    main()
