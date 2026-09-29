import argparse
import sys
import os
import ast
import numpy as np

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedplan.config import Config

def parse_plan(plan_raw):
    try:
        parsed = ast.literal_eval(plan_raw)
    except Exception:
        return []
    if isinstance(parsed, list):
        if all(isinstance(x, str) for x in parsed):
            return parsed
        if all(isinstance(x, list) for x in parsed) and parsed and all(isinstance(y, str) for y in parsed[0]):
            return parsed[0]
    return []

def get_action_types_from_values(values):
    unique_types = set()
    for plan_str in values['plan']:
        actions = parse_plan(plan_str)
        for action in actions:
            clean_action = action.strip("() \t\n")
            if not clean_action:
                continue
            action_type = clean_action.split()[0]
            unique_types.add(action_type)
    return len(unique_types)

def validate_domain_lightweight(domain):
    # 1. Load base data (DataFrames only, no embeddings)
    data = Config.read_factorized(domain)
    if data is None:
        return None
    df, values = data

    # Map goal distances
    goal_dists = values['goal_distance']
    df['goal_val'] = df['goal_distance_idx'].map(lambda x: int(goal_dists[x]))

    # 2. Calculate Statistics directly
    # Transitions are steps where goal_distance > 0 (assuming valid plans)
    transitions_df = df[df['goal_val'] > 0]

    n_problems = df['problem_idx'].nunique()
    n_states = df['state_description_idx'].nunique()
    n_transitions = len(transitions_df)

    n_plans = df['plan_id_idx'].nunique()
    avg_plan = n_transitions / n_plans if n_plans > 0 else 0

    n_actions = get_action_types_from_values(values)

    return {
        "Domain": domain.capitalize(),
        "Problems": n_problems,
        "States": n_states,
        "Transitions": n_transitions,
        "Avg. Plan": avg_plan,
        "Actions": n_actions
    }

def main():
    parser = argparse.ArgumentParser(description="Validate dataset statistics (Lightweight).")
    args = parser.parse_args()

    domains = ["blocksworld", "ferry", "logistics", "depot", "rovers", "satellite", "floortile", "goldminer", "grid"]

    print(f"\n{'Domain':<12} | {'Prob':<6} | {'States':<8} | {'Trans':<8} | {'AvgPlan':<8} | {'Acts':<5}")
    print("-" * 65)


    totals = {
        "Problems": 0, "States": 0, "Transitions": 0, "Avg. Plan": []
    }

    for domain in domains:
        stats = validate_domain_lightweight(domain)
        if stats:
            print(f"{stats['Domain']:<12} | {stats['Problems']:<6} | {stats['States']:<8,} | {stats['Transitions']:<8,} | {stats['Avg. Plan']:<8.1f} | {stats['Actions']:<5}")

            totals["Problems"] += stats["Problems"]
            totals["States"] += stats["States"]
            totals["Transitions"] += stats["Transitions"]
            totals["Avg. Plan"].append(stats["Avg. Plan"])
        else:
            print(f"{domain.capitalize():<12} | {'ERR':<6} | {'ERR':<8} | {'ERR':<8} | {'ERR':<8} | {'ERR':<5}")

    print("-" * 65)
    avg_plan_total = np.mean(totals["Avg. Plan"]) if totals["Avg. Plan"] else 0
    print(f"{'Total':<12} | {totals['Problems']:<6} | {totals['States']:<8,} | {totals['Transitions']:<8,} | {avg_plan_total:<8.1f} | {'--':<5}")
    print("\n")

if __name__ == "__main__":
    main()
