from typing import Dict, Tuple

import pandas as pd


def create_pddl_prompt(row: pd.Series, values: Dict) -> str:
    """Create a prompt for PDDL text type."""
    pddl_domain = values["PDDL_domain"][row.PDDL_domain_idx]
    pddl_problem = values["PDDL_problem"][row.PDDL_problem_idx]
    state_predicates = values["state"][row.state_idx]

    return (
        "### DOMAIN DEFINITION ###\n"
        f"{pddl_domain}\n\n"
        "### PROBLEM DEFINITION ###\n"
        f"{pddl_problem}\n\n"
        "### CURRENT STATE ###\n"
        "The world is currently described by the following predicates:\n"
        f"{state_predicates}"
    )


def create_original_prompt(row: pd.Series, values: Dict) -> str:
    """Create a prompt for original text type."""
    problem_description = values["problem"][row.problem_idx]
    state_description = values["state_description"][row.state_description_idx]
    goal_description = values["goal_description"][row.goal_description_idx]

    return (
        "### PROBLEM DESCRIPTION ###\n"
        f"{problem_description}\n\n"
        "### CURRENT STATE ###\n"
        f"{state_description}\n\n"
        "### GOAL DESCRIPTION ###\n"
        f"{goal_description}"
    )


def create_alt_prompt(row: pd.Series, values: Dict, generator) -> Tuple[str, bool]:
    """Create a prompt for alternative text type, potentially generating new content."""
    problem_description = values["problem"][row.problem_idx]
    state_description = values["state_description"][row.state_description_idx]
    goal_description = values["goal_description"][row.goal_description_idx]

    # Check if we already have an alternative state description
    if "alt_state_description" in values and len(values["alt_state_description"]) > row.state_description_idx:
        new_state_description = values["alt_state_description"][row.state_description_idx]
        if new_state_description:  # Make sure it's not None or empty
            prompt = (
                "### PROBLEM DESCRIPTION ###\n"
                f"{problem_description}\n\n"
                "### REPHRASED CURRENT STATE ###\n"
                f"{new_state_description}\n\n"
                "### GOAL DESCRIPTION ###\n"
                f"{goal_description}"
            )
            return prompt, False  # No modification needed

    # Generate new alternative description
    if generator is not None:
        prompt = (f"The following problem description provides context. "
                  f"Do NOT change it. Your task is to rephrase the state description "
                  f"in a way that keeps the same meaning but uses different wording and structure. "
                  f"You may occasionally introduce natural-sounding redundant phrases, parentheticals, "
                  f"or descriptive padding — but only if it sounds natural. Vary sentence structure, vocabulary, and tone. "
                  f"Do not introduce new facts or remove important details. "
                  f"Output only the rephrased state description, and nothing else.\n\n"
                  f"Problem Description:\n{problem_description}\n\n"
                  f"State Description:\n{state_description}\n\n"
                  f"Goal Description:\n{goal_description}\n\n"
                  f"Rephrased State Description:")
        new_state_description = generator([prompt])[0].replace(prompt, "").strip()

        # Ensure the alt_state_description list exists and is long enough
        if "alt_state_description" not in values:
            values["alt_state_description"] = [None] * len(values["state_description"])
        while len(values["alt_state_description"]) <= row.state_description_idx:
            values["alt_state_description"].append(None)

        # Store the new alternative description
        values["alt_state_description"][row.state_description_idx] = new_state_description

        prompt = (
            "### PROBLEM DESCRIPTION ###\n"
            f"{problem_description}\n\n"
            "### REPHRASED CURRENT STATE ###\n"
            f"{new_state_description}\n\n"
            "### GOAL DESCRIPTION ###\n"
            f"{goal_description}"
        )
        return prompt, True  # Values were modified

    # Default fallback if no generator provided
    return create_original_prompt(row, values), False


def create_prompt(row: pd.Series, text_type: str, values: Dict, generator=None) -> Tuple[str, bool]:
    """Factory function to create prompts based on text type."""
    if text_type == "pddl":
        return create_pddl_prompt(row, values), False
    elif text_type == "original":
        return create_original_prompt(row, values), False
    elif text_type == "alt":
        return create_alt_prompt(row, values, generator)
    else:
        raise ValueError(f"Unsupported text type: {text_type}")

