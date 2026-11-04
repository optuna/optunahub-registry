# mypy: ignore-errors

# Demonstrates a manifold (structured) variable type through the declarative construction
# mode + wrap(): a DAG variable over 5 tasks, each with a processing cost, where the objective
# rewards a densely connected precedence graph while penalizing edges between expensive tasks.
# sim() receives the DAG already decoded as a {(from, to): 0/1} dict. You do not write the
# "<name>@<from>,<to>" names by hand, and none of the ~20 edge flags need their own
# trial.suggest_*() call.

import optuna
import optunahub


OptQuestSampler = optunahub.load_module(package="samplers/optquest")

TASKS = ["design", "build", "test", "review", "ship"]
COST = {"design": 3.0, "build": 5.0, "test": 2.0, "review": 1.0, "ship": 4.0}

model = [
    {"cmd": "DAG", "name": "precedence", "nodes": TASKS},
    {"cmd": "OUTPUT", "name": "penalty"},
    {"cmd": "OBJECTIVE", "name": "obj", "direction": "min", "expression": "penalty"},
]

sampler = OptQuestSampler.OptQuestSampler(statements=model, license="")


def sim(params):
    edges = params["precedence"]  # {(from, to): 0/1} for every ordered pair of different TASKS
    n_edges = sum(edges.values())
    if n_edges == 0:
        return None  # reject: an empty precedence graph isn't a useful schedule
    expensive_edge_cost = sum(
        COST[a] + COST[b] for (a, b), on in edges.items() if on and COST[a] + COST[b] > 6
    )
    # reward more precedence structure, penalize chaining expensive tasks together
    return {"penalty": expensive_edge_cost - n_edges}


study = optuna.create_study(sampler=sampler, directions=sampler.directions)
study.optimize(sampler.wrap(sim), n_trials=50)  # the unlicensed demo engine caps at 50 iterations

print("best penalty:", study.best_value)
print("best precedence graph:")
best_trial = study.best_trial
for key, value in best_trial.params.items():
    if value:
        print(" ", key)
