# mypy: ignore-errors

import optuna
import optunahub


OptQuestSampler = optunahub.load_module(package="samplers/optquest")


def sim(params):
    x, y = params["x"], params["y"]
    return {"out00": x**2 + y**2, "out01": (x - 5) ** 2 + (y - 5) ** 2}


model = [
    {"cmd": "CONTINUOUS", "name": "x", "min": -5.0, "max": 10.0},
    {"cmd": "CONTINUOUS", "name": "y", "min": -5.0, "max": 10.0},
    {"cmd": "OUTPUT", "name": "out00"},
    {"cmd": "OUTPUT", "name": "out01"},
    {"cmd": "OBJECTIVE", "name": "obj00", "direction": "min", "expression": "out00"},
    {"cmd": "OBJECTIVE", "name": "obj01", "direction": "min", "expression": "out01"},
]

sampler = OptQuestSampler.OptQuestSampler(statements=model, license="")

study = optuna.create_study(sampler=sampler, directions=sampler.directions)

study.optimize(sampler.wrap(sim), n_trials=50)
