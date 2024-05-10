from optuna.samplers import RandomSampler


class DemoSampler(RandomSampler):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        print("This is DemoSampler.")
