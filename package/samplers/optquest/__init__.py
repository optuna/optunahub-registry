from .OptQuestSampler import OptQuestSampler


try:
    from optquest import OptQuestModel
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please run `pip install optquest` to use `OptQuestSampler`.")


__all__ = ["OptQuestSampler", "OptQuestModel"]
