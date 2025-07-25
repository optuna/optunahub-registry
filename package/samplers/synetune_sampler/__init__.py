try:
    from .syne_tune_sampler import SyneTuneSampler
except ImportError:
    raise ModuleNotFoundError(
        "\nSome dependencies of syne-tune could not be found."
        "\nPlease install them via `pip install syne-tune[extra]`."
    )


__all__ = ["SyneTuneSampler"]
