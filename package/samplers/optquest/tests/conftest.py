"""Skips these tests when the OptQuest engine cannot be loaded.

`pip install optquest` ships engine binaries for Windows and Linux (x86-64) only. On any other
platform the package installs but the engine cannot be loaded, so there is nothing to test.
"""

import warnings


collect_ignore_glob: list[str] = []

try:
    import optquest

    optquest.get_version_info()
except Exception as e:  # ImportError, or OSError when the engine library cannot be loaded
    warnings.warn(f"Skipping the OptQuestSampler tests: the OptQuest engine is unavailable ({e}).")
    collect_ignore_glob.append("test_*.py")
