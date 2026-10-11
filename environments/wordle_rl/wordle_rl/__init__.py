"""Wordle engine and verifiers taskset.

verifiers loads a taskset by the name of its package and takes the exported `Env` subclass as
the env of that taskset (`loaders.environment_class`). Without the export of `WordleEnv`, a run
falls back to a single-turn env, and the game would not be played.
"""

from wordle_rl.taskset import (
    WordleConfig,
    WordleData,
    WordleEnv,
    WordleEnvConfig,
    WordleTask,
    WordleTaskset,
)

__all__ = [
    "WordleConfig",
    "WordleData",
    "WordleEnv",
    "WordleEnvConfig",
    "WordleTask",
    "WordleTaskset",
]
