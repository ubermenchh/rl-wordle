"""Tests for the data side of the Wordle taskset (Mark I, step 2).

The taskset gives mid-game states. Each task holds the secret and the prior guesses. The next
step (`Env.run`) replays the priors into a `Game`, so the engine recomputes the feedback and it
can never disagree with the secret.

Run from the repo root: `uv run python -m tests.test_taskset`.
"""

import itertools

from wordle_rl.render import FEWSHOT_SECRETS, SYSTEM, render
from wordle_rl.taskset import WordleConfig, WordleTaskset
from wordle_rl.wordle import ANSWERS, MAX_TURNS, WORD_LEN, Game, split_answers


def _tasks(n: int, **config):
    """The first n tasks of a taskset. load() can be an endless generator, so we cut it."""
    taskset = WordleTaskset(WordleConfig(**config))
    return list(itertools.islice(taskset.load(), n))


def _split():
    # The same split that the gate and the SFT traces use (decisions of 2026-10-02).
    reserved = [w for w in FEWSHOT_SECRETS if w in ANSWERS]
    return split_answers(reserved=reserved)


def _replay(secret: str, priors: list[str]) -> Game:
    game = Game(secret)
    for word in priors:
        game.guess(word)
    return game


def _test_a_task_is_a_rendered_state():
    """The prompt must be the rendered state, and the system prompt must be the engine's.
    If they differ, the model trains on a prompt that the eval never shows."""
    for task in _tasks(5):
        data = task.data
        assert len(data.secret) == WORD_LEN and data.secret in ANSWERS
        game = _replay(data.secret, data.priors)
        assert data.prompt == render(game)
        assert data.system_prompt == SYSTEM


def _test_priors_are_playable_and_do_not_end_the_game():
    """Replaying the priors must not raise, and the game must still be open.
    A prior that wins, or a repeat, would give a state that no real game reaches."""
    for task in _tasks(200):
        data = task.data
        game = _replay(data.secret, data.priors)  # Edge: raises on a repeat or a bad word.
        assert data.secret not in data.priors
        assert not game.over and game.turns_used < MAX_TURNS


def _test_priors_cover_zero_to_four():
    """The 0 to 4 prior guesses are the biggest lever (PLAN.md section 1, finding 2).
    A taskset that gives only one length gives only one kind of state."""
    counts = {n: 0 for n in range(5)}
    for task in _tasks(500):
        counts[len(task.data.priors)] += 1
    assert all(n >= 50 for n in counts.values()), counts  # 500 draws: about 100 each.
    assert sum(counts.values()) == 500, counts            # No state has 5 or more priors.


def _test_the_train_split_never_shows_an_eval_word():
    """Held-out words must not appear as a secret or as a prior guess in training.
    A prior is a hint about the answer set, so a leak makes the eval look better than it is."""
    _, eval_words = _split()
    held_out = set(eval_words)
    for task in _tasks(500, split="train"):
        assert task.data.secret not in held_out
        assert not held_out & set(task.data.priors), task.data.priors


def _test_the_eval_split_uses_only_eval_secrets():
    _, eval_words = _split()
    held_out = set(eval_words)
    assert all(task.data.secret in held_out for task in _tasks(200, split="eval"))


def _test_the_same_seed_gives_the_same_tasks():
    """--resume rebuilds the first n tasks from the seed (verifiers docs, tasksets.md).
    Two seeds must differ, or the seed does nothing."""
    def key(tasks):
        return [(t.data.secret, tuple(t.data.priors)) for t in tasks]

    assert key(_tasks(30, seed=0)) == key(_tasks(30, seed=0))
    assert key(_tasks(30, seed=0)) != key(_tasks(30, seed=1))


def run_tests():
    for name, test in sorted(globals().items()):
        if name.startswith("_test_"):
            test()
            print(f"  ok  {name}")
    print(f"\n{sum(n.startswith('_test_') for n in globals())} tests passed.")


if __name__ == "__main__":
    run_tests()
