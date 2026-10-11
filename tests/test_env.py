"""Tests for the game loop of the Wordle env (Mark I, step 3).

`play_game(task, interaction)` plays one game. `WordleEnv.run` only opens the interaction and
calls it. The tests use a fake interaction, so they need no model and no server.

The fake follows the real `Interaction` of verifiers (`agent.py`): a bare `turn()` takes the
model's reply to the task prompt, and `turn(message)` sends the next user message. The
`Segment` class is the real one.

Run from the repo root: `uv run python -m tests.test_env`.
"""

import asyncio
from typing import cast

import verifiers.v1 as vf
from verifiers.v1.agent import Segment

from wordle_rl.render import render
from wordle_rl.taskset import WordleConfig, WordleData, WordleTask, play_game
from wordle_rl.wordle import ANSWERS, MAX_TURNS, Game


class _Trace:
    def __init__(self):
        self.info: dict = {}


class _FakeInteraction:
    """Gives the scripted replies one by one. When they are gone, the run has ended."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.sent: list[str | None] = []  # What the env passed to each turn(). None = bare turn.
        self.trace = _Trace()

    async def turn(self, message=None) -> Segment:
        self.sent.append(message)
        if not self.replies:
            return Segment(messages=[], terminated=True)
        return Segment(messages=[], root_reply=self.replies.pop(0))


def _tag(word: str) -> str:
    return f"<guess>{word}</guess>"


def _task(secret: str, priors: list[str]) -> WordleTask:
    game = Game(secret)
    for word in priors:
        game.guess(word)
    data = WordleData(name="test", prompt=render(game), secret=secret, priors=priors)
    return WordleTask(data, WordleConfig().task)


def _play(secret: str, priors: list[str], replies: list[str]):
    interaction = _FakeInteraction(replies)
    # The fake has only the two members that play_game uses (turn and trace), so we cast it.
    outcome = asyncio.run(play_game(_task(secret, priors), cast(vf.Interaction, interaction)))
    return outcome, interaction


def _state_after(secret: str, guesses: list[str]) -> str:
    """The text that the env must send after these guesses: the render of the engine."""
    game = Game(secret)
    for word in guesses:
        game.guess(word)
    return render(game)


def _test_a_win_ends_the_game_at_once():
    """After a win the env must not ask for another guess. A turn after the end is wasted
    generation, and the model may even break a won game."""
    outcome, interaction = _play("CRANE", ["SLATE"], [_tag("CRANE")])
    assert outcome == "won"
    assert interaction.sent == [None]  # Only the opening reply. No turn after the win.
    assert interaction.trace.info["outcome"] == "won"


def _test_the_next_message_is_the_rendered_state():
    """Choice 1a: each user message is the full pre-digested state, as in the first prompt.
    The model then sees one format on every turn."""
    outcome, interaction = _play("CRANE", ["SLATE"], [_tag("CRAZE"), _tag("CRANE")])
    assert outcome == "won"
    assert interaction.sent == [None, _state_after("CRANE", ["SLATE", "CRAZE"])]


def _test_an_illegal_reply_ends_the_episode_at_once():
    """Choice 2a: no retry. The kind must be the same as in reward.py, so that the reward and
    the metrics agree. The next reply is never asked for, even if it would be a win."""
    cases = {
        "I think the answer is crane": "format",  # No tag.
        _tag("CAT"): "bad_length",
        _tag("CR4NE"): "not_alpha",
        _tag("ZZZZZ"): "not_a_word",
        _tag("SLATE"): "repeat",  # SLATE is a prior guess.
    }
    for reply, kind in cases.items():
        outcome, interaction = _play("CRANE", ["SLATE"], [reply, _tag("CRANE")])
        assert outcome == kind, (reply, outcome)
        assert interaction.sent == [None], (reply, interaction.sent)
        assert interaction.trace.info["outcome"] == kind


def _test_the_game_is_lost_after_the_last_turn():
    """The game stops at MAX_TURNS and the env asks for no more. Edge: the 6th wrong guess
    gets no feedback message, because there is no next turn."""
    wrong = [w for w in ANSWERS if w != "CRANE"][:MAX_TURNS]
    outcome, interaction = _play("CRANE", [], [_tag(w) for w in wrong])
    assert outcome == "lost"
    assert len(interaction.sent) == MAX_TURNS  # One bare turn + 5 state messages.
    assert interaction.sent[-1] == _state_after("CRANE", wrong[: MAX_TURNS - 1])


def _test_priors_count_as_turns():
    """A task with 4 priors has 2 turns left. The env must replay the priors into the game."""
    priors = [w for w in ANSWERS if w != "CRANE"][:4]
    wrong = [w for w in ANSWERS if w != "CRANE"][4:6]
    outcome, interaction = _play("CRANE", priors, [_tag(w) for w in wrong])
    assert outcome == "lost"
    assert len(interaction.sent) == 2


def _test_a_run_that_ends_early_is_reported():
    """The run can end (a timeout or a limit) before the game does. This must not look like a
    loss: the reward must be able to tell the two apart."""
    outcome, interaction = _play("CRANE", [], [_tag("CRAZE")])
    assert outcome == "terminated"
    assert interaction.trace.info["outcome"] == "terminated"


def _test_verifiers_pairs_the_taskset_with_this_env():
    """verifiers takes the exported Env of the package as the env of the taskset. If
    __init__.py stops exporting WordleEnv, a run falls back to a single-turn env and the game
    is never played. No other test would notice."""
    from verifiers.v1.utils.loaders import environment_class, taskset_class

    from wordle_rl.taskset import WordleEnv, WordleTaskset

    for taskset_id in ("wordle_rl", "wordle-rl"):
        assert taskset_class(taskset_id) is WordleTaskset
        assert environment_class(taskset_id) is WordleEnv


def _test_the_player_runs_in_a_plain_local_process():
    """The default of verifiers is a bash harness in a remote Prime sandbox. For a chat game
    that gives the model a shell it must not have, and it needs Prime credentials and costs
    money. Our env must set the null harness and the subprocess runtime as its default, so
    that `vf-eval wordle-rl` and prime-rl need no flags. Found with `vf-eval --dry-run`."""
    from wordle_rl.taskset import WordleEnvConfig

    player = WordleEnvConfig().player
    assert player.harness is not None and player.harness.id == "null"
    assert player.runtime.type == "subprocess"


def run_tests():
    for name, test in sorted(globals().items()):
        if name.startswith("_test_"):
            test()
            print(f"  ok  {name}")
    print(f"\n{sum(n.startswith('_test_') for n in globals())} tests passed.")


if __name__ == "__main__":
    run_tests()
