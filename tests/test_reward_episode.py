"""Tests for the episode reward (Mark I, step 4).

`episode_reward(data, replies)` gives the reward of a whole game: the reward of the LAST reply,
in the state before it (decision of 2026-10-11, choice 4a). It is gated and not summed. The
function is pure, so these tests need no model and no trace object.

Secret CRANE. "Grey words" use no letter of CRANE, so every feedback is all grey and many
words stay consistent. This lets a lost game end with a consistent or an inconsistent guess.

Run from the repo root: `uv run python -m tests.test_reward_episode`.
"""

from wordle_rl.render import render
from wordle_rl.taskset import WordleData, episode_reward
from wordle_rl.wordle import ANSWERS, Game, constraints_from, is_consistent

SECRET = "CRANE"
GREY = [w for w in ANSWERS if not set(w) & set(SECRET)]  # BILLY, BLOOD, BLOOM, BOBBY, BOOST, ...


def _tag(word: str) -> str:
    return f"<guess>{word}</guess>"


def _data(priors: list[str], shaping: float = 0.0) -> WordleData:
    game = Game(SECRET)
    for word in priors:
        game.guess(word)
    # Pass `shaping` only when it is set, so the older tests do not need the field.
    extra = {"shaping": shaping} if shaping else {}
    return WordleData(name="test", prompt=render(game), secret=SECRET, priors=priors, **extra)


def _reward(priors: list[str], *replies: str, shaping: float = 0.0) -> float:
    return episode_reward(_data(priors, shaping), list(replies))


def _is(value: float, expected: float) -> bool:
    return abs(value - expected) < 1e-9


def _last_turn_state() -> tuple[list[str], str, str]:
    """5 grey priors, so the next guess is the last turn. Gives a consistent and an
    inconsistent legal word for it, found with the engine and not written by hand."""
    priors = GREY[:5]
    game = Game(SECRET)
    for word in priors:
        game.guess(word)
    constraints = constraints_from(game)
    consistent = next(
        w for w in ANSWERS if w != SECRET and w not in priors and is_consistent(w, constraints)
    )
    inconsistent = next(w for w in GREY if w not in priors and not is_consistent(w, constraints))
    return priors, consistent, inconsistent


def _test_a_win_pays_more_the_sooner_it_comes():
    """Solving in 2 must beat solving in 5 (TASKS task 5). Priors count as turns used."""
    win_now = _reward([], _tag(SECRET))                        # turn 1: 1.0 + 0.5 / 1
    win_turn_3 = _reward(GREY[:2], _tag(SECRET))               # 2 priors, then the win
    win_turn_6 = _reward(GREY[:5], _tag(SECRET))
    assert _is(win_now, 1.5) and _is(win_turn_3, 1.0 + 0.5 / 3)
    assert win_now > win_turn_3 > win_turn_6 > 1.0


def _test_earlier_replies_count_as_turns():
    """The env asks for one reply for each turn. The reward must replay the earlier replies,
    or a win in turn 2 would score like a win in turn 1."""
    assert _is(_reward([], _tag(GREY[0]), _tag(SECRET)), 1.25)  # The same as 1 prior, then the win.


def _test_a_lost_game_gets_the_rung_of_its_last_guess():
    """Gated, not summed: the last reply decides. The six guesses of a lost game do not add up."""
    priors, consistent, inconsistent = _last_turn_state()
    assert _is(_reward(priors, _tag(consistent)), 0.3)
    assert _is(_reward(priors, _tag(inconsistent)), 0.1)
    # Edge: the earlier replies do not change the score. Here they are replies and not priors.
    assert _is(_reward([], *[_tag(w) for w in priors], _tag(consistent)), 0.3)


def _test_an_illegal_reply_scores_its_rung_at_any_turn():
    """The env ends the episode at the first illegal reply, so the last reply is that one."""
    assert _is(_reward([], "I think the answer is crane"), -0.1)      # format
    assert _is(_reward([], _tag("ZZZZZ")), 0.0)                       # not_a_word
    assert _is(_reward([], _tag("CAT")), 0.0)                         # bad_length
    assert _is(_reward(GREY[:2], _tag(GREY[0])), 0.0)                 # repeat of a prior
    assert _is(_reward([], _tag(GREY[0]), "no tag"), -0.1)            # format, in turn 2


def _test_a_run_that_ends_early_scores_like_a_format_failure():
    """Choice 4a, default: terminated = format. The last reply is legal but the game is still
    open, and a legal rung (0.3 or 0.1) would pay the model for being cut off."""
    assert _is(_reward([], _tag("CRAZE")), -0.1)
    assert _is(_reward([]), -0.1)  # No reply at all.


def _test_format_failure_is_the_worst_outcome():
    """charbull's model learned to emit nothing because doing nothing was cheaper than a wrong
    guess (PLAN.md section 7). Nothing else may score as low as a format failure."""
    priors, consistent, inconsistent = _last_turn_state()
    others = [
        _reward([], _tag("ZZZZZ")),
        _reward(priors, _tag(inconsistent)),
        _reward(priors, _tag(consistent)),
        _reward([], _tag(SECRET)),
    ]
    assert min(others) > _reward([], "no tag at all")


def _play(priors: list[str], kinds: str) -> list[str]:
    """Replies for `kinds`: "c" = a consistent word, "i" = a legal word that breaks the feedback.
    The engine picks the words, and each one is judged in the state before it."""
    game = Game(SECRET)
    for word in priors:
        game.guess(word)
    replies = []
    for kind in kinds:
        # All answers, not only grey ones: grey words use up the 21 free letters after 4 guesses.
        # The secret is left out, because guessing it ends the game.
        word = next(
            w
            for w in ANSWERS
            if w != SECRET and w not in game.guesses and is_consistent(w, constraints_from(game)) == (kind == "c")
        )
        replies.append(_tag(word))
        game.guess(word)
    return replies


# Shaping (Mark III, run 3): the reward of the last reply stays, and each EARLIER reply that fits
# the feedback adds shaping / (number of earlier replies). Measured on 2026-10-18: only 18% to 23%
# of the legal replies fit all feedback, and the old reward never looked at replies before the last.
SHAPING = 0.2


def _test_shaping_is_off_by_default():
    priors = GREY[:3]
    replies = _play(priors, "cic")  # 3 priors + 3 replies = 6 turns: the game is over
    assert _is(_reward(priors, *replies), 0.3)


def _test_each_earlier_consistent_reply_adds_its_share():
    priors = GREY[:3]
    replies = _play(priors, "cic")  # earlier replies: c, i -> 1 of 2 fits; the last one fits
    assert _is(_reward(priors, *replies, shaping=SHAPING), 0.3 + SHAPING * 1 / 2)


def _test_a_single_reply_gets_no_shaping():
    priors = GREY[:5]
    replies = _play(priors, "c")  # the last turn: no earlier reply
    assert _is(_reward(priors, *replies, shaping=SHAPING), 0.3)


def _test_a_win_still_beats_every_game_without_one():
    priors = GREY[:4]
    best_loss = _reward(priors, *_play(priors, "cc"), shaping=SHAPING)  # 0.3 + the full share
    win = _reward(priors, *_play(priors, "c"), _tag(SECRET), shaping=SHAPING)
    assert win > best_loss + 0.5, (win, best_loss)


def _test_an_illegal_last_reply_keeps_the_share_of_the_earlier_ones():
    priors = GREY[:2]
    replies = _play(priors, "cc") + [_tag("ZZZZZ")]  # not a word: rung 0.0, both earlier fit
    assert _is(_reward(priors, *replies, shaping=SHAPING), 0.0 + SHAPING)


def _test_a_run_that_ends_early_gets_no_shaping():
    priors = GREY[:2]
    replies = _play(priors, "cc")  # legal, and the game is still open: scored as a format failure
    assert _is(_reward(priors, *replies, shaping=SHAPING), -0.1)


def run_tests():
    for name, test in sorted(globals().items()):
        if name.startswith("_test_"):
            test()
            print(f"  ok  {name}")
    print(f"\n{sum(n.startswith('_test_') for n in globals())} tests passed.")


if __name__ == "__main__":
    run_tests()
