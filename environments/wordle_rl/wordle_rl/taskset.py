import itertools
import random
from collections.abc import Iterator
from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.harnesses.null import NullHarnessConfig

from wordle_rl.render import FEWSHOT_SECRETS, SYSTEM, parse_tag, render
from wordle_rl.reward import reward
from wordle_rl.wordle import ANSWERS, constraints_from, Game, is_consistent, InvalidGuess, sample_state, split_answers

MAX_PRIOR = 4 # 0 to 4 prior guesses is the biggest lever

class WordleData(vf.TaskData):
    secret: str
    priors: list[str]
    shaping: float = 0.0 # weight of the bonus for earlier replies that fit the feedback

class WordleConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"
    seed: int = 0
    shaping: float = 0.0

def episode_reward(data: WordleData, replies: list[str]) -> float:
    """Reward of the whole game: the reward of the LAST reply, in the state before it.

    Gated and not summed. A win pays 1.0 plus a bonus for each turn saved.
    A lost game gets the rung of its last guess. An illegal reply gets its own rung.
    """
    # replay the priors and every reply except the last.
    # They are all legal: the env ends the episode at the first illegal reply,
    # so an illegal reply is always the last one.
    game = Game(data.secret)
    for word in data.priors:
        game.guess(word)
    earlier_fit = 0
    try:
        for reply in replies[:-1]:
            # validate first: an illegal word raises here, as before.
            word = game.validate(parse_tag(reply) or "")
            earlier_fit += is_consistent(word, constraints_from(game))
            game.guess(word)
    except InvalidGuess:
        return reward("", game)[0]

    # reward("", game) is the format failure
    if not replies:
        return reward("", game)[0]

    # Score the last reply, in the state before it.
    score, kind = reward(replies[-1], game)

    # a legal last reply, but the game is still open
    # the run ended early (timeout or cut reply)
    legal = kind in ("solved", "consistent", "valid")
    game_over = kind == "solved" or game.turns_used + 1 >= game.max_turns
    if legal and not game_over:
        return reward("", game)[0]

    # Subtle: the bonus is at most `shaping` (0.2), below the gap between a win (at least 1.08)
    # and the best lost (0.3), so a win always beats a loss. max(..., 1) avoids dividing by 0
    # for a game of one reply
    return score + data.shaping * earlier_fit / max(len(replies) - 1, 1)

class WordleTask(vf.Task[WordleData, vf.State, vf.TaskConfig]):
    # vf.reward marks the scoring function; reads whole trace with every message
    @vf.reward
    async def ladder(self, trace: vf.Trace) -> float:
        replies = [m.content or "" for m in trace.assistant_messages]
        return episode_reward(self.data, replies)

    # a metric reports a number for each episode and does not change the reward.
    # The mean of 'won' over the episodes is the win rate
    @vf.metric
    async def won(self, trace: vf.Trace) -> float:
        return float(trace.info.get("outcome") == "won")

    # Any illegal reply ends the episode
    # A loss and terminated run are not illegal: the game was played.
    @vf.metric
    async def illegal(self, trace: vf.Trace) -> float:
        return float(trace.info.get("outcome") not in ("won", "lost", "terminated"))

    # the missing tag, apart from a wrong word
    # action validity is the main failure of small models
    @vf.metric
    async def format_error(self, trace: vf.Trace) -> float:
        return float(trace.info.get("outcome") == "format")

class WordleTaskset(vf.Taskset[WordleTask, WordleConfig]):
    INFINITE = True

    def load(self) -> Iterator[WordleTask]:
        train_words, eval_words = split_answers(reserved=[w for w in FEWSHOT_SECRETS if w in ANSWERS])
        held_out = set(eval_words)
        secrets = train_words if self.config.split == "train" else eval_words
        rng = random.Random(self.config.seed)

        for i in itertools.count():
            n_prior = rng.randint(0, MAX_PRIOR)
            secret = rng.choice(secrets)
            while True:
                game = sample_state(n_prior, rng, secret=secret)
                landed = len(game.guesses) == n_prior
                leaks = self.config.split == "train" and held_out & set(game.guesses)
                if landed and not leaks:
                    break

            yield WordleTask(
                WordleData(
                    name=f"{self.config.split}#{i}",
                    prompt=render(game),
                    system_prompt=SYSTEM,
                    secret=game.secret,
                    priors=list(game.guesses),
                    shaping=self.config.shaping,
                ),
                self.config.task,
            )

async def play_game(task: WordleTask, interaction: vf.Interaction) -> str:
    """Play one game. Return how it ended: won, lost, terminated or the kind of illegal reply.

    kinds: (format, bad_length, not_alpha, not_a_word, repeat)
    """
    # Replay the prior guesses. The turns they used are gone from the game.
    game = Game(task.data.secret)
    for word in task.data.priors:
        game.guess(word)

    # "terminated" stays if the run ends (a timeout or a limit) before the game does.
    # The reward must tell this apart from a loss
    outcome = "terminated"

    # The task prompt is the first state, so the model speaks first
    segment = await interaction.turn()
    while not segment.terminated:
        word = parse_tag(segment.last_reply)
        if word is None:
            outcome = "format"
            break
        try:
            game.guess(word)
        except InvalidGuess as e:
            # end at once, a retry would let a model that never learns the format
            # waste its tokens, and it would hide the failure from the reward
            outcome = e.kind
            break
        if game.over:
            outcome = "won" if game.won else "lost"
            break

        # send the full state, as in the first prompt
        # the model sees one format
        segment = await interaction.turn(render(game))

    interaction.trace.info["outcome"] = outcome
    return outcome

class WordleEnvConfig(vf.EnvConfig):
    player: vf.AgentConfig = vf.AgentConfig(
        harness=NullHarnessConfig(id="null"),
        runtime=vf.SubprocessConfig(),
    )

class WordleEnv(vf.Env[WordleEnvConfig]):
    async def run(self, task, agents):
        # Leaving the context closes the exchange, and the hooks and the scoring run then
        async with agents.player.interaction(task) as interaction:
            await play_game(task, interaction)
