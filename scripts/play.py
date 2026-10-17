"""Watch the model play one Wordle game (Mark IV, step 2c).

The game follows the rules of the eval: 6 turns, and an illegal reply (no tag, not a word, a
repeat) ends the game at once. The model sees the same prompts as in training: the system prompt,
then `render(game)` after each guess, in one chat.

Run: uv run python -m scripts.play --seed 3
     uv run python -m scripts.play --secret CRANE
"""

import argparse
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wordle_rl.render import SYSTEM, parse_tag, render
from wordle_rl.wordle import ANSWERS, Game, InvalidGuess

REPO = "ubermenchh/qwen3-1.7b-wordle-rl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--secret", help="a 5-letter word. Default: a random answer.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    secret = (args.secret or random.Random(args.seed).choice(ANSWERS)).upper()
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForCausalLM.from_pretrained(args.repo, dtype=torch.bfloat16).to(device)
    torch.manual_seed(args.seed)

    game = Game(secret)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": render(game)}]
    while not game.over:
        # Thinking off and temperature 1.0, as in every eval of this project.
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok(text, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=1.0)
        reply = tok.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        print(f"\nmodel: {reply.strip()}")

        word = parse_tag(reply)
        if word is None:
            print("-> no <guess> tag. The game ends.")
            break
        try:
            game.guess(word)
        except InvalidGuess as e:
            print(f"-> illegal: {e}. The game ends.")
            break
        print(f"   {word}  {game.feedback[-1]}")
        messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": render(game)}]

    print(f"\nThe secret was {secret}. " + ("The model won in " + f"{game.turns_used} turns." if game.won else "The model did not win."))


if __name__ == "__main__":
    main()
