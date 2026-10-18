# Is the reasoning line right, and does the word obey it? (2026-10-18)

Source: the traces of the 400-game held-out evals (`scripts/check_reasoning.py`; the Must-contain split was an inline script, not saved).
The true line is the one that the SFT teacher writes (`sft/teacher.py::explain`). The parser was tested on the teacher's own 1,960 training traces with feedback: the line is right in 1,960 and the word obeys it in 1,960.
A reply counts if it is a legal word and the game has feedback to use (turn 1 with no priors is left out).

| | SFT | RL run 1 | RL run 2 |
|---|---|---|---|
| Legal replies with feedback to use | 795 | 1,140 | 1,121 |
| Word obeys the model's own line | 25% | 28% | 27% |
| Word consistent with all feedback, line exactly right | 18% (126 of 690) | 23% (81 of 346) | 24% (96 of 407) |
| Word consistent, line wrong | 5% | 10% | 7% |
| Replies where "Must contain" is needed | 549 | 768 | 750 |
| ... the model writes that section | 100% | 15% | 12% |
| ... "Fixed" exactly right where it is needed | 88% | 72% | 91% |
| Word consistent, "Must contain" written | 12% (65 of 547) | 29% (33 of 114) | 32% (28 of 88) |
| Word consistent, "Must contain" left out | 0 of 2 | 7% (45 of 654) | 7% (45 of 662) |

How the inconsistent replies break the constraints (can overlap): misses a letter that must be in 65% to 75%, breaks a green slot 61% to 66%, uses a grey letter 53% to 60%, uses a letter in a slot it was ruled out of 43% to 49%.

What it means:

- The word is the main failure. Even when the line is exactly right, the word is consistent in only 18% to 24% of the replies. Only about one word in four obeys the facts that the model wrote one line before.
- RL changed the line, and not for the better. After SFT the model wrote the "Must contain" section whenever it was needed. After RL it writes it in 12% to 15% of the cases. A possible cause (not tested): RL shortened the replies, and that section is the part that needs the most work.
- When the model does write "Must contain", the word is consistent more often (29% to 32%, against 7% when it is left out). The numbers are small (114 and 88 replies), and the sections probably appear in the easy states, so this is not proof that writing the section helps.
- The "facts line exactly right" share is not comparable between SFT and RL, because the RL models play more late turns, with more facts to state. This is why the table uses the Must-contain split.
- Not tested: a run that rewards the section, or an SFT that is longer or has more examples of late turns.
