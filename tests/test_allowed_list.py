"""
Contract for the allowed-guess list: it ships with the repo and does NOT come from the OS.

Why this exists: wordle.py used to build ALLOWED from macOS's /usr/share/dict/words and quietly
fell back to the 900 answers when that file was missing. A Linux Modal image has no such file, so
a remote gate/eval would have run with a 900-word dictionary instead of 8,511, shifting `valid`,
`not_a_word` and every reward without any error. Local and remote numbers are only comparable if
both read the same list.

Run from the repo root: `uv run python -m tests.test_allowed_list`.
"""

from pathlib import Path

import wordle_rl.wordle as wordle

ROOT = Path(__file__).resolve().parent.parent
# The file that wordle.py really reads: it sits next to the module inside the wordle_rl package.
ALLOWED_FILE = Path(wordle.__file__).with_name("allowed.txt")
SYSTEM_DICT_PREFIX = "/usr/share/dict"


def _file_words() -> list[str]:
    return ALLOWED_FILE.read_text().split()


def _test_allowed_file_is_shipped_and_well_formed():
    # Sorted + unique + uppercase 5-letter alphabetic: same invariants as answers.txt.
    assert ALLOWED_FILE.exists(), "allowed.txt must live in the repo so it ships to Modal"
    words = _file_words()
    assert len(words) > 5000, len(words)
    assert all(len(w) == wordle.WORD_LEN and w.isalpha() and w.isupper() for w in words)
    assert words == sorted(set(words)), "allowed.txt must be unique and sorted"


def _test_allowed_is_the_file_plus_answers():
    # Every answer must stay a legal guess even if the file ever lacks one.
    assert wordle.ALLOWED == set(_file_words()) | set(wordle.ANSWERS)


def _test_loading_never_touches_the_os_dictionary():
    # Implementation-agnostic: make any read, exists() check or open() of /usr/share/dict raise,
    # then rebuild the list. Passing means the result cannot depend on the OS dictionary.
    real_read, real_exists, real_open = Path.read_text, Path.exists, Path.open

    def guard(real):
        def wrapped(self, *a, **kw):
            if str(self).startswith(SYSTEM_DICT_PREFIX):
                raise AssertionError(f"touched the OS dictionary: {self}")
            return real(self, *a, **kw)

        return wrapped

    Path.read_text, Path.exists, Path.open = guard(real_read), guard(real_exists), guard(real_open)
    try:
        rebuilt = wordle.load_allowed()
    finally:
        Path.read_text, Path.exists, Path.open = real_read, real_exists, real_open
    assert rebuilt == wordle.ALLOWED
    assert len(rebuilt) > 5000, len(rebuilt)


def run_tests():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_tests()
