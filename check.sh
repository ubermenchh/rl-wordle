#!/usr/bin/env sh
# One command for "is the repo green": self-tests first (they gate correctness), then lint and
# types. Check mode only, so it never rewrites files.
#
# Tests stop the script on failure (set -e) because a red test makes lint output irrelevant.
# Lint and types run to completion and fail the script at the end, so you see both reports.
set -e

# In-module self-tests (the existing `_test_*` + `run_tests()` convention).
# They live in the wordle_rl package now, so run them as modules (`-m`), not as scripts.
# constrained.py and evaluate.py are left out: the first needs the HF cache, the second has none.
for m in wordle render reward; do
  echo "== wordle_rl.$m"
  uv run python -m "wordle_rl.$m" test
done

# Separate test files (new tests live here, not inside user-owned modules).
# Run as modules (`-m tests.test_x`) so the repo root is importable without a sys.path hack.
for t in tests/test_*.py; do
  [ -e "$t" ] || continue
  m=$(echo "${t%.py}" | tr / .)
  echo "== $m"
  uv run python -m "$m"
done

set +e
status=0
echo "== ruff"
uv run ruff check . || status=1
echo "== ty"
uv run ty check || status=1
exit $status
