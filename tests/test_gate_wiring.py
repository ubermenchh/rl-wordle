"""
Wiring test for `gate.gate()`: the new signal flags must be counted on EVERY decoding path.

group_signal is unit-tested on its own, but gate() needs a model, so a slip in how it is called
(counted on one branch only, or never returned) is invisible to the other tests, and shows up
as a silent "0% signal" in a real sweep. Here the model is replaced by fake samplers whose
outputs are fixed, so the expected flags are known in advance and no weights are loaded.

Run from the repo root: `uv run python -m tests.test_gate_wiring`.
"""

import gate as G

N_STATES, K = 3, 4


def _run(constrained: bool, fewshot: bool = False) -> dict:
    # tokenizer/model are passed in so gate() never calls load(); the fakes below never touch them.
    return G.gate(
        "fake-model", max_prior=2, n_states=N_STATES, k=K, temp=1.0, max_new=8, seed=0,
        tokenizer=object(), model=object(), fewshot=fewshot, constrained=constrained,
    )


def _patch(monkey: dict):
    saved = {name: getattr(G, name) for name in monkey}
    for name, fn in monkey.items():
        setattr(G, name, fn)
    return saved


def _restore(saved: dict):
    for name, fn in saved.items():
        setattr(G, name, fn)


def _fake_sample_k(tokenizer, model, prompt, k, temp, max_new):
    # "" has no tag (reward -0.1) and ZZZZZ is not a word (reward 0.0): always different rewards,
    # whatever state the gate sampled, so nondegen_reward must be True for every state.
    return ["<guess>ZZZZZ</guess>", "", "<guess>ZZZZZ</guess>", ""]


def _fake_sample_words(tokenizer, model, prompt, pool, k, temp):
    # One non-word and one real, unguessed answer: rewards 0.0 vs >= 0.1, so always mixed.
    return ["ZZZZZ", pool[0], "ZZZZZ", pool[0]]


def _test_plain_path_counts_the_new_flags():
    saved = _patch({"sample_k": _fake_sample_k, "build_prompt": lambda *a, **kw: ""})
    try:
        r = _run(constrained=False)
    finally:
        _restore(saved)
    assert r["nd_reward"] == 1.0, r
    assert 0.0 <= r["nd_consist"] <= 1.0, r


def _test_fewshot_path_counts_the_new_flags():
    saved = _patch({"sample_k": _fake_sample_k, "build_prompt": lambda *a, **kw: ""})
    try:
        r = _run(constrained=False, fewshot=True)
    finally:
        _restore(saved)
    assert r["nd_reward"] == 1.0, r


def _test_constrained_path_counts_the_new_flags():
    saved = _patch({"sample_words": _fake_sample_words, "build_prompt": lambda *a, **kw: ""})
    try:
        r = _run(constrained=True)
    finally:
        _restore(saved)
    assert r["nd_reward"] == 1.0, r
    assert 0.0 <= r["nd_consist"] <= 1.0, r


def run_tests():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_tests()
