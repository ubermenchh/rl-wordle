# Gate: sft-solver-v1 (A10, k=16, 10 states per stratum, T=1.0, plain prompt)

Code: commit `10b89fb` (with uncommitted changes)

| prior | tag | len5 | valid | repeat | consistent | solve | nondegen | nd_reward | nd_consist |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 100% | 100% | 100% | 0% | 100% | 0% | 0% | 0% | 0% |
| 1 | 100% | 88% | 79% | 0% | 22% | 0% | 0% | 100% | 80% |

Examples: prior=1, type=wrong_length
- {'history': [('LAYER', '.....')], 'output': 'Absent: A, E, L, R, Y THROW fits all of this.\n<guess>THROWING</guess>', 'parsed': 'THROWING'}
- {'history': [('LAYER', '.....')], 'output': 'Absent: A, E, L, R, Y BENJOINT fits all of this.\n<guess>BENJOINT</guess>', 'parsed': 'BENJOINT'}
- {'history': [('LAYER', '.....')], 'output': 'Absent: A, E, L, R, Y SCISPIN fits all of this.\n<guess>SCISPIN</guess>', 'parsed': 'SCISPIN'}

Examples: prior=1, type=not_allowed
- {'history': [('XEBEC', '..-=.')], 'output': 'Fixed: slot 4 is E Must contain: B Absent: C, X SHEEL fits all of this.\n<guess>SHEEL</guess>', 'parsed': 'SHEEL'}
- {'history': [('XEBEC', '..-=.')], 'output': 'Fixed: slot 4 is E Must contain: B Absent: C, X FORBE fits all of this.\n<guess>FORBE</guess>', 'parsed': 'FORBE'}
- {'history': [('XEBEC', '..-=.')], 'output': 'Fixed: slot 4 is E Must contain: B Absent: C, X TIBER fits all of this.\n<guess>TIBER</guess>', 'parsed': 'TIBER'}
| 2 | 100% | 82% | 65% | 0% | 2% | 0% | 0% | 100% | 30% |

Examples: prior=2, type=wrong_length
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': 'Absent: A, E, I, L, N, R, X, Y LATH muts بيﾞ fits all of this.\n<guess>LATH</guess>', 'parsed': 'LATH'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': 'Absent: A, E, I, L, N, R, X, Y COVER fits all of this.\n<guess>COVERAL</guess>', 'parsed': 'COVERAL'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': 'Absent: A, E, I, L, N, R, X, Y JUMOP has not been unlocked yet, so open with UMBANISH fits all of this.\n<guess>UMbanish</guess>', 'parsed': 'UMBANISH'}

Examples: prior=2, type=not_allowed
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': 'Absent: A, E, I, L, N, R, X, Y BUCK fits all of this.\n<guess>BUCKC</guess>', 'parsed': 'BUCKC'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': 'Absent: A, E, I, L, N, R, X, Y CHOFT fits all of this.\n<guess>CHOFT</guess>', 'parsed': 'CHOFT'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': 'Absent: A, E, I, L, N, R, X, Y WOMBO fits all of this.\n<guess>WOMBO</guess>', 'parsed': 'WOMBO'}
| 3 | 100% | 76% | 65% | 2% | 4% | 2% | 20% | 100% | 30% |

Examples: prior=3, type=wrong_length
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': 'Must contain: H, O, T, U Absent: A, E, I, L, M, N, R, X, Y FORTH fits all of this.\n<guess>FOORTH</guess>', 'parsed': 'FOORTH'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': 'Must contain: H, O, T, U Absent: A, E, I, L, M, N, R, X, Y BOOT fits all of this.\n<guess>BOOT</guess>', 'parsed': 'BOOT'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': 'Must contain: H, O, T, U Absent: A, E, I, L, M, N, R, X, Y TOOL fits all of this.\n<guess>TOOL</guess>', 'parsed': 'TOOL'}

Examples: prior=3, type=not_allowed
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': 'Must contain: H, O, T, U Absent: A, E, I, L, M, N, R, X, Y THINK fits all of this.\n<guess>THUNK</guess>', 'parsed': 'THUNK'}
- {'history': [('DAIRY', '...-.'), ('MERGE', '.---.'), ('CIVIC', '.....')], 'output': 'Must contain: E, G, R Absent: A, C, D, I, M, V, Y GREER fits all of this.\n<guess>GREER</guess>', 'parsed': 'GREER'}
- {'history': [('PASTE', '.-.-.'), ('VITAL', '..--.'), ('CRAFT', '-==.-')], 'output': 'Fixed: slot 2 is R, slot 3 is A Must contain: C, T Absent: E, F, I, L, P, S, V CARCO fits all of this.\n<guess>CARCO</guess>', 'parsed': 'CARCO'}
| 4 | 100% | 86% | 69% | 0% | 4% | 4% | 10% | 90% | 10% |

Examples: prior=4, type=wrong_length
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': 'Must contain: H, O, T, U Absent: A, E, I, L, M, N, R, X, Y HOFT fits all of this.\n<guess>HOFT</guess>', 'parsed': 'HOFT'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': 'Must contain: H, O, T, U Absent: A, E, I, L, M, N, R, X, Y FORTH fits all of this.\n<guess>FOORTH</guess>', 'parsed': 'FOORTH'}
- {'history': [('DAIRY', '.....'), ('GUESS', '.-...'), ('CIVIC', '.....'), ('THUMB', '-.=-.')], 'output': 'Fixed: slot 3 is U Must contain: M, T Absent: A, B, C, D, E, G, H, I, R, S, V, Y WHILE fits all of this.\n<guess>WHILELY</guess>', 'parsed': 'WHILELY'}

Examples: prior=4, type=not_allowed
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': 'Must contain: H, O, T, U Absent: A, E, I, L, M, N, R, X, Y BOUTH fits all of this.\n<guess>BOUTH</guess>', 'parsed': 'BOUTH'}
- {'history': [('DAIRY', '.....'), ('GUESS', '.-...'), ('CIVIC', '.....'), ('THUMB', '-.=-.')], 'output': 'Fixed: slot 3 is U Must contain: M, T Absent: A, B, C, D, E, G, H, I, R, S, V, Y MOTUM fits all of this.\n<guess>MOTUM</guess>', 'parsed': 'MOTUM'}
- {'history': [('DAIRY', '.....'), ('GUESS', '.-...'), ('CIVIC', '.....'), ('THUMB', '-.=-.')], 'output': 'Fixed: slot 3 is U Must contain: M, T Absent: A, B, C, D, E, G, H, I, R, S, V, Y FULTY fits all of this.\n<guess>FULTY</guess>', 'parsed': 'FULTY'}

