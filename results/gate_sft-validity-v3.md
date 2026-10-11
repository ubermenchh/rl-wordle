# Gate: sft-validity-v3 (A10, k=16, 100 states per stratum, T=1.0, plain prompt)

Code: commit `10b89fb` (with uncommitted changes)

| prior | tag | len5 | valid | repeat | consistent | solve | nondegen | nd_reward | nd_consist |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 100% | 100% | 100% | 0% | 100% | 0% | 0% | 0% | 0% |
| 1 | 100% | 95% | 91% | 0% | 47% | 0% | 0% | 95% | 85% |

Examples: prior=1, type=missing_tag
- {'history': [('LAYER', '.....')], 'output': '<guess_stack</guess>', 'parsed': 'GUESS'}
- {'history': [('JUDGE', '.....')], 'output': '<guess[MICRO</guess>', 'parsed': 'GUESS'}
- {'history': [('SPILL', '-....')], 'output': '<guess份额</guess>', 'parsed': 'GUESS'}

Examples: prior=1, type=wrong_length
- {'history': [('XEBEC', '..-=.')], 'output': '<guess>BOBE</guess>', 'parsed': 'BOBE'}
- {'history': [('TOUCH', '..-..')], 'output': '<guess>AKUP</guess>', 'parsed': 'AKUP'}
- {'history': [('GUESS', '-=-..')], 'output': '<guess>BUGE</guess>', 'parsed': 'BUGE'}

Examples: prior=1, type=not_allowed
- {'history': [('XEBEC', '..-=.')], 'output': '<guess>ALBAM</guess>', 'parsed': 'ALBAM'}
- {'history': [('XEBEC', '..-=.')], 'output': '<guess>BADEN</guess>', 'parsed': 'BADEN'}
- {'history': [('TOUCH', '..-..')], 'output': '<guess>FAÚST</guess>', 'parsed': 'FAÚST'}
| 2 | 100% | 87% | 80% | 0% | 13% | 1% | 5% | 92% | 45% |

Examples: prior=2, type=wrong_length
- {'history': [('TOUCH', '..-..'), ('SUGAR', '.=.=-')], 'output': '<guess>RUNNING</guess>', 'parsed': 'RUNNING'}
- {'history': [('DAIRY', '.....'), ('GUESS', '.-...')], 'output': '<guess>FLU</guess>', 'parsed': 'FLU'}
- {'history': [('DAIRY', '.....'), ('GUESS', '.-...')], 'output': '<guess>FOULOS</guess>', 'parsed': 'FOULOS'}

Examples: prior=2, type=not_allowed
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>BADGU</guess>', 'parsed': 'BADGU'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>BADGU</guess>', 'parsed': 'BADGU'}
- {'history': [('DAIRY', '.....'), ('GUESS', '.-...')], 'output': '<guess>FOULA</guess>', 'parsed': 'FOULA'}
| 3 | 100% | 83% | 73% | 2% | 12% | 4% | 10% | 91% | 32% |

Examples: prior=3, type=missing_tag
- {'history': [('WHELK', '.-...'), ('MIGHT', '.=.--'), ('BIRTH', '.=.==')], 'output': '<guess sitcom</guess>', 'parsed': 'SITCOM'}

Examples: prior=3, type=wrong_length
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>THUD</guess>', 'parsed': 'THUD'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>FOOT</guess>', 'parsed': 'FOOT'}
- {'history': [('PASTE', '.-.-.'), ('VITAL', '..--.'), ('CRAFT', '-==.-')], 'output': '<guess>TR autob</guess>', 'parsed': 'TR AUTOB'}

Examples: prior=3, type=not_allowed
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>WOUTH</guess>', 'parsed': 'WOUTH'}
- {'history': [('PASTE', '.-.-.'), ('VITAL', '..--.'), ('CRAFT', '-==.-')], 'output': '<guess>CRANT</guess>', 'parsed': 'CRANT'}
- {'history': [('PASTE', '.-.-.'), ('VITAL', '..--.'), ('CRAFT', '-==.-')], 'output': '<guess>ARTOR</guess>', 'parsed': 'ARTOR'}
| 4 | 100% | 79% | 68% | 4% | 5% | 2% | 14% | 86% | 24% |

Examples: prior=4, type=missing_tag
- {'history': [('HAKAM', '.=.=.'), ('NAVAL', '-=.==')], 'output': "<guess'>WAN</guess>", 'parsed': 'GUESS'}
- {'history': [('HAKAM', '.=.=.'), ('NAVAL', '-=.==')], 'output': '<guessolist>XNA</guessolist>', 'parsed': 'OLIST'}

Examples: prior=4, type=wrong_length
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>FOOT</guess>', 'parsed': 'FOOT'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>FOOUTH</guess>', 'parsed': 'FOOUTH'}
- {'history': [('DAIRY', '.....'), ('GUESS', '.-...'), ('CIVIC', '.....'), ('THUMB', '-.=-.')], 'output': '<guess>FTUN</guess>', 'parsed': 'FTUN'}

Examples: prior=4, type=not_allowed
- {'history': [('TWICE', '..=.='), ('ASIDE', '.-=.='), ('SPINE', '=.=.='), ('PORTA', '.....')], 'output': '<guess>SHIKE</guess>', 'parsed': 'SHIKE'}
- {'history': [('TWICE', '..=.='), ('ASIDE', '.-=.='), ('SPINE', '=.=.='), ('PORTA', '.....')], 'output': '<guess>SKILE</guess>', 'parsed': 'SKILE'}
- {'history': [('TWICE', '..=.='), ('ASIDE', '.-=.='), ('SPINE', '=.=.='), ('PORTA', '.....')], 'output': '<guess>SHILE</guess>', 'parsed': 'SHILE'}

