# Gate: sft-action-v2 (A10, k=16, 10 states per stratum, T=1.0, plain prompt)

Code: commit `10b89fb` (with uncommitted changes)

| prior | tag | len5 | valid | repeat | consistent | solve | nondegen | nd_reward | nd_consist |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 100% | 100% | 100% | 0% | 100% | 0% | 0% | 0% | 0% |
| 1 | 99% | 86% | 77% | 0% | 27% | 0% | 0% | 100% | 80% |

Examples: prior=1, type=missing_tag
- {'history': [('TOUCH', '..-..')], 'output': '<guess.REDIM</guess>', 'parsed': 'GUESS'}
- {'history': [('DROVE', '.....')], 'output': '<guess FLUID</guess>', 'parsed': 'FLUID'}

Examples: prior=1, type=wrong_length
- {'history': [('LAYER', '.....')], 'output': '<guess>RIPH</guess>', 'parsed': 'RIPH'}
- {'history': [('LAYER', '.....')], 'output': '<guess>高峰论坛</guess>', 'parsed': '高峰论坛'}
- {'history': [('LAYER', '.....')], 'output': '<guess>MOBBING</guess>', 'parsed': 'MOBBING'}

Examples: prior=1, type=not_allowed
- {'history': [('LAYER', '.....')], 'output': '<guess>CHAUD</guess>', 'parsed': 'CHAUD'}
- {'history': [('LAYER', '.....')], 'output': '<guess>WOMIN</guess>', 'parsed': 'WOMIN'}
- {'history': [('XEBEC', '..-=.')], 'output': '<guess>BARLE</guess>', 'parsed': 'BARLE'}
| 2 | 100% | 80% | 69% | 0% | 9% | 2% | 20% | 100% | 90% |

Examples: prior=2, type=wrong_length
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>THOM</guess>', 'parsed': 'THOM'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>BADOOP</guess>', 'parsed': 'BADOOP'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>THOUGH</guess>', 'parsed': 'THOUGH'}

Examples: prior=2, type=not_allowed
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>VOMOC</guess>', 'parsed': 'VOMOC'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>TOBBY</guess>', 'parsed': 'TOBBY'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>COMBO</guess>', 'parsed': 'COMBO'}
| 3 | 99% | 74% | 66% | 3% | 16% | 8% | 20% | 100% | 40% |

Examples: prior=3, type=missing_tag
- {'history': [('SETUP', '.-.-.'), ('JUICE', '.=..-'), ('QUERY', '===..')], 'output': '<guess QUEN</guess>', 'parsed': 'QUEN'}
- {'history': [('TOUGH', '.....'), ('ADDED', '-..-.'), ('MAYBE', '.-..=')], 'output': '<guessSENSE</guess>', 'parsed': 'SENSE'}

Examples: prior=3, type=wrong_length
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>QUAL</guess>', 'parsed': 'QUAL'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>OUTH</guess>', 'parsed': 'OUTH'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>FOCU</guess>', 'parsed': 'FOCU'}

Examples: prior=3, type=not_allowed
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>QUOTO</guess>', 'parsed': 'QUOTO'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>BOUTH</guess>', 'parsed': 'BOUTH'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>POUTH</guess>', 'parsed': 'POUTH'}
| 4 | 100% | 77% | 59% | 0% | 8% | 6% | 40% | 100% | 50% |

Examples: prior=4, type=wrong_length
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>FOCU</guess>', 'parsed': 'FOCU'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>HOUR</guess>', 'parsed': 'HOUR'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>FOUK</guess>', 'parsed': 'FOUK'}

Examples: prior=4, type=not_allowed
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>GOUTH</guess>', 'parsed': 'GOUTH'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>BOUTH</guess>', 'parsed': 'BOUTH'}
- {'history': [('DAIRY', '.....'), ('GUESS', '.-...'), ('CIVIC', '.....'), ('THUMB', '-.=-.')], 'output': '<guess>MOUST</guess>', 'parsed': 'MOUST'}

