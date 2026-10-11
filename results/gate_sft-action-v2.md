# Gate: sft-action-v2 (A10, k=16, 100 states per stratum, T=1.0, plain prompt)

Code: commit `10b89fb` (with uncommitted changes)

| prior | tag | len5 | valid | repeat | consistent | solve | nondegen | nd_reward | nd_consist |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 100% | 100% | 100% | 0% | 100% | 0% | 0% | 1% | 1% |

Examples: prior=0, type=wrong_length
- {'history': [], 'output': '<guess>business</guess>', 'parsed': 'BUSINESS'}
| 1 | 100% | 88% | 82% | 0% | 37% | 0% | 1% | 100% | 87% |

Examples: prior=1, type=missing_tag
- {'history': [('DAIRY', '.....')], 'output': '<guess SCRIPT</guess>', 'parsed': 'SCRIPT'}
- {'history': [('CRAZY', '.....')], 'output': '<guessSNELL</guess>', 'parsed': 'SNELL'}
- {'history': [('ALGAL', '..-..')], 'output': '<guess GOODS</guess>', 'parsed': 'GOODS'}

Examples: prior=1, type=wrong_length
- {'history': [('LAYER', '.....')], 'output': '<guess>COBINA</guess>', 'parsed': 'COBINA'}
- {'history': [('LAYER', '.....')], 'output': '<guess>THIG</guess>', 'parsed': 'THIG'}
- {'history': [('LAYER', '.....')], 'output': '<guess>MONG</guess>', 'parsed': 'MONG'}

Examples: prior=1, type=not_allowed
- {'history': [('LAYER', '.....')], 'output': '<guess>SINON</guess>', 'parsed': 'SINON'}
- {'history': [('LAYER', '.....')], 'output': '<guess>RITCH</guess>', 'parsed': 'RITCH'}
- {'history': [('XEBEC', '..-=.')], 'output': '<guess>BOOLE</guess>', 'parsed': 'BOOLE'}
| 2 | 100% | 84% | 72% | 0% | 12% | 1% | 8% | 100% | 54% |

Examples: prior=2, type=missing_tag
- {'history': [('TOUCH', '..-..'), ('SUGAR', '.=.=-')], 'output': '<guess DRAM</guess>', 'parsed': 'DRAM'}
- {'history': [('LARRY', '==-.-')], 'output': '<guessLAURA</guess>', 'parsed': 'LAURA'}
- {'history': [('BURST', '..-..'), ('RAPID', '--..-')], 'output': '<guessANDLE</guess>', 'parsed': 'ANDLE'}

Examples: prior=2, type=wrong_length
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>THOMOOD</guess>', 'parsed': 'THOMOOD'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>THOMAD</guess>', 'parsed': 'THOMAD'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>BUDGET</guess>', 'parsed': 'BUDGET'}

Examples: prior=2, type=not_allowed
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>THOUS</guess>', 'parsed': 'THOUS'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>COMBO</guess>', 'parsed': 'COMBO'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>THOST</guess>', 'parsed': 'THOST'}
| 3 | 100% | 80% | 61% | 1% | 11% | 2% | 12% | 95% | 40% |

Examples: prior=3, type=missing_tag
- {'history': [('WARLY', '=....'), ('WHOSE', '==...')], 'output': '<guess>WHI<TResult</guess>', 'parsed': 'GUESS'}
- {'history': [('DIARY', '-.-..'), ('ADOPT', '=-...'), ('AMEND', '=.=.=')], 'output': '<guessachel</guess>', 'parsed': 'ACHEL'}
- {'history': [('PODAL', '.....'), ('CANDY', '..=.='), ('OZENA', '..--.')], 'output': '<guessGRNEY</guess>', 'parsed': 'GRNEY'}

Examples: prior=3, type=wrong_length
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>BOTH</guess>', 'parsed': 'BOTH'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>STUD</guess>', 'parsed': 'STUD'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>FROUNT</guess>', 'parsed': 'FROUNT'}

Examples: prior=3, type=not_allowed
- {'history': [('DAIRY', '...-.'), ('MERGE', '.---.'), ('CIVIC', '.....')], 'output': '<guess>FRENG</guess>', 'parsed': 'FRENG'}
- {'history': [('DAIRY', '...-.'), ('MERGE', '.---.'), ('CIVIC', '.....')], 'output': '<guess>REGRE</guess>', 'parsed': 'REGRE'}
- {'history': [('DAIRY', '...-.'), ('MERGE', '.---.'), ('CIVIC', '.....')], 'output': '<guess>GLEEB</guess>', 'parsed': 'GLEEB'}
| 4 | 100% | 80% | 60% | 2% | 7% | 3% | 22% | 97% | 34% |

Examples: prior=4, type=missing_tag
- {'history': [('BULAK', '.....'), ('GOUTY', '....='), ('PENNY', '.=..='), ('CHOGA', '.....')], 'output': '<guess FEVER</guess>', 'parsed': 'FEVER'}
- {'history': [('ACTOR', '.-...'), ('ABRET', '.....'), ('LUNCH', '...==')], 'output': '<guess THICK</guess>', 'parsed': 'THICK'}
- {'history': [('NASTY', '..-.-')], 'output': '<guessOLDER</guess>', 'parsed': 'OLDER'}

Examples: prior=4, type=wrong_length
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>FOCUST</guess>', 'parsed': 'FOCUST'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>JOUR</guess>', 'parsed': 'JOUR'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>FOCU</guess>', 'parsed': 'FOCU'}

Examples: prior=4, type=not_allowed
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>FOUST</guess>', 'parsed': 'FOUST'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>WHOUR</guess>', 'parsed': 'WHOUR'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>FOULT</guess>', 'parsed': 'FOULT'}

