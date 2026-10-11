# Gate: sft-validity-v4 (A10, k=16, 100 states per stratum, T=1.0, plain prompt)

Code: commit `10b89fb` (with uncommitted changes)

| prior | tag | len5 | valid | repeat | consistent | solve | nondegen | nd_reward | nd_consist |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 100% | 100% | 100% | 0% | 100% | 0% | 0% | 2% | 2% |

Examples: prior=0, type=missing_tag
- {'history': [], 'output': '<guess>', 'parsed': 'GUESS'}

Examples: prior=0, type=not_allowed
- {'history': [], 'output': '<guess>RASPB</guess>', 'parsed': 'RASPB'}
| 1 | 100% | 90% | 85% | 0% | 45% | 0% | 0% | 98% | 89% |

Examples: prior=1, type=missing_tag
- {'history': [('BELLY', '.--=.')], 'output': '<guessALTER</guess>', 'parsed': 'ALTER'}

Examples: prior=1, type=wrong_length
- {'history': [('LAYER', '.....')], 'output': '<guess>ST friction</guess>', 'parsed': 'ST FRICTION'}
- {'history': [('LAYER', '.....')], 'output': '<guess>MONBET</guess>', 'parsed': 'MONBET'}
- {'history': [('XEBEC', '..-=.')], 'output': '<guess>BUKE</guess>', 'parsed': 'BUKE'}

Examples: prior=1, type=not_allowed
- {'history': [('LAYER', '.....')], 'output': '<guess>THOWN</guess>', 'parsed': 'THOWN'}
- {'history': [('XEBEC', '..-=.')], 'output': '<guess>BADEN</guess>', 'parsed': 'BADEN'}
- {'history': [('TOUCH', '..-..')], 'output': '<guess>RUPPY</guess>', 'parsed': 'RUPPY'}
| 2 | 100% | 87% | 75% | 0% | 14% | 2% | 6% | 96% | 50% |

Examples: prior=2, type=missing_tag
- {'history': [('WIDTH', '.....'), ('BACON', '.=-.-')], 'output': '<guess CARE</guess>', 'parsed': 'CARE'}
- {'history': [('OPERA', '....-'), ('ALBUM', '-=-..')], 'output': '<guessBLAST</guess>', 'parsed': 'BLAST'}

Examples: prior=2, type=wrong_length
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>LOGO</guess>', 'parsed': 'LOGO'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>UPOUND</guess>', 'parsed': 'UPOUND'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>ODD</guess>', 'parsed': 'ODD'}

Examples: prior=2, type=not_allowed
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>THOTO</guess>', 'parsed': 'THOTO'}
- {'history': [('TOUCH', '..-..'), ('SUGAR', '.=.=-')], 'output': '<guess>RUBBA</guess>', 'parsed': 'RUBBA'}
- {'history': [('TOUCH', '..-..'), ('SUGAR', '.=.=-')], 'output': '<guess>RUBIN</guess>', 'parsed': 'RUBIN'}
| 3 | 100% | 81% | 69% | 1% | 11% | 3% | 9% | 96% | 36% |

Examples: prior=3, type=missing_tag
- {'history': [('JIMMY', '.....'), ('AGREE', '-....'), ('TIGHT', '.....')], 'output': '<guess狻</guess>', 'parsed': 'GUESS'}
- {'history': [('CREEK', '-...-'), ('UPBAY', '-...=')], 'output': '<guess>KOCK<nusty>STUDY</guess>', 'parsed': 'GUESS'}
- {'history': [('CUTCH', '.....'), ('ARMIL', '.=...'), ('PROOF', '===..')], 'output': '<guessPROXY</guess>', 'parsed': 'PROXY'}

Examples: prior=3, type=wrong_length
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>OUTH</guess>', 'parsed': 'OUTH'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>OUTH</guess>', 'parsed': 'OUTH'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>OUTH</guess>', 'parsed': 'OUTH'}

Examples: prior=3, type=not_allowed
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>BOUTH</guess>', 'parsed': 'BOUTH'}
- {'history': [('DAIRY', '...-.'), ('MERGE', '.---.'), ('CIVIC', '.....')], 'output': '<guess>REGRU</guess>', 'parsed': 'REGRU'}
- {'history': [('PASTE', '.-.-.'), ('VITAL', '..--.'), ('CRAFT', '-==.-')], 'output': '<guess>TRAND</guess>', 'parsed': 'TRAND'}
| 4 | 100% | 82% | 68% | 2% | 7% | 3% | 15% | 94% | 24% |

Examples: prior=4, type=missing_tag
- {'history': [('BULAK', '.....'), ('GOUTY', '....='), ('PENNY', '.=..='), ('CHOGA', '.....')], 'output': '<guess fullfield</guess>', 'parsed': 'FULLFIEL'}

Examples: prior=4, type=wrong_length
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>OUTH</guess>', 'parsed': 'OUTH'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>OUTH</guess>', 'parsed': 'OUTH'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>OUTH</guess>', 'parsed': 'OUTH'}

Examples: prior=4, type=not_allowed
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>TOTUS</guess>', 'parsed': 'TOTUS'}
- {'history': [('DAIRY', '.....'), ('GUESS', '.-...'), ('CIVIC', '.....'), ('THUMB', '-.=-.')], 'output': '<guess>MOTUS</guess>', 'parsed': 'MOTUS'}
- {'history': [('TWICE', '..=.='), ('ASIDE', '.-=.='), ('SPINE', '=.=.='), ('PORTA', '.....')], 'output': '<guess>SIGUE</guess>', 'parsed': 'SIGUE'}

