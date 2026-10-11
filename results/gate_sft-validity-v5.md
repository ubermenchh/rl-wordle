# Gate: sft-validity-v5 (A10, k=16, 100 states per stratum, T=1.0, plain prompt)

Code: commit `10b89fb` (with uncommitted changes)

| prior | tag | len5 | valid | repeat | consistent | solve | nondegen | nd_reward | nd_consist |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 100% | 100% | 99% | 0% | 99% | 0% | 0% | 11% | 11% |

Examples: prior=0, type=missing_tag
- {'history': [], 'output': '<guess(Cardboard</guess>', 'parsed': 'GUESS'}

Examples: prior=0, type=wrong_length
- {'history': [], 'output': '<guess>Citizen</guess>', 'parsed': 'CITIZEN'}
- {'history': [], 'output': '<guess>Spirit</guess>', 'parsed': 'SPIRIT'}
- {'history': [], 'output': '<guess>Nursing</guess>', 'parsed': 'NURSING'}

Examples: prior=0, type=not_allowed
- {'history': [], 'output': '<guess>JAGAN</guess>', 'parsed': 'JAGAN'}
- {'history': [], 'output': '<guess>VISNS</guess>', 'parsed': 'VISNS'}
- {'history': [], 'output': '<guess>CABOR</guess>', 'parsed': 'CABOR'}
| 1 | 100% | 92% | 82% | 0% | 45% | 0% | 0% | 96% | 87% |

Examples: prior=1, type=missing_tag
- {'history': [('HOPED', '...-=')], 'output': '<guess-TAKEN</guess>', 'parsed': 'TAKEN'}

Examples: prior=1, type=wrong_length
- {'history': [('LAYER', '.....')], 'output': '<guess>QUOT</guess>', 'parsed': 'QUOT'}
- {'history': [('XEBEC', '..-=.')], 'output': '<guess>BOلت</guess>', 'parsed': 'BOلت'}
- {'history': [('XEBEC', '..-=.')], 'output': '<guess>BLERRY</guess>', 'parsed': 'BLERRY'}

Examples: prior=1, type=not_allowed
- {'history': [('LAYER', '.....')], 'output': '<guess>SIXAN</guess>', 'parsed': 'SIXAN'}
- {'history': [('LAYER', '.....')], 'output': '<guess>BOSIC</guess>', 'parsed': 'BOSIC'}
- {'history': [('LAYER', '.....')], 'output': '<guess>NTIME</guess>', 'parsed': 'NTIME'}
| 2 | 100% | 87% | 74% | 0% | 13% | 1% | 9% | 92% | 51% |

Examples: prior=2, type=missing_tag
- {'history': [('LOTUS', '.-..='), ('CROSS', '=.-.=')], 'output': '<guess,COSVE</guess>', 'parsed': 'GUESS'}

Examples: prior=2, type=wrong_length
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess> Chattanooga</guess>', 'parsed': 'CHATTANOOGA'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>CHOUGH</guess>', 'parsed': 'CHOUGH'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>CUMU</guess>', 'parsed': 'CUMU'}

Examples: prior=2, type=not_allowed
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>THUNK</guess>', 'parsed': 'THUNK'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....')], 'output': '<guess>THOUS</guess>', 'parsed': 'THOUS'}
- {'history': [('TOUCH', '..-..'), ('SUGAR', '.=.=-')], 'output': '<guess>RAYAL</guess>', 'parsed': 'RAYAL'}
| 3 | 100% | 87% | 69% | 3% | 12% | 4% | 11% | 94% | 36% |

Examples: prior=3, type=missing_tag
- {'history': [('ROUGH', '.====')], 'output': '<guess监听</guess>', 'parsed': 'GUESS'}

Examples: prior=3, type=wrong_length
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>WHOUTH</guess>', 'parsed': 'WHOUTH'}
- {'history': [('PASTE', '.-.-.'), ('VITAL', '..--.'), ('CRAFT', '-==.-')], 'output': '<guess>ORCH</guess>', 'parsed': 'ORCH'}
- {'history': [('PASTE', '.-.-.'), ('VITAL', '..--.'), ('CRAFT', '-==.-')], 'output': '<guess>DROT</guess>', 'parsed': 'DROT'}

Examples: prior=3, type=not_allowed
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>WOUTH</guess>', 'parsed': 'WOUTH'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>WOUTH</guess>', 'parsed': 'WOUTH'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>WOUTH</guess>', 'parsed': 'WOUTH'}
| 4 | 100% | 86% | 66% | 3% | 5% | 2% | 16% | 94% | 25% |

Examples: prior=4, type=wrong_length
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>DUKE</guess>', 'parsed': 'DUKE'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>WUSH</guess>', 'parsed': 'WUSH'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>WHOUTH</guess>', 'parsed': 'WHOUTH'}

Examples: prior=4, type=not_allowed
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>THOCK</guess>', 'parsed': 'THOCK'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>WOUTH</guess>', 'parsed': 'WOUTH'}
- {'history': [('LAYER', '.....'), ('XENIA', '.....'), ('MOUTH', '.----')], 'output': '<guess>HOUTH</guess>', 'parsed': 'HOUTH'}

