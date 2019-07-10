#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyautomaton.finit import DFA,NFA,EpsilonNFA,RE

re = RE('ε+(st+s)*+uuu+(s+t)u')
e_nfa = re.generate_e_nfa()
nfa = e_nfa.generate_nfa()
dfa = nfa.generate_dfa()
min_dfa = dfa.generate_minimum_dfa()


e_nfa.export('tmp/e_nfa')
nfa.export('tmp/nfa')
dfa.export('tmp/dfa')
min_dfa.export('tmp/min-dfa')

re2 = min_dfa.generate_reguler_expresion()
e_nfa2 = re2.generate_e_nfa()
nfa2 = e_nfa2.generate_nfa()
dfa2 = nfa2.generate_dfa()
min_dfa2 = dfa2.generate_minimum_dfa()
min_dfa2.export('tmp/min-dfa2')

print(*re2.regex, sep='')
print(min_dfa == min_dfa2)

print(min_dfa.run('su'))
print(min_dfa.run('sssssssst'))
