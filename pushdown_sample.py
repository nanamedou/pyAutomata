#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque
from pyautomaton.pushdown import PDA,EPCILON

EPCILON = ''
states = {0,1,2}
input_symbols = '01'
stack_symbols = '01Z'
transition = {
    (0,'Z',''):{(1,'Z')},
    (1,'Z',''):{(2,'Z')},
    (0,'Z','0'):{(0,'0Z'), (1,'0Z')},
    (0,'Z','1'):{(0, '1Z'), (1,'1Z')},
    (0,'0','0'):{(0, '00'), (1,'00')},
    (0,'0','1'):{(0, '10'), (1,'10')},
    (0,'1','0'):{(0, '01'), (1,'01')},
    (0,'1','1'):{(0, '11'), (1,'11')},
    (1,'0','0'):{(1,'')},
    (1,'1','1'):{(1,'')},
    }
start_symbol = 'S'

pda = PDA(states,input_symbols, stack_symbols, transition, 0,'Z',{2})


case = {(pda.start_state, pda.initial_stack_state)}
print (case)
for w in (*'001100',EPCILON):
    ncase = set()
    for state, stack in case:
        stack_last = EPCILON
        if(len(stack) > 0):
            stack_last = stack[0]
        for a, b in pda.input_symbol(state, stack_last, w):
            ncase.add((a, (*b,*stack[1:])))

    case = ncase
    print (case)

accepted = False
for state, stack in case:
    if(state in pda.accepting_stetes):
        accepted = True

print (accepted)



