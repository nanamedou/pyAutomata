#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyautomaton.pushdown import CNF

nonterminal_symbols = {'S','A','B','C'}
terminal_symbols = {'a', 'b'}
rules = {
    'S': {'AB','BB'}, 
    'A': {'CC', 'AB','a'},
    'B': {'BB', 'CA', 'b'},
    'C': {'AA', 'BA', 'b'}
    }
start_symbol = 'S'

cnf = CNF(nonterminal_symbols,terminal_symbols, rules, start_symbol)
print(cnf)

print('------------------')
gnf = cnf.to_gnf()


print(gnf)

print('------------------')

re, tree = cnf.cyk('aabb')
print(re)
print(tree)

