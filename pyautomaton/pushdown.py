#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Pushdown automaton
# Context-free grammar

from collections import deque
from graphviz import Digraph

from finit import EPCILON


class _symbol:
    index = 0

    def __init__(self, identity=None):
        if(identity is None):
            self.identity = '<Symbol' + str(_symbol.index) + '>'
            _symbol.index += 1
        else:
            self.identity = identity

    def __eq__(self, other):
        if(isinstance(other, _symbol)):
            return self.identity == other.identity
        else:
            return False

    def __str__(self):
        return str(self.identity)

    def __hash__(self):
        return hash(self.identity)


def _add_rule(rule_set, left, right):
    if(rule_set.get(left) is None):
        rule_set[left] = set()
    rule_set[left].add(right)


def _remove_rule(rule_set, left, right):
    rule_set[left].remove(right)


class CFG:

    def __init__(self, nonterminal_symbols, terminal_symbols, rules, start_symbol):
        self.nonterminal_symbols = nonterminal_symbols
        self.terminal_symbols = terminal_symbols
        self.rules = dict()
        for left, right_set in rules.items():
            self.rules[left] = set()
            for right in right_set:
                self.rules[left].add(tuple(right))

        self.start_symbol = start_symbol

    def __str__(self):
        s = 'nonterminal_symbols:\t{'
        if(len(self.nonterminal_symbols) >= 1):
            it = iter(self.nonterminal_symbols)
            s += str(next(it))
            for symbol in it:
                s += ', ' + str(symbol)
        s += '}\n'

        s += 'terminal_symbols:\t{'
        if(len(self.terminal_symbols) >= 1):
            it = iter(self.terminal_symbols)
            s += str(next(it))
            for symbol in it:
                s += ', ' + str(symbol)
        s += '}\n'

        s += 'rules:\t' + '\n'
        for left, right_set in self.rules.items():
            s += '\t' + str(left) + '\t->\t'

            if(len(right_set) > 0):
                it = iter(right_set)
                right = next(it)
                if(len(right) == 0):
                    s += str('ε')
                for w in right:
                    s += str(w)
                for right in it:
                    s += '|'
                    if(len(right) == 0):
                        s += str('ε')
                    for w in right:
                        s += str(w)
            s += '\n'

        s += 'start_symbol:\t' + str(self.start_symbol)

        return s

    def __remove_start_symbol_from_right(self, V, W, R, S):
        """すべての規則の右辺から開始記号を取り除く

        新開始記号S0を追加
        規則S0->Sを追加
        """
        V0 = V.copy()
        R0 = dict()
        S0 = _symbol('S0')

        for rule in R.items():
            left, right_set = rule
            R0[left] = right_set.copy()

        V0.add(S0)
        R0[S0] = {(self.start_symbol,)}

        return (V0, W, R0, S0)

    def __remove_nonsingle_end_symbol_from_right(self, V, W, R, S):
        """単独でない終端記号をすべての規則の右辺から取り除く

        規則A -> B0.B1.B2.... において 右辺の大きさが2以上の時かつ Bn in W ならば
        新記号Xbnを追加
        規則Xbn -> Bnを追加
        規則A -> B0.B1.B2.... で Bn in W を Xbn に取り換える
            規則A -> B0.Xb1.B2....Xbi.....みたいな感じ
        """
        V0 = V.copy()
        R0 = dict()

        for rule in R.items():
            left, right_set = rule
            R0[left] = right_set.copy()

        for left, right_set in R0.copy().items():
            for right in right_set.copy():
                if(len(right) >= 2):
                    nright = list(right)
                    for i in range(len(right)):
                        c = right[i]
                        if(c in W):
                            sx = _symbol(('X', c))
                            V0.add(sx)
                            nright[i] = sx
                            _add_rule(R0, sx, (c,))
                    _remove_rule(R0, left, right)
                    _add_rule(R0, left, tuple(nright))

        return (V0, W, R0, S)

    def __split_right_to_size_less2(self, V, W, R, S):
        """右辺の長さが３以上の規則の右辺を分割して2以下にする

        規則A -> B0.B1.B2....Bn において n > 2 のとき
        新記号Xを追加
        元の規則を削除して
        X -> B1.B2....Bn と
        A -> B0.Xを追加する
        追加できなくなるまで繰り返す
        """
        V0 = V.copy()
        R0 = dict()

        for rule in R.items():
            left, right_set = rule
            R0[left] = right_set.copy()

        que = list(R0.keys())
        while len(que):
            left = que.pop()
            right_set = R0[left]
            for right in right_set.copy():
                if(len(right) >= 3):
                    x = _symbol()
                    V0.add(x)
                    _remove_rule(R0, left, right)
                    _add_rule(R0, left, (right[0], x))
                    _add_rule(R0, x, right[1:])
                    que.append(x)

        return (V0, W, R0, S)

    def __remove_epcilon_rules(self, V, W, R, S):
        """空文字のある規則を取り除く

        A => εになる記号Aの集合Veを作る
        規則R0を A -> B(B != ε)とする
        規則R0において A -> B.C(B in Ve)ならばA -> Cを追加
        規則R0において A -> B.C(C in Ve)ならばA -> Bを追加
        S in Ve ならS -> εを追加する、R0に
        """
        Ve = set()

        # A => εになる記号Aの集合Veを作る
        for left in V:
            if tuple() in R[left]:
                Ve.add(left)

        lve = 0
        while(len(Ve) != lve):
            lve = len(Ve)
            for left in V - Ve:
                for right in R[left]:
                    r_is_ve = True
                    for c in right:
                        if c not in Ve:
                            r_is_ve = False
                    if(r_is_ve):
                        Ve.add(left)

        # R0を構成
        R0 = dict()
        for left in V:
            for right in R[left]:
                if (right != tuple()):
                    _add_rule(R0, left, right)
        for left, right_set in R0.items():
            for right in right_set.copy():
                if (len(right) == 2):
                    if(right[0] in Ve):
                        _add_rule(R0, left, (right[1],))
                    if(right[1] in Ve):
                        _add_rule(R0, left, (right[0],))
        if(S in Ve):
            _add_rule(R0, S, tuple())

        return (V, W, R0, S)

    def __remove_unit_rules(self, V, W, R, S):
        """単位規則を取り除く

        A -> B (A,B in V)ならば
            A -> c (B -> c in R)も規則に含む
        A -> B (A,B in V)をすべて取り除く
        """
        R0 = dict()
        for rule in R.items():
            left, right_set = rule
            R0[left] = right_set.copy()

        # A -> B (A,B in V)ならば
        # A -> c (B -> c in R0)もR0に含める
        pushed = True
        while(pushed):
            pushed = False
            for left, right_set in R0.items():
                for right in right_set.copy():
                    if(len(right) == 1):
                        if(right[0] in V):
                            for right_right in R0[right[0]]:
                                if(right_right not in right_set):
                                    pushed = True
                                    _add_rule(R0, left, right_right)

        # A -> B (A,B in V)をR0から取り除く
        for left, right_set in R0.items():
            for right in right_set.copy():
                if(len(right) == 1):
                    if(right[0] in V):
                        _remove_rule(R0, left, right)

        return (V, W, R0, S)

    def __remove_unused_rule(self, V, W, R, S):
        """不要なルールと記号を取り除く

        Rで終端記号からさかのぼって到達可能な記号のみV0に含む
        R0を A->B in R (A in V0 かつ B in (V0|W))で構成

        R0で開始記号から順にたどって到達可能な記号のみV1に含む
        R1を A->B in R0 (A in V1 かつ B in (V1|W))で構成

        """
        # Rで終端記号からさかのぼって到達可能な記号のみV0に含む
        V0 = set()
        for left in V:
            for right in R[left]:
                r_is_v0 = True
                for c in right:
                    if(c not in W):
                        r_is_v0 = False
                        break
                if(r_is_v0):
                    V0.add(left)

        lv0 = 0
        while(len(V0) != lv0):
            lv0 = len(V0)
            for left in V - V0:
                for right in R[left]:
                    r_is_v0 = True
                    for c in right:
                        if(c not in (V0 | W)):
                            r_is_v0 = False
                            break
                    if(r_is_v0):
                        V0.add(left)

        # R0を A->B in R (A in V0 かつ B in (V0|W))で構成
        R0 = dict()
        for left, right_set in R.items():
            if(left not in V0):
                continue
            for right in right_set:
                r_is_r0 = True
                for c in right:
                    if(c not in (V0 | W)):
                        r_is_r0 = False
                        break
                if(r_is_r0):
                    _add_rule(R0, left, right)

        # R0で開始記号から順にたどって到達可能な記号のみV1に含む
        V1 = {S}
        lv1 = 0
        while(len(V1) != lv1):
            lv1 = len(V1)
            for left in V1.copy():
                for right in R0[left]:
                    for c in right:
                        if(c in V0):
                            V1.add(c)

        # R1を A->B in R0 (A in V1 かつ B in (V1|W))で構成
        R1 = dict()
        for left, right_set in R0.items():
            if(left not in V1):
                continue
            for right in right_set:
                r_is_r1 = True
                for c in right:
                    if(c not in (V1 | W)):
                        r_is_r1 = False
                        break
                if(r_is_r1):
                    _add_rule(R1, left, right)

        return (V1, W, R1, S)

    def to_cnf(self):
        v0, w, r0, s0 = self.nonterminal_symbols, self.terminal_symbols, self.rules, self.start_symbol

        v0, w, r0, s0 = self.__remove_start_symbol_from_right(
            v0, w, r0, s0)

        v0, w, r0, s0 = self.__remove_nonsingle_end_symbol_from_right(
            v0, w, r0, s0)

        v0, w, r0, s0 = self.__split_right_to_size_less2(v0, w, r0, s0)

        v0, w, r0, s0 = self.__remove_epcilon_rules(v0, w, r0, s0)

        v0, w, r0, s0 = self.__remove_unit_rules(v0, w, r0, s0)

        v0, w, r0, s0 = self.__remove_unused_rule(v0, w, r0, s0)

        return CNF(v0, w, r0, s0)


class CNF(CFG):
    """Chomsky Normal Form
    """

    def __init__(self, nonterminal_symbols, terminal_symbols, rules, start_symbol):
        super().__init__(nonterminal_symbols, terminal_symbols, rules, start_symbol)

    def cyk(self, string_of_symbols):
        """check if this cfg can make input string
        """
        _, _, r0, s0 = self.nonterminal_symbols, self.terminal_symbols, self.rules, self.start_symbol

        # N is length of input string
        # a is input string
        # R is rules of cfg
        # (rule1)
        #   0 <= i < N
        #   X(0, i) = {A -> a[i]: A -> a[i] in R}
        #
        # (rule2)
        #   1 <= j < N,  0 <= i < N - j, 0 <= k <= j
        #   x(j, i, k) =
        #       {A -> BC:
        #           (A -> BC in R) & (B in X(j - k, i)) & (C in X(j - k, i + k))
        #       }
        #   X(j, i) = x(j, i, 0) | x(j, i, 1) | ... | x(j, i, j)
        #
        # if start symbol in X(N -1, 0)

        # RR: B -> {A: (A -> B in R)}
        reversed_r0 = dict()
        for left, right_set in r0.items():
            for right in right_set:
                rr0r = reversed_r0.get(right)
                if(rr0r == None):
                    rr0r = set()
                    reversed_r0[right] = rr0r
                rr0r.add(left)

        len_s = len(string_of_symbols)

        # (rule1)
        #   0 <= i < N
        #   X(0, i) = {A -> a[i]: A -> a[i] in R}
        x0 = [None] * len_s
        x0[0] = [None] * len_s
        i = 0
        for a in string_of_symbols:
            rr0a = reversed_r0.get((a,))
            if(rr0a == None):
                return False
            x0[0][i] = dict()
            for left in rr0a:
                x0[0][i][left] = ((a,), -1)
            i += 1

        # (rule2)
        #   1 <= j < N,  0 <= i < N - j, 0 <= k < j
        #   x(j, i, k) =
        #       {A:
        #           (A -> BC in R) & (B in X(k, i)) & (C in X(j - k - 1, i + k + 1))
        #       }
        #   X(j, i) = x(j, i, 0) | x(j, i, 1) | ... | x(j, i, j)
        for j in range(1, len_s):
            x0[j] = [None] * (len_s - j)
            for i in range(0, len_s - j):
                x0[j][i] = dict()
                for k in range(0, j):
                    for b in x0[k][i]:
                        for c in x0[j - k - 1][i + k + 1]:
                            bc = (b, c)
                            rr0bc = reversed_r0.get(bc)
                            if(rr0bc == None):
                                continue
                            for left in rr0bc:
                                x0[j][i][left] = (bc, k)

        can_cfg_mkstr = s0 in x0[len_s - 1][0].keys()
        tree = None
        if(can_cfg_mkstr):
            tree = str(s0) + '\n'
            q0 = deque([(len_s - 1, 0, s0)])
            q1 = deque([])

            new_line = ''
            while(len(q0)):
                j, i, left = q0.popleft()
                right, k = x0[j][i][left]

                if(len(new_line) != 0):
                    new_line += ', '
                if(k == -1):
                    new_line += str(right[0])
                else:
                    new_line += str(right[0]) + str(right[1])
                    q1.append((k, i, right[0]))
                    q1.append((j - k - 1, i + k + 1, right[1]))

                if(len(q0) == 0):
                    tree += new_line + '\n'
                    new_line = ''
                    q = q0
                    q0 = q1
                    q1 = q

        return can_cfg_mkstr, tree

    def to_cnf(self):
        v0, w, r0, s0 = self.nonterminal_symbols, self.terminal_symbols, self.rules, self.start_symbol
        return CFG(v0, w, r0, s0)

    def to_gnf(self):

        S = self.start_symbol

        has_epcilon = (tuple() in self.rules[S])
        if(has_epcilon):
            self.rules[S] -= {tuple()}

        V, W, R, S = self.nonterminal_symbols, self.terminal_symbols, self.rules, S

        Z = [_symbol() for i in range(len(V))]
        V0 = V.copy()

        R0 = dict()

        sym_to_num = dict()
        num_to_sym = [None] * len(V)
        i = 0
        for rule in R.items():
            left, right_set = rule
            sym_to_num[left] = i
            num_to_sym[i] = left
            i += 1
            R0[left] = right_set.copy()

        for i in range(len(V)):
            left = num_to_sym[i]
            unchecked = deque(R0[left])

            Ra = set()

            while(len(unchecked) > 0):
                right = unchecked.popleft()
                if(right[0] not in V):
                    continue
                left_id = sym_to_num[left]
                right_id = sym_to_num[right[0]]

                # 生成規則の置換
                if(left_id > right_id):
                    _remove_rule(R0, left, right)
                    for right_right in R0[right[0]]:
                        new_right = (*right_right, *right[1:])
                        _add_rule(R0, left, new_right)
                        unchecked.append(new_right)

                # 左再帰をメモ
                if left_id == right_id:
                    Ra.add(right)

            # 左再帰を取り除く
            if(len(Ra) > 0):
                V0.add(Z[i])
                for right in Ra:
                    _remove_rule(R0, left, right)
                    _add_rule(R0, Z[i], right[1:])
                    _add_rule(R0, Z[i], (*right[1:], Z[i]))
                for right in R0[left] - Ra:
                    _add_rule(R0, left, (*right, Z[i]))

        # 右辺の左端が終端記号でないようにする

        for i in range(len(V) - 1, -1, -1):
            left = num_to_sym[i]
            unchecked = deque(R0[left])
            while(len(unchecked) > 0):
                right = unchecked.popleft()
                if(right[0] in W):
                    continue
                # right[0] in Z
                if(right[0] not in V):
                    continue
                _remove_rule(R0, left, right)
                for right_right in R0[right[0]].copy():
                    new_right = (*right_right, *right[1:])
                    _add_rule(R0, left, new_right)
                    unchecked.append(new_right)

        for i in range(0, len(V)):
            left = Z[i]
            right_set = R0.get(left)
            if(right_set is None):
                continue
            unchecked = deque(right_set)
            while(len(unchecked) > 0):
                right = unchecked.popleft()
                if(right[0] in W):
                    continue
                # right[0] in Z
                if(right[0] not in V):
                    continue
                _remove_rule(R0, left, right)
                for right_right in R0[right[0]]:
                    new_right = (*right_right, *right[1:])
                    _add_rule(R0, left, new_right)
                    unchecked.append(new_right)

        if(has_epcilon):
            self.rules[S] |= {tuple()}
            R0 != {tuple()}

        return GNF(V0, W, R0, S)


class GNF(CFG):
    """Greibach Normal Form
    """

    def __init__(self, nonterminal_symbols, terminal_symbols, rules, start_symbol):
        super().__init__(nonterminal_symbols, terminal_symbols, rules, start_symbol)


class PDA:
    """Pushdown automaton
    """

    def __init__(self, states, input_symbols, stack_symbols, transition_relation, start_state, initial_stack_state, accepting_stetes):
        self.states = set(states)
        self.input_symbols = set(input_symbols)
        self.stack_symbols = set(stack_symbols)
        self.transition_relation = transition_relation
        self.start_state = start_state
        self.initial_stack_state = tuple(initial_stack_state)
        self.accepting_stetes = set(accepting_stetes)

    def input_symbol(self, symbol):
        cur_nodes = deque([(self.start_state, self.initial_stack_state)])

        next_nodes = deque([])
        for state, stack in cur_nodes:

            # T(state, stack_front, input)
            #   = {(next_state, next_stack_front[])}
            stack_front = EPCILON
            if(len(stack) > 0):
                stack_front = stack[0]
            next_datas = self.transition_relation.get((state, stack_front, symbol))
            if(next_datas is None):
                continue

            # next_state = next_state
            # next_stack = next_stack_front  + stack[]
            for next_state, next_stack_front in next_datas:
                next_stack = (*next_stack_front, *stack[1:])
                next_state_tuple = (next_state, next_stack)
                next_nodes.append(next_state_tuple)

        # exist state in self.accepting_stetes

        return [PDA(
            self.states,
            self.input_symbols,
            self.stack_symbols,
            self.transition_relation,
            state,
            stack,
            self.accepting_stetes
        )
            for state, stack in next_nodes]
