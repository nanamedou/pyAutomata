#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Finit-state machine
# Regular grammar

from graphviz import Digraph


AUTOMATON_DEBUG = False
AUTOMATON_EXPORT_TYPE = 'png'  # svg


class _EPCILON:
    def __str__(self):
        return 'ε'


EPCILON = _EPCILON()


def _calc_reachable_nodes(start_state, input_symbols, transition_func, is_dfa=True):

    states = {start_state}

    unchecked_nodes = [start_state]
    while len(unchecked_nodes):
        s = unchecked_nodes.pop()
        for w in input_symbols:
            t_set = transition_func(s, w)
            if(t_set is None):
                continue
            if(is_dfa):
                t_set = {t_set}
            for t in t_set:
                if(t not in states):
                    unchecked_nodes.append(t)
                    states.add(t)

    return states


def _calc_equivalence_pairs(node, symbols, transition_func, accept_func):
    # Q = any node
    # n <= F (n <= Q and accept(n) is true)
    # G = Q - F
    F = set()
    G = set()
    for n in node:
        if(accept_func(n)):
            F.add(n)
        else:
            G.add(n)

    # unmarked = F x F | G x G
    unmarked = {(p, q) for p in F for q in F} | {(p, q) for p in G for q in G}

    # calc unmarked pairs
    # unmarked regulation
    #   any (p,q) <= unmarked
    #   any w <= symbols
    #   (transition(p), transition(q)) <= unmarked
    while(True):
        transitionable = set()
        for p, q in unmarked:
            is_unmarked = True

            for w in symbols:
                s = transition_func(p, w)
                t = transition_func(q, w)

                if (s, t) not in unmarked:
                    is_unmarked = False

            if(is_unmarked):
                transitionable.add((p, q))

        if(len(unmarked - transitionable) == 0):
            return unmarked

        else:
            unmarked = transitionable


def _equivalence_partition(pairs):
    same_set_list = []
    checked = set()
    # p <= pairs
    for (p, q) in pairs:
        if(p in checked):
            if(q in checked):
                # set A, set B (s.t p <= A, q <= B)
                # unite A and B
                sa = None
                sb = None
                for ssl in same_set_list:
                    if(p in ssl):
                        sa = ssl
                    if(q in ssl):
                        sb = ssl
                if(sa is not sb):
                    same_set_list.remove(sb)
                    sa |= sb
            else:
                # set A(s.t p <= A)
                # add q into A
                sa = None
                for ssl in same_set_list:
                    if(p in ssl):
                        sa = ssl
                        break
                sa.add(q)
                checked.add(q)
        elif(q in checked):
            # set B(s.t q <= B)
            # add p into B
            sb = None
            for ssl in same_set_list:
                if(q in ssl):
                    sb = ssl
                    break
            sb.add(p)
            checked.add(p)
        else:
            # create set A(s.t p,q <= A)
            same_set_list.append({p, q})
            checked |= {p, q}

    return same_set_list


class DFA:
    """決定性有限オートマトン
    """

    def __init__(self, start_state, input_symbols, transition_func, accept_func):
        """初期化関数

        DFAを初期化する。

        Args:
            start_state : DFAの初期状態
            input_symbols (set): DFAの入力に使うすべての文字集合
            transition_func: DFAの状態遷移関数 (now_state, input_symbol) -> next_state
            accept_func: 状態受理関数 state -> bool 受理されるときTrue, 拒否はFalse
        """
        self.start_state = start_state
        self.input_symbols = input_symbols
        self.transition_func = transition_func
        self.accept_func = accept_func

        self.states = _calc_reachable_nodes(
            start_state, input_symbols, transition_func)

    def generate_minimum_dfa(self):
        """最小化された等価なdfaを生成する

        最小化された等価なdfaを生成する。このDFA(Ma)新しく生成されるDFA(Mb)でL(Ma) == L(Mb)

        Returns:
            MinimumDFA: 最小化されたdfa
        """
        _equivalence_pairs = _calc_equivalence_pairs(
            self.states,
            self.input_symbols,
            self.transition_func,
            self.accept_func)
        equivalence_node = _equivalence_partition(_equivalence_pairs)

        # generte new node
        new_accept_set = set()
        for i in range(len(equivalence_node)):
            it = iter(equivalence_node[i])
            s = next(it)
            if(self.accept_func(s)):
                new_accept_set.add(i)

        # calc new start node
        new_start_state = None
        for i in range(len(equivalence_node)):
            if(self.start_state in equivalence_node[i]):
                new_start_state = i
                break

        # genetate new edge
        edge = dict()
        for i in range(len(equivalence_node)):
            it = iter(equivalence_node[i])
            s = next(it)
            for symbol in self.input_symbols:
                t = self.transition_func(s, symbol)
                for j in range(len(equivalence_node)):
                    if(t in equivalence_node[j]):
                        edge[(i, symbol)] = j
                        break

        # generate new transition func
        def transition(node, symbol):
            return edge[(node, symbol)]

        # generate new accept func
        def accept(node):
            return node in new_accept_set

        if AUTOMATON_DEBUG in globals():
            for i in range(len(equivalence_node)):
                print(str(i) + ': [', end='')
                print(*sorted(equivalence_node[i]), sep=',', end='')
                print(']')

        # return new minimized dfa
        return MinimumDFA(new_start_state, self.input_symbols, transition, accept, edge, new_accept_set)

    def export(self, path):
        """オートマトンを画像ファイルで出力する

        オートマトンをpng形式画像

        Args:
            path (string): 出力するファイルパス
        Returns:
            MinimumDFA: 最小化されたdfa
        """

        G = Digraph(format=AUTOMATON_EXPORT_TYPE)
        G.attr('node', shape='circle')

        # render node
        for s in self.states:
            s_name = str(s)
            additional_element = dict()
            if(self.accept_func(s)):
                additional_element['shape'] = 'doublecircle'
            if(s == self.start_state):
                additional_element['fillcolor'] = 'gray'
                additional_element['style'] = 'solid,filled'
            G.node(s_name, s_name, **additional_element)

        # render edge
        for s in self.states:
            edge = dict()
            for w in self.input_symbols:
                t = self.transition_func(s, w)
                if(t is None):
                    continue
                if(edge.get(t) == None):
                    edge[t] = str(w)
                else:
                    edge[t] += ', ' + str(w)
            for t, w in edge.items():
                G.edge(str(s), str(t), label=edge[t])

        G.render(path)

    def run(self, string_of_symbols):
        """オートマトンへの入力が受理されるかどうか確かめる

        Args:
            string_of_symbols: input_symbolsに含まれるシンボルの列
        Returns:
            最終状態の値と、受理かどうかのbool値の長さ2のタプルが戻る
        """

        node = self.start_state
        for symbol in string_of_symbols:
            if symbol not in self.input_symbols:
                return (None, False)
            node = self.transition_func(node, symbol)
            if(node is None):
                return (None, False)
        return (node, self.accept_func(node))


class MinimumDFA(DFA):
    """最小状態の決定性有限オートマトン
    """

    def __init__(self, start_state, input_symbols, transition_func, accept_func, edge_dict, accept_set):
        """初期化関数

        MinimumDFAを初期化する。

        Args:
            start_state: DFAの初期状態
            input_symbols (set): DFAの入力に使うすべての文字集合
            transition_func: DFAの状態遷移関数 (now_state, input_symbol) -> next_state
            accept_func: 状態受理関数 state -> bool 受理されるときTrue, 拒否はFalse
            edge_dict (dict): 状態遷移の表 edge_dict[(状態,入力)] -> 遷移先の状態
            accept_set (set): すべての受理状態の集合
        """
        super().__init__(start_state, input_symbols, transition_func, accept_func)
        self.edge = edge_dict
        self.accept_states = accept_set

    def generate_minimum_dfa(self):
        """最小化された等価なdfaを生成する

        自身のコピーを返す

        Returns:
            MinimumDFA: 最小化されたdfa
        """

        return MinimumDFA(self.start_state, self.input_symbols, self.transition_func, self.accept_func, self.edge, self.accept_states)

    def generate_reguler_expresion(self):
        """等価な正規表現を生成する

        Returns:
            RE: 正規表現
        """

        # 言語を木で表す
        # ノードNが(ラベル, 値)になっていて
        #   N('sum', [参照ノード0,ノード1,...])
        #   N('connect', [参照ノード0,ノード1,...])
        #   N('clojure', 参照ノード)
        #   N('val', シンボル)
        # の四種類が存在

        def reg_sum(a, b):
            """言語a,bの和
            """
            if(a is None and b is None):
                return None
            if(a is None):
                return b
            if(b is None):
                return a

            a_tree = None
            b_tree = None
            if(a[0] == 'sum'):
                a_tree = a[1]
            else:
                a_tree = [a]
            if(b[0] == 'sum'):
                b_tree = b[1]
            else:
                b_tree = [b]
            return ('sum', a_tree + b_tree)

        def reg_conn(a, b):
            """言語a,bの順列
            """
            if(a is None and b is None):
                return None
            if(a is None):
                return b
            if(b is None):
                return a

            a_tree = None
            b_tree = None
            if(a[0] == 'connect'):
                a_tree = a[1]
            else:
                a_tree = [a]
            if(b[0] == 'connect'):
                b_tree = b[1]
            else:
                b_tree = [b]
            return ('connect', a_tree + b_tree)

        def reg_closure(a):
            """言語a,bの閉包を作る
            """
            if(a is None):
                return None
            if(a[0] != 'closure'):
                a = ('closure', a)
            return a

        def get_dij(d, i, j):
            """状態i -> j への遷移を表す言語を取る
            """
            di = d.get(i)
            if(di is None):
                return None
            return di.get(j)

        F = dict()
        T = dict()
        # 接続されたノードの遷移を表す言語を作る
        for ((p, symbol), q) in self.edge.items():
            if(F.get(p) is None):
                F[p] = dict()
            if(F[p].get(q) is None):
                F[p][q] = ('val', symbol)
            else:
                F[p][q] = reg_sum(F[p][q], ('val', symbol))
            if(T.get(q) is None):
                T[q] = set()
            T[q].add(p)

        def bypass(A, F, T, q):
            """状態q を取り除く

            遷移p -> q, q -> rをまとめてp->rにする
            状態集合Aからqを取り除く
            """
            # L(pr) = L(pr) + L(pq).L(qq)*.L(qr)
            for p in T[q]:
                if(p not in A):
                    continue
                if(F.get(q) is None):
                    break
                for r in F[q].keys():
                    if(r not in A):
                        continue
                    if(p == q) or (q == r):
                        continue
                    qq = F[q].get(q)
                    S = None
                    S = reg_conn(
                        reg_conn(F[p][q], reg_closure(qq)), F[q][r])

                    F[p][r] = reg_sum(F[p].get(r), S)

                    T[r].add(p)
            A.remove(q)

        def mk_fit_F_T(A, F, T):
            """F,Tから不要な遷移を取り除く
            """
            new_F = dict()
            new_T = dict()
            for p in A:
                if(F.get(p) is None):
                    continue
                for q in F[p].keys():
                    if(q in A):
                        if(new_F.get(p) is None):
                            new_F[p] = dict()
                        new_F[p][q] = F[p][q]
                        if(new_T.get(q) is None):
                            new_T[q] = set()
                        new_T[q].add(p)
            return new_F, new_T

        # 開始ノードと受理ノード以外を取り除く
        active_node = self.states.copy()
        remove_nodes = (self.states -
                        {self.start_state} - self.accept_states)
        while len(remove_nodes):
            q = remove_nodes.pop()
            bypass(active_node, F, T, q)

        # 開始ノードsと受理ノードaひとつを残しノードをすべて取り除く
        #   言語Lを構成
        #   状態数1のときL = L(aa)*
        #   状態数2のときL = (L(ss) + L(sa).L(aa)*.L(as))*.L(sa).L(aa)*
        # 元に戻す
        # これを全部の受理ノードで試す
        # 構成すべての言語の和がもとのDFAの正規表現になる
        S = None
        for a in self.accept_states:

            nac = active_node.copy()
            nF, nT = mk_fit_F_T(active_node, F, T)
            remove_nodes = active_node - {self.start_state, a}
            while len(remove_nodes):
                q = remove_nodes.pop()
                bypass(nac, nF, nT, q)

            m = None
            if(self.start_state == a):
                a_a = get_dij(nF, a, a)
                if(a_a is None):
                    m = ('val', 'ε')
                else:
                    m = reg_closure(a_a)
            else:
                s = self.start_state
                s_s = get_dij(nF, s, s)
                s_a = get_dij(nF, s, a)
                a_s = get_dij(nF, a, s)
                a_a = get_dij(nF, a, a)
                n = None
                if (s_a is not None and a_s is not None):
                    n = reg_conn(s_a, reg_conn(reg_closure(a_a), a_s))
                m = reg_closure(reg_sum(s_s, n))
                m = reg_conn(m, s_a)
                m = reg_conn(m, reg_closure(a_a))
            S = reg_sum(S, m)

        regex = []

        # s = (str, [])
        def translate_to_regex(s):
            if(s[0] == 'sum'):
                # a+b+c
                it = iter(s[1])
                translate_to_regex(next(it))
                for w in it:
                    regex.append('+')
                    translate_to_regex(w)
            if(s[0] == 'connect'):
                # abc
                for w in s[1]:
                    # ab(c + d)
                    if(w[0] == 'sum'):
                        regex.append('(')
                        translate_to_regex(w)
                        regex.append(')')
                    # abc
                    else:
                        translate_to_regex(w)
            if(s[0] == 'closure'):
                if(s[1][0] == 'sum' or s[1][0] == 'connect'):
                    regex.append('(')
                    translate_to_regex(s[1])
                    regex.append(')')
                else:
                    translate_to_regex(s[1])
                regex.append('*')
            if(s[0] == 'val'):
                regex.append(s[1])

        translate_to_regex(S)

        return RE(regex)

    def __eq__(self, other):
        if(not isinstance(other, MinimumDFA)):
            return False

        if(len(self.states) != len(other.states)):
            return False
        if(len(set(self.input_symbols) ^ set(other.input_symbols)) != 0):
            return False

        # Tree A, B
        # A equal B means...
        #   arbitrary x in A.states
        #   exist y in B.states
        #   exist F: x -> y
        #   s.t
        #     arbitrary w in symbols
        #     A.transition(x,w) == B.transition(y,w)
        #     A.accept(x) == B.accept(y)
        if(self.accept_func(self.start_state) != other.accept_func(other.start_state)):
            return False
        node_other_id = {self.start_state: other.start_state}
        unchecked_nodes = [self.start_state]
        while len(unchecked_nodes):
            self_s = unchecked_nodes.pop()
            for w in self.input_symbols:
                self_t = self.transition_func(self_s, w)
                other_t = other.transition_func(node_other_id[self_s], w)

                checked_other_t_id = node_other_id.get(self_t)
                if(checked_other_t_id is None):
                    if(self.accept_func(self_t) != other.accept_func(other_t)):
                        return False
                    node_other_id[self_t] = other_t
                    unchecked_nodes.append(self_t)
                elif(checked_other_t_id != other_t):
                    return False

        return True


class NFA:

    # transition_func: (node, symbol) => {node}
    def __init__(self, start_state, input_symbols, transition_func, accept_func):
        self.start_state = start_state
        self.input_symbols = input_symbols
        self.transition_func = transition_func
        self.accept_func = accept_func

        self.states = _calc_reachable_nodes(
            start_state, input_symbols, transition_func, is_dfa=False)

    def generate_dfa(self):

        # calc new start node
        new_start_state = (self.start_state,)

        # generate new transition func
        def transition(node, symbol):
            transitionable_node = set()
            for n in node:
                transitionable_node |= self.transition_func(n, symbol)
            return tuple(sorted(list(transitionable_node)))

        # generate new accept func
        def accept(node):
            is_ac = False
            for n in node:
                if(self.accept_func(n)):
                    is_ac = True
                    break
            return is_ac

        # return new minimized dfa
        return DFA(new_start_state, self.input_symbols, transition, accept)

    def export(self, path):
        G = Digraph(format=AUTOMATON_EXPORT_TYPE)
        G.attr('node', shape='circle')

        for s in self.states:
            s_name = str(s)
            additional_element = dict()
            if(self.accept_func(s)):
                additional_element['shape'] = 'doublecircle'
            if(s == self.start_state):
                additional_element['fillcolor'] = 'gray'
                additional_element['style'] = 'solid,filled'
            G.node(s_name, s_name, **additional_element)

        for s in self.states:
            edge = dict()
            for w in self.input_symbols:
                t_set = self.transition_func(s, w)
                for t in t_set:
                    if(edge.get(t) == None):
                        edge[t] = str(w)
                    else:
                        edge[t] += ', ' + str(w)
            for t, w in edge.items():
                G.edge(str(s), str(t), label=edge[t])

        print(G)

        G.render(path)

    # check automaton accept string or not
    def run(self, string_of_symbols):
        nodes = {self.start_state}
        for symbol in string_of_symbols:
            if symbol not in self.input_symbols:
                return (None, False)
            new_nodes = set()
            for n in nodes:
                new_nodes |= self.transition_func(n, symbol)
            nodes = new_nodes

        accepting = False
        for n in nodes:
            if(self.accept_func(n)):
                accepting = True

        return (nodes, accepting)


class EpsilonNFA:

    # transition_func: (node, symbol) => {node}
    def __init__(self, start_state, input_symbols, transition_func, accept_func):
        self.start_state = start_state
        self.input_symbols = set(input_symbols)
        self.transition_func = transition_func
        self.accept_func = accept_func

        self.states = _calc_reachable_nodes(
            start_state, set(input_symbols) | {EPCILON}, transition_func, is_dfa=False)

    def export(self, path):
        G = Digraph(format=AUTOMATON_EXPORT_TYPE)
        G.attr('node', shape='circle')

        for s in self.states:
            s_name = str(s)
            additional_element = dict()
            if(self.accept_func(s)):
                additional_element['shape'] = 'doublecircle'
            if(s == self.start_state):
                additional_element['fillcolor'] = 'gray'
                additional_element['style'] = 'solid,filled'
            G.node(s_name, s_name, **additional_element)

        for s in self.states:
            edge = dict()
            for w in self.input_symbols | {EPCILON}:
                t_set = self.transition_func(s, w)
                for t in t_set:
                    if(edge.get(t) == None):
                        edge[t] = str(w)
                    else:
                        edge[t] += ', ' + str(w)
            for t, w in edge.items():
                G.edge(str(s), str(t), label=edge[t])

        G.render(path)

    def generate_nfa(self):

        # calc new start node
        new_start_state = self.start_state

        # make table
        table = dict()
        accepts = set()
        # transition'(p,a) := transition(ECLOSE(p),a)
        # exist p in ECLOSE(q) s.t p is accept => q in accepts

        for n in self.states:
            eclose_ns = _calc_reachable_nodes(
                n, {EPCILON}, self.transition_func, is_dfa=False)
            for a in self.input_symbols:
                new_nodes = set()
                for en in eclose_ns:
                    new_nodes |= self.transition_func(en, a)
                table[(n, a)] = new_nodes

            for en in eclose_ns:
                if(self.accept_func(en)):
                    accepts.add(n)
                    break

        # generate new transition func
        def transition(node, symbol):
            trans = table.get((node, symbol))
            if(trans is None):
                return set()
            return trans

        # generate new accept func
        def accept(node):
            return node in accepts

        # return
        return NFA(new_start_state, self.input_symbols, transition, accept)

    # check actomaton accept string or not
    def run(self, string_of_symbols):
        # calc nodes = ECLOSE(start)
        nodes = _calc_reachable_nodes(
            self.start_state, {EPCILON}, self.transition_func, is_dfa=False)

        # calc
        # o(n, xa) = ECLOSE(o(o(n,x),a))
        for symbol in string_of_symbols:
            # discord undefined symbols
            if symbol not in self.input_symbols:
                return (set(), False)

            # new_node = ECLOSE(transition(nodes, symbol))
            new_nodes = set()
            for n in nodes:
                for m in self.transition_func(n, symbol):
                    new_nodes |= _calc_reachable_nodes(
                        m, {EPCILON}, self.transition_func, is_dfa=False)
            nodes = new_nodes

        accepting = False
        for n in nodes:
            if(self.accept_func(n)):
                accepting = True

        return (nodes, accepting)


class RE:

    def __init__(self, regular_expression):
        self.regex = regular_expression

    def generate_e_nfa(self):

         # calc new start node
        new_start_state = 0

        # make table
        def add_edge(dictionaly, start, label, goal):
            key = (start, label)
            v = dictionaly.get(key)
            if(v is None):
                dictionaly[key] = goal
            else:
                v |= goal

        edge_dict = dict()
        accepts = set()
        symbols = set()

        # generate tree subsets
        def gen_parts(regex, node_s=0, node_t=1, node_cnt=2):
            node_cur = node_s
            node_repeet_beg = node_s

            add_edge(edge_dict, node_s, EPCILON, {node_cnt})
            node_repeet_beg = node_cur
            node_cur = node_cnt
            node_cnt += 1

            i = 0
            while i < len(regex):
                r = regex[i]
                if(r == '+'):
                    add_edge(edge_dict, node_cur, EPCILON, {node_t})
                    node_cur = node_s

                    add_edge(edge_dict, node_s, EPCILON, {node_cnt})
                    node_repeet_beg = node_cur
                    node_cur = node_cnt
                    node_cnt += 1
                elif(r == '*'):
                    add_edge(edge_dict, node_repeet_beg, EPCILON, {node_cur})
                    add_edge(edge_dict, node_cur, EPCILON, {node_repeet_beg})
                elif(r == '('):
                    parentheses_depth = 1
                    j = i + 1
                    while j < len(regex):
                        if(regex[j] == '('):
                            parentheses_depth += 1
                        elif(regex[j] == ')'):
                            parentheses_depth -= 1
                            if(parentheses_depth == 0):
                                new_node_cnt = gen_parts(
                                    regex[i + 1:j], node_cur, node_cnt, node_cnt + 1)
                                node_repeet_beg = node_cur
                                node_cur = node_cnt
                                node_cnt = new_node_cnt
                                break
                        j += 1
                    i = j
                elif(r == 'ε'):
                    add_edge(edge_dict, node_cur, EPCILON, {node_t})
                else:
                    symbols.add(r)
                    add_edge(edge_dict, node_cur, r, {node_cnt})
                    node_repeet_beg = node_cur
                    node_cur = node_cnt
                    node_cnt += 1
                i += 1

            add_edge(edge_dict, node_cur, EPCILON, {node_t})

            return node_cnt

        gen_parts(self.regex, 0, 1, 2)
        accepts.add(1)

        # generate new transition func
        def transition(node, symbol):
            trans = edge_dict.get((node, symbol))
            if(trans is None):
                return set()
            return trans

        # generate new accept func
        def accept(node):
            return node in accepts

        # return
        return EpsilonNFA(new_start_state, symbols, transition, accept)
