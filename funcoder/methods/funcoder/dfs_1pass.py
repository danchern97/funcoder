from ...langrt import LrtFunctionDef, LrtNode, LrtProgram, SymbolName
from ..shared import CodeGenContext, CodeGenJournal, CodeGenJournalist
from .dfs_2pass import Node
from .gen_once import GenOnceSig


class FunCoderDfs1Pass:
    def __init__(
        self,
        ctx: CodeGenContext,
        opt_max_depth: int,
        gen_pass_1: GenOnceSig,
        ancestors: list[LrtNode],
        func: LrtFunctionDef,
        descendants: list[LrtNode],
    ):
        self.ctx = ctx
        self.opt_max_depth = opt_max_depth
        self.gen_pass_1 = gen_pass_1

        self.funcs: dict[SymbolName, LrtFunctionDef] = {}
        self.vis: dict[SymbolName, bool] = {}
        self.shared_descendant_nodes: list[LrtNode] = []
        self.shared_descendant_funcs: list[SymbolName] = []

        for n in ancestors:
            if n.kind == "function":
                self.funcs[n.name] = n
                self.vis[n.name] = True
            else:
                self.shared_descendant_nodes.append(n)
        for n in descendants:
            if n.kind == "function":
                self.funcs[n.name] = n
                self.vis[n.name] = True
                self.shared_descendant_funcs.append(n.name)
            else:
                self.shared_descendant_nodes.append(n)
        self.funcs[func.name] = func
        self.vis[func.name] = False

        self.root = Node(
            ancestor_nodes=[n for n in ancestors if n.kind != "function"],
            ancestor_funcs=[n.name for n in ancestors if n.kind == "function"],
            cur=func.name,
            siblings=[],
            children=[],
        )
        self._ancestors = ancestors
        self._func = func
        self._descendants = descendants

    async def run(self) -> tuple[tuple[LrtFunctionDef, list[LrtNode]] | None, CodeGenJournal]:
        self.ctx.log.in_scope(f"funcoder_dfs_1pass[{self._func.name}(...)]")
        _sj = CodeGenJournalist(self.ctx, "funcoder_dfs_1pass", (self._ancestors, self._func, self._descendants))

        root, _sj_ch = await self.dfs(self.root, 0)
        _sj.append(_sj_ch)
        if root is None:
            _err = "failed to generate program"
            self.ctx.log.warn(_err)
            return None, _sj.collect_err(error=_err)

        # verify results
        program = self._subtree_of(root)
        program = self.ctx.lrt.prettify(LrtProgram(module=(), nodes=program))
        if not (ret_func := program.find(LrtFunctionDef, self._func.name)):
            _err = "requested function not found in generated program"
            self.ctx.log.warn(_err)
            return None, _sj.collect_err(error=_err)
        ret_nodes = program.excluding(ret_func)

        return (ret_func, ret_nodes), _sj.collect_gen((ret_func, ret_nodes))

    async def dfs(self, node: Node, depth: int) -> tuple[Node | None, CodeGenJournal | None]:
        if self.vis.get(node.cur, False):
            return node, None
        self.vis[node.cur] = True
        _sj = CodeGenJournalist(
            self.ctx,
            "funcoder_dfs_1pass::dfs",
            (self._ancestors_of(node), self.funcs[node.cur], self._descendants_of(node)),
        )
        if depth >= self.opt_max_depth:
            return None, _sj.collect_err("max depth reached")

        # [[ pass 1 ]]
        fn_original = self.funcs[node.cur]
        fn = fn_original
        vis_1 = {n for n, vis in self.vis.items() if vis}
        ret_1, _sj_1 = await self.gen_pass_1(
            self.ctx, self._ancestors_of(node), fn, node.siblings + self._descendants_of(node)
        )
        _sj.append(_sj_1)
        if ret_1 is None:
            return None, _sj.collect_err("dfs pass 1 failed")
        fn, children = ret_1
        self.funcs[fn.name] = fn

        # [[ iter children ]]
        for child in children:
            if child.kind != "function":
                node.siblings.append(child)
                continue
            if child.name in vis_1:
                continue
            self.funcs[child.name] = child
            ch_node = Node(
                ancestor_nodes=node.ancestor_nodes + node.siblings,
                ancestor_funcs=node.ancestor_funcs + [node.cur],
                cur=child.name,
                siblings=[],
                children=[],
            )
            ch_node, _sj_ch = await self.dfs(ch_node, depth + 1)
            _sj.append(_sj_ch)
            if ch_node is not None:
                node.children.append(ch_node)

        log = _sj.collect_gen((fn, sum([self._subtree_of(ch) for ch in node.children], [])))
        self.funcs[fn.name] = fn
        return node, log

    # same with dfs2p

    def _ancestors_of(self, node: Node) -> list[LrtNode]:
        ret = [self.funcs[n] for n in node.ancestor_funcs] + node.ancestor_nodes
        return self.ctx.lrt._parse.deduplicate_nodes(ret)

    def _descendants_of(self, node: Node) -> list[LrtNode]:
        # all descendants of a node, excluding itself
        def __recurse(p: Node) -> list[LrtNode]:
            ret = []
            for ch in p.children:
                ret.append(self.funcs[ch.cur])
                ret.extend(ch.siblings)
                ret += __recurse(ch)
            return ret

        ret = __recurse(node)
        ret += self.shared_descendant_nodes
        ret += [self.funcs[n] for n in self.shared_descendant_funcs]
        return self.ctx.lrt._parse.deduplicate_nodes(ret)

    def _subtree_of(self, node: Node) -> list[LrtNode]:
        ret: list[LrtNode] = []
        ret.append(self.funcs[node.cur])
        ret.extend(node.siblings)
        for child in node.children:
            ret += self._subtree_of(child)
        return self.ctx.lrt._parse.deduplicate_nodes(ret)

    pass
