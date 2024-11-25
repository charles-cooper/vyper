from dataclasses import dataclass,field
from vyper.utils import OrderedSet
from vyper.venom.analysis import CFGAnalysis, IRAnalysis

from vyper.venom.basicblock import IRBasicBlock

@dataclass
class SCC:
    root: IRBasicBlock

    basicblocks: OrderedSet[IRBasicBlock] #= field(default_factory=OrderedSet)

    #next_sccs: OrderedSet["SCC"] #= field(default_factory = OrderedSet)

class SCCAnalysis(IRAnalysis):
    """
    Strongly Connected Components Analysis
    """
    def analyze(self):
        cfg = self.analyses_cache.request_analysis(CFGAnalysis)

        self._sccs = {}
        self._lowlinks = {}
        self._stack = []
        self._dfs_order = 0

        for bb in self.function.get_basic_blocks():
            if bb not in self._lowlinks:
                self._analyze_r(bb)

    def _analyze_r(self, bb: "IRBasicBlock"):
        self._lowlinks[bb] = self._dfs_order
        self._dfs_order += 1
        self._stack.append(bb)

        for next_bb in bb.cfg_out:
            if next_bb not in self._lowlinks:
                self._analyze_r(next_bb)
                self._lowlinks[bb] = min(self._lowlinks[bb], self._lowlinks[next_bb])
            elif next_bb in self._stack:
                self._lowlinks[bb] = min(self._lowlinks[bb], self._lowlinks[next_bb])

        if self._lowlinks[bb] == bb:
            root = bb
            scc = OrderedSet(bb, OrderedSet())
            while True:
                bb = self.stack.pop()
                scc.basicblocks.add(bb)
                self.sccs[bb] = scc
                if bb == root:
                    break
