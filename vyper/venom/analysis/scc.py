from vyper.utils import OrderedSet
from vyper.venom.analysis import CFGAnalysis, IRAnalysis
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vyper.venom.basic_block import IRBasicBlock

class SCCAnalysis(IRAnalysis):
    """
    Strongly Connected Components Analysis
    """
    def analyze(self):
        cfg = self.request_analysis(CFGAnalysis)

        self._sccs = {}

        self._stack = OrderedSet()

        self._analyze_r(self.function.entry)

    def _analyze_r(self, bb: "IRBasicBlock"):
        if bb in self._sccs:
            return

        if bb not in self._stack:
            self._stack.add(bb)
            for next_bb in bb.cfg_out:
                self._analyze_r(next_bb)
        else:
            # found a cycle.
            root = bb
            while True:
                bb = self._stack.pop()
                self._sccs[bb] = root
                if bb == root:
                    break
