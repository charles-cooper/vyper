from __future__ import annotations

from vyper.venom.passes.base_pass import IRPass
from vyper.venom.passes.internal_return_copy_forwarding import InternalReturnCopyForwardingPass
from vyper.venom.passes.readonly_invoke_arg_copy_forwarding import (
    ReadonlyInvokeArgCopyForwardingPass,
)


class InvokeArgCopyForwardingPass(IRPass):
    """
    Compatibility wrapper that runs both invoke-related copy forwarding passes.

    Kept as the public pipeline entrypoint while logic lives in:
      - InternalReturnCopyForwardingPass
      - ReadonlyInvokeArgCopyForwardingPass
    """

    def run_pass(self):
        InternalReturnCopyForwardingPass(self.analyses_cache, self.function).run_pass()
        ReadonlyInvokeArgCopyForwardingPass(self.analyses_cache, self.function).run_pass()
