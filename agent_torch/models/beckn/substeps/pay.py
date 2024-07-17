# substeps/pay.py
# `pay` operation

import torch
import torch.nn as nn
from agent_torch.core.registry import Registry
from agent_torch.core.substep import (
    SubstepObservation,
    SubstepAction,
    SubstepTransition,
)
from .utils import read_var


@Registry.register_substep("get_order_details", "observation")
class GetOrderDetails(SubstepObservation):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state):
        order_bap_id = read_var(state, self.input_variables["order_bap_id"])
        order_bpp_id = read_var(state, self.input_variables["order_bpp_id"])
        order_status = read_var(state, self.input_variables["order_status"])

        update_mask = order_status == 3
        involved_baps = order_bap_id[update_mask].int()
        involved_bpps = order_bpp_id[update_mask].int()

        return {
            "involved_baps": involved_baps,
            "involved_bpps": involved_bpps,
            "orders_to_update": update_mask,
        }


@Registry.register_substep("calculate_payment", "policy")
class CalculatePayment(SubstepAction):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state, observation):
        bpp_price = read_var(state, self.input_variables["bpp_price"])
        order_quantity = read_var(state, self.input_variables["order_quantity"])
        involved_baps, involved_bpps, update_mask = (
            observation["involved_baps"],
            observation["involved_bpps"],
            observation["orders_to_update"],
        )

        quantity_ordered = order_quantity[update_mask]
        payments_to_make = torch.zeros(involved_baps.shape)

        for i, bpp_id in enumerate(involved_bpps):
            payments_to_make[i] = bpp_price[bpp_id] * quantity_ordered[i]

        return {
            "payments_to_make": payments_to_make,
            "involved_baps": involved_baps,
            "involved_bpps": involved_bpps,
        }


@Registry.register_substep("update_wallets", "transition")
class UpdateWallets(SubstepTransition):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state, action):
        bap_wallet = read_var(state, self.input_variables["bap_wallet"])
        involved_baps, payments_to_make = (
            action["bap"]["involved_baps"],
            action["bap"]["payments_to_make"],
        )

        # bap_id is same as index
        for i, bap_id in enumerate(involved_baps):
            bap_wallet[bap_id] -= payments_to_make[i]

        return {
            "bap_wallet": bap_wallet,
        }
