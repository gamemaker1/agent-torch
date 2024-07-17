# substeps/order.py
# `order` confirmation

import torch
import torch.nn as nn
from agent_torch.core.registry import Registry
from agent_torch.core.substep import (
    SubstepObservation,
    SubstepAction,
    SubstepTransition,
)
from .utils import read_var


@Registry.register_substep("check_availability", "observation")
class CheckAvailability(SubstepObservation):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state):
        bpp_capacity = read_var(state, self.input_variables["bpp_capacity"])
        is_available = bpp_capacity > 0

        return {"is_available": is_available}


@Registry.register_substep("confirm_order", "policy")
class ConfirmOrder(SubstepAction):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state, observation):
        is_available = observation["is_available"]

        confirmation = torch.where(
            is_available,
            torch.ones_like(is_available, dtype=torch.int),
            torch.zeros_like(is_available, dtype=torch.int),
        )

        return {"order_confirmation": confirmation}


@Registry.register_substep("update_order_status", "transition")
class UpdateOrderStatus(SubstepTransition):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state, action):
        bpp_capacity = read_var(state, self.input_variables["bpp_capacity"])
        order_bpp_id = read_var(state, self.input_variables["order_bpp_id"])
        order_status = read_var(state, self.input_variables["order_status"])
        confirmation = action["bpp"]["order_confirmation"]

        orders_to_update = order_status == 0
        selected_bpps = order_bpp_id[orders_to_update].int()
        confirmation_for_bpps = confirmation[selected_bpps] == 1

        i = 0
        for j, should_update in enumerate(orders_to_update):
            if not should_update:
                continue

            if confirmation_for_bpps[i] and bpp_capacity[selected_bpps[i]] > 0:
                order_status[j] = 1
                bpp_capacity[selected_bpps[i]] -= 1
            else:
                order_status[j] = 4
            i += 1

        return {"order_status": order_status, "bpp_capacity": bpp_capacity}
