# substeps/select.py
# `select` operation

import torch
import torch.nn as nn
from agent_torch.core.registry import Registry
from agent_torch.core.substep import (
    SubstepObservation,
    SubstepAction,
    SubstepTransition,
)
from .utils import read_var


@Registry.register_substep("find_nearby_bpps", "observation")
class FindNearbyBPPs(SubstepObservation):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        self.max_distance = arguments.get("max_distance", 100.0)

    def forward(self, state):
        bap_positions = read_var(state, self.input_variables["bap_positions"])
        bpp_positions = read_var(state, self.input_variables["bpp_positions"])

        distances = torch.cdist(bap_positions, bpp_positions)
        nearby_mask = distances <= self.max_distance

        return {"distances": distances, "mask": nearby_mask}


@Registry.register_substep("select_bpp", "policy")
class SelectBPP(SubstepAction):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        self.price_weight = arguments.get("price_weight", 0.3)
        self.distance_weight = arguments.get("distance_weight", 0.3)
        self.capacity_weight = arguments.get("capacity_weight", 0.4)

    def forward(self, state, observation):
        bpp_prices = read_var(state, self.input_variables["bpp_prices"])
        bpp_capacity = read_var(state, self.input_variables["bpp_capacity"])
        resource_level = read_var(state, self.input_variables["resource_level"])

        distances = observation["distances"]
        mask = observation["mask"]

        # Normalize distances and prices
        max_distance = torch.max(distances)
        normalized_distances = distances / max_distance
        max_price = torch.max(bpp_prices)
        normalized_prices = bpp_prices / max_price
        max_capacity = torch.max(bpp_capacity)
        normalized_capacity = bpp_capacity / max_capacity

        # Calculate scores (lower is better)
        scores = (
            self.price_weight
            * torch.flatten(normalized_prices.T)
            .unsqueeze(0)
            .repeat(normalized_distances.shape[0], 1)
            + self.distance_weight * normalized_distances
            + self.capacity_weight
            * torch.flatten(normalized_capacity.T)
            .unsqueeze(0)
            .repeat(normalized_distances.shape[0], 1)
        )

        # Set scores for BPPs that are too far to a large value
        scores[~mask] = float("inf")

        # Select the BPP with the lowest score for each BAP
        selected_bpp_indices = torch.argmin(scores, dim=1)
        selected_bpp_indices = selected_bpp_indices[selected_bpp_indices.nonzero()]

        return {"selected_bpps": selected_bpp_indices}


@Registry.register_substep("create_order", "transition")
class CreateOrder(SubstepTransition):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        self.max_quantity = arguments.get("max_quantity", 100.0)

    def forward(self, state, action):
        selected_bpps = action["bap"]["selected_bpps"]
        bap_ids = read_var(state, self.input_variables["bap_id"])
        resource_level = read_var(state, self.input_variables["resource_level"])
        order_bap_id = read_var(state, self.input_variables["order_bap_id"])
        order_bpp_id = read_var(state, self.input_variables["order_bpp_id"])
        order_quantity = read_var(state, self.input_variables["order_quantity"])
        order_status = read_var(state, self.input_variables["order_status"])
        last_order_id = read_var(state, self.input_variables["last_order_id"])

        quantity_needed = self.max_quantity - resource_level

        next_order_id = int((last_order_id + 1).item())
        for i, (bap_id, bpp_id, quantity) in enumerate(
            zip(bap_ids, selected_bpps, quantity_needed)
        ):
            order_bap_id[next_order_id + i] = bap_id
            order_bpp_id[next_order_id + i] = bpp_id
            order_quantity[next_order_id + i] = quantity
            order_status[next_order_id + i] = 0

        return {
            "order_bap_id": order_bap_id,
            "order_bpp_id": order_bpp_id,
            "order_quantity": order_quantity,
            "order_status": order_status,
        }
