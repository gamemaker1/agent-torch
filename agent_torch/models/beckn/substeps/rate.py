# substeps/rate.py
# `rate` operation

import torch
import torch.nn as nn
from agent_torch.core.registry import Registry
from agent_torch.core.substep import (
    SubstepObservation,
    SubstepAction,
    SubstepTransition,
)
from .utils import read_var


@Registry.register_substep("get_service_experience", "observation")
class GetServiceExperience(SubstepObservation):
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
        }


@Registry.register_substep("calculate_rating", "policy")
class CalculateRating(SubstepAction):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state, observation):
        bpp_rating = read_var(state, self.input_variables["bpp_rating"])
        bpp_num_ratings = read_var(state, self.input_variables["bpp_num_ratings"])
        involved_baps, involved_bpps = (
            observation["involved_baps"],
            observation["involved_bpps"],
        )

        ratings_given = 5 * torch.rand(involved_bpps.shape)
        num_ratings_given = torch.zeros(involved_bpps.shape)
        for i, bpp_id in enumerate(involved_bpps):
            ratings_given[i] = (
                (bpp_rating[bpp_id] * bpp_num_ratings[bpp_id]) + ratings_given[i]
            ) / (bpp_num_ratings[bpp_id] + 1)
            num_ratings_given[i] += 1

        return {
            "ratings_given": ratings_given,
            "num_ratings": num_ratings_given,
            "involved_baps": involved_baps,
            "involved_bpps": involved_bpps,
        }


@Registry.register_substep("update_bpp_rating", "transition")
class UpdateBPPRating(SubstepTransition):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state, action):
        bpp_rating = read_var(state, self.input_variables["bpp_rating"])
        bpp_num_ratings = read_var(state, self.input_variables["bpp_num_ratings"])
        involved_bpps, ratings_given, num_ratings = (
            action["bap"]["involved_bpps"],
            action["bap"]["ratings_given"],
            action["bap"]["num_ratings"],
        )

        # bpp_id is same as index
        for i, bpp_id in enumerate(involved_bpps):
            bpp_rating[bpp_id] = ratings_given[i]
            bpp_num_ratings[bpp_id] += num_ratings[i]

        return {"bpp_rating": bpp_rating, "bpp_num_ratings": bpp_num_ratings}
