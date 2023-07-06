import pandas as pd
import numpy as np 
import torch
import jax

from AgentTorch import Runner, Registry

def create_registry():
    reg = Registry()

    from substep.purchase_product.transition import NewQExp, NewPurchasedBefore
    reg.register(NewQExp, "new_Q_exp", key="transition")
    reg.register(NewPurchasedBefore, "new_purchased_before", key="transition")

    from substep.purchase_product.action import PurchaseProduct
    reg.register(PurchaseProduct, "purchase_product", key="policy")

    from substep.purchase_product.observation import GetFromState, GetNeighborsSum, GetNeighborsSumReduced
    reg.register(GetFromState, "get_from_state", key="observation")
    reg.register(GetNeighborsSum, "get_neighbors_sum", key="observation")
    reg.register(GetNeighborsSumReduced, "get_neighbors_sum_reduced", key="observation")

    from AgentTorch.helpers import zeros, random_normal, constant, grid_network
    reg.register(zeros, "zeros", key="initialization")
    reg.register(random_normal, "random_normal", key="initialization")
    reg.register(constant, "constant", key="initialization")
    reg.register(grid_network, "grid", key="network")

    from substep.utils import random_normal_col_by_col
    reg.register(random_normal_col_by_col, "random_normal_col_by_col", key="initialization")


class OpDynRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward():
        for episode in range(self.config['simulation_metadata']['num_episodes']):
            num_steps_per_episode = self.config["simulation_metadata"]["num_steps_per_episode"]
            self.step(num_steps_per_episode)

            self.controller.learn_after_episode(jax.tree_map(lambda x: x[-1], self.trajectory), self.initializer, self.optimizer)


if __name__ == '__main__':
    print("The runner file..")
    args = parser.parse_args()
    
    config_file = args.config

    # create runner object
    runner = OpDynRunner(config_file)    
    runner.execute()
    
    for name, param in runner.named_parameters(): 
        print(name, param.data) 