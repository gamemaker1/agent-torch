# substeps/utils.py
# utility functions for all substeps

import regex as re
from agent_torch.core.helpers import get_by_path


def read_var(state, var):
    """
    Retrieves a value from the current state of the model.
    """
    return get_by_path(state, re.split("/", var))
