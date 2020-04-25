# note:
from __future__ import absolute_import
from .base_input_feed import *
from .click_simulation_feed import *
from .direct_label_feed import *
from .clicksimulation_with_type import *
from .DirectLabelFeed_with_clicks import *

from .deterministic_online_simulation_feed import *
from .stochastic_online_simulation_feed import *
from .interleaving_deterministic_online_simulation_feed import *
from .clicksimulation_with_type import *

def list_available() -> list:
    from .base_input_feed import BaseInputFeed
    from ultra.utils.sys_tools import list_recursive_concrete_subclasses
    return list_recursive_concrete_subclasses(BaseInputFeed)

