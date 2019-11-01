from abc import ABC, abstractmethod
from utils.rctc_component import RCTCComponent


class ReplacementStrategy(ABC, RCTCComponent):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def replace(self, data, mode, target_vals = None, replacement_vals=None):
        pass

