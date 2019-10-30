from feature_engineering.preprocessing.replacement_strategies.abstract_replacement_strategy import ReplacementStrategy
from overrides import overrides
import traceback
import os

class MeanReplacementStrategy(ReplacementStrategy):

    def __init__(self):
        super().__init__()

    @overrides
    def replace(self, data, target, target_vals = None, replacement_vals=None):
        try:
            if target == 'NaN':
                return data.fillna(data.mean())
        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)