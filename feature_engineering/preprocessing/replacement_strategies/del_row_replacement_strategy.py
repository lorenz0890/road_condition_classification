from feature_engineering.preprocessing.replacement_strategies.abstract_replacement_strategy import ReplacementStrategy
from overrides import overrides
import traceback
import os

class DelRowReplacementStrategy(ReplacementStrategy):

    def __init__(self):
        super().__init__()

    @overrides
    def replace(self, data, target, target_vals = None, replacement_vals=None):
        try:
            if target == 'NaN':
                return data.dropna()
            if target == 'unwanted_labels':
                for key in target_vals.keys():
                    for val in target_vals [key]:
                        data = data[data[key]!=val]
                return data

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)