from pipeline.feature_engineering.preprocessing.replacement_strategies.abstract_replacement_strategy import ReplacementStrategy
from overrides import overrides
import traceback
import os
import pandas


class ReplacementValReplacementStrategy(ReplacementStrategy):

    def __init__(self):
        super().__init__()

    @overrides
    def replace(self, data, mode, target_vals = None, replacement_vals=None):
        try:
            if data is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not isinstance(data, pandas.DataFrame):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)

            if mode == 'NaN':
                return data.fillna(replacement_vals)

            raise ValueError(self.messages.PROVIDED_MODE_DOESNT_EXIST.value)

        except (FileNotFoundError, ValueError, TypeError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)