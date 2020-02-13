from abc import ABC, abstractmethod
from utils.rctc_component import RCTCComponent


class PipelineFacade(ABC, RCTCComponent):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def execute_training(self, config):
        """
        Run training based on config
        :param config: dict
        """
        pass

    @abstractmethod
    def execute_inference(self, config):
        """
        Run inference based on config
        :param config: dict
        """
        pass

    @abstractmethod
    def init_pipeline(self, config):
        """
        Initialize pipline based on config
        :param config: dict
        """
        pass
