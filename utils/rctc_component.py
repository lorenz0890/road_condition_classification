import logging
from utils.messages import Messages

class RCTCComponent():

    def __init__(self):
        """
        SCommon funcitonalities
        Sources:
            https://docs.python.org/3/howto/logging-cookbook.html
        """
        #Logging
        self.logger = logging.getLogger('rctc_logger')
        self.logger.setLevel(logging.DEBUG)
        self.fh = logging.FileHandler('./rctc.log') #TODO: Pack in config/.env
        self.fh.setLevel(logging.DEBUG)
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.ERROR)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') #TODO: Pack in config/.env
        self.fh.setFormatter(self.formatter)
        self.ch.setFormatter(self.formatter)
        self.logger .addHandler(self.fh)
        self.logger .addHandler(self.ch)

        #Messages
        self.messages = Messages