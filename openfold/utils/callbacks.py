from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class EarlyStoppingVerbose(EarlyStopping):

    def _evalute_stopping_criteria(self, *args):
        should_stop, reason = super()._evalute_stopping_criteria(*args)
        if(should_stop):
            rank_zero_info(f"{reason}\n")

        return should_stop, reason
