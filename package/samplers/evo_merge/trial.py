from langchain.llms.base import BaseLLM
from optuna.study import Study
from optuna.trial import Trial


class EvoMergeTrial(Trial):
    def __init__(self, study: Study, trial_id: int) -> None:
        super(EvoMergeTrial, self).__init__(study, trial_id)
        self._trial_id = trial_id

    def suggest_model(self) -> BaseLLM:
        return self.study.sampler.sample_model(self.study, self)
