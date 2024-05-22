from __future__ import annotations

import copy
import tempfile
from typing import Any

from langchain.llms.base import BaseLLM
from langchain_community.llms import HuggingFacePipeline
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions
from mergekit.merge import run_merge
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import CmaEsSampler
from optuna.study import Study
from optuna.trial import FrozenTrial
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import pipeline
import yaml

from package.samplers.evo_merge.trial import EvoMergeTrial


class EvoMergeSampler(BaseSampler):
    def __init__(self, base_config: str, seed: None | int = None) -> None:
        self.seed = seed
        self._cmaes = CmaEsSampler()

        with open(base_config, "r", encoding="utf-8") as fp:
            self._merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return param_distribution._asdict()

    def reseed_rng(self, seed: int) -> None:
        self.seed = seed

    def _sample(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return param_distribution._asdict()

    def sample_model(self, study: Study, trial: EvoMergeTrial) -> BaseLLM:
        merge_config = copy.deepcopy(self._merge_config)
        for i, m in enumerate(merge_config.models):
            if i == 0:
                # No parameters necessary for base model.
                continue
            model_name = m.model
            search_space = {f"{model_name}-{k}": FloatDistribution(0, 1) for k in m.parameters}
            params = self._cmaes.sample_relative(study, trial, search_space)
            for name in params:
                param_value = params[name]
                distribution = search_space[name]
                param_value_in_internal_repr = distribution.to_internal_repr(param_value)
                study._storage.set_trial_param(
                    trial._trial_id, name, param_value_in_internal_repr, distribution
                )

            if len(params) == 0:
                continue
            for k in m.parameters:
                m.parameters[k] = params[f"{model_name}-{k}"]

        with tempfile.TemporaryDirectory() as output_path:
            run_merge(
                merge_config,
                out_path=output_path,
                options=MergeOptions(
                    lora_merge_cache="/tmp",
                    cuda=torch.cuda.is_available(),
                    copy_tokenizer=True,
                    lazy_unpickle=False,
                    low_cpu_memory=False,
                ),
            )
            llm = load_model(output_path)

        return llm


def load_model(model_id: str) -> BaseLLM:
    bnbconf = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnbconf)
    llm = HuggingFacePipeline(
        pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=16,
            temperature=0.7,
            do_sample=True,
            return_full_text=False,
        )
    )
    return llm
