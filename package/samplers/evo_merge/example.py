from __future__ import annotations

import argparse
import datetime
import json
import os

from datasets import load_dataset
from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
import optuna

from package.samplers.evo_merge.sampler import EvoMergeSampler
from package.samplers.evo_merge.trial import EvoMergeTrial


# EvoMergeSampler = optunahub.load_module("samplers/evo_merge").EvoMergeSampler
# EvoMergeTrial = optunahub.load_module("samplers/evo_merge").EvoMergeTrial

TEMPLATE = "質問に答えなさい。質問: {question} 回答: "


if not os.path.exists("./simple.jsonl"):
    dataset = load_dataset("SakanaAI/gsm8k-ja-test_250-1319", split="test")
    with open("./simple.jsonl", "w") as fout:
        for q, a in zip(dataset["question"][:100], dataset["answer_number"][:100]):
            fout.write(json.dumps({"question": q, "answer_number": a}) + "\n")

dataset = []
with open("./simple.jsonl") as fin:
    for line in fin:
        dataset.append(json.loads(line.strip()))


def eval_jaqket(llm_chain: LLMChain) -> int:
    correct = 0
    for problem in dataset:
        out = llm_chain.run(question=problem["question"])
        if len(out.strip()) != 0:
            out = out.strip().split()[0].strip()
        if problem["answer_number"] in out:
            correct += 1

    return correct


def try_model(llm: BaseLLM) -> float:
    ptemplate = PromptTemplate.from_template(TEMPLATE)
    llm_chain = LLMChain(prompt=ptemplate, llm=llm)

    correct = eval_jaqket(llm_chain)

    return correct / len(dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", help="Optuna study name")
    args = parser.parse_args()

    if args.study_name:
        study_name = args.study_name
    else:
        study_name = f"optuna-merge-ja-{datetime.datetime.now().isoformat(timespec='minutes')}"

    sampler = EvoMergeSampler(base_config="./config.yml")
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=sampler,
    )

    for _ in range(100):
        trial = study.ask()
        evo_merge_trial = EvoMergeTrial(study, trial._trial_id)
        model = evo_merge_trial.suggest_model()

        acc = try_model(model)

        study.tell(trial, acc)

    print(study.trials_dataframe(attrs=("number", "value")))


if __name__ == "__main__":
    main()
