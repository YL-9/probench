# Probench

 The repository for the paper [ProBench: Benchmarking Large Language Models in Competitive Programming](https://arxiv.org/abs/2502.20868)

## Introduction

Probench collects competition problems from [codeforces](https://codeforces.com/), [luogu](https://www.luogu.com.cn/), and [nowcoder](https://ac.nowcoder.com/) to evaluate models' code reasoning capabilities in competitive programming. It ensures code robustness through online code evaluation, while also providing comprehensive analysis of models' code reasoning abilities.

## Usage

**Data**. We have provided all problem descriptions in the `codeforce`, `luogu`, and `nowcoder` folders, and the statistical information for these problems is displayed in `pred/problem_list.json`.

**Get Responses**. First, you need to use your model to generate responses and solution code based on the provided problems. We offer code for generation using vLLM and APIs (e.g., OpenAI) in `pred/get_response.py`. If you have alternative code, you can refer to `generate_prompts` and `save_response` in `pred/utils.py` to ensure a unified output format.

**Code Evaluation**. Due to platform restrictions, we cannot publicly share submission scripts. You can send the model-generated data from `data/model` to [yl.shadow.yl@gmail.com](https://mailto:yl.shadow.yl@gmail.com/) and contact us. We will return the evaluation results of the code as soon as possible.

## Leaderboard

| Rank | Model                        | Size    | Reasoning | Pass@1 |
| ---- | ---------------------------- | ------- | --------- | ------ |
| 1    | QwQ-32B-Preview              | 32B     | 1         | 20.93  |
| 2    | DeepSeek-V3                  | 37/671B | 0         | 16.38  |
| 3    | Qwen2.5-72B-Instruct         | 72B     | 0         | 11.50  |
| 4    | Mistral-Large-Instruct-2411  | 123B    | 0         | 10.54  |
| 5    | Qwen2.5-Coder-32B-Instruct   | 32B     | 0         | 9.48   |
| 6    | Llama-3.1-70B-Instruct       | 70B     | 0         | 7.99   |
| 7    | Codestral-22B-v0.1           | 22B     | 0         | 5.08   |
| 8    | Skywork-o1-Open-Llama-3.1-8B | 8B      | 1         | 5.06   |
| 9    | Mixtral-8x22B-Instruct-v0.1  | 22/176B | 0         | 4.27   |

## Citation

```
@article{yang2025probench,
  title={ProBench: Benchmarking Large Language Models in Competitive Programming},
  author={Yang, Lei and Jin, Renren and Shi, Ling and Peng, Jianxiang and Chen, Yue and Xiong, Deyi},
  journal={arXiv preprint arXiv:2502.20868},
  year={2025}
}
```

