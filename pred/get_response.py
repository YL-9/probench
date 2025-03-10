import argparse
import copy
import json
import os
import time
import multiprocessing
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import logging

from pred.utils import generate_prompts, save_response
from pred.client import OpenAIClient


client_map = {
    "openai": {
        "client": OpenAIClient,
        "api_key": "your openai api key"
    },
}

error_file_path = "pred/error.txt"


def process_single_problem(example):
    problem, config = example

    close = config["close"]
    need_samples = config["n"]
    existing_count = config["existing"]

    client = client_map[close]["client"](api_key=client_map[close]["api_key"], url=client_map[close].get("url", None))
    problem_id = problem["prompt_path"].split("/")[-1].split(".")[0]
    
    response_texts = []
    sub_responses = []
    sub_problems = []

    try:
        logging.info(f"{problem_id} is processing, need {need_samples}...")
        if close in ["openai", "silicon"]:
            if close == "silicon":
                config["model"] = f"deepseek-ai/{config['model']}"

            try:
                response_texts = client.completion_with_backoff(**config)
            except Exception as e:
                logging.info(f"{problem_id} error: {e}")
        else:
            for i in range(need_samples):
                try:
                    response_text = client.completion_with_backoff(**config)
                    response_texts.append(response_text)
                    logging.info(f"{problem_id} {i+1} is success...")
                except Exception as e:
                    logging.info(f"{problem_id} {i+1} error: {e}")

        error_count = 0
        for i, response_text in enumerate(response_texts):
            if response_text is None or response_text == '':
                error_count += 1
                continue

            sub_responses.append(response_text)
            problem_copy = copy.deepcopy(problem)
            problem_copy["sample_number"] = i + existing_count - error_count
            sub_problems.append(problem_copy)

        logging.info(f"Generated {len(sub_responses)} responses for {problem_id}")
        return sub_responses, sub_problems

    except Exception as e:
        logging.info(f"Error processing {problem_id}: {e}")
        return [], []

def get_from_api(args, prompts, problems):
    logging.info(f"get from {args.model_name} api!")

    exist = {}
    merge_data = []
    if args.merge:
        if os.path.exists(args.merge):
            with open(args.merge, 'r', encoding='utf-8') as f:
                merge_data = json.load(f)

            for problem in merge_data:
                problem_id = problem["prompt_path"].split("/")[-1].split(".")[0]
                if problem_id not in exist:
                    exist[problem_id] = 0
                exist[problem_id] += 1

    tasks = []
    for messages, problem in zip(prompts, problems):
        problem_id = problem["prompt_path"].split("/")[-1].split(".")[0]
        existing_count = exist.get(problem_id, 0)
        samples_needed = max(0, args.samples - existing_count)

        if samples_needed > 0:
            task_config = {
                "close": args.close,
                "model": args.model_name,
                "max_tokens": args.max_tokens,
                "stream": args.stream,
                "existing": existing_count,
                "n": samples_needed,
                "reasoning": args.reasoning,
                "messages": messages,
            }
            tasks.append((problem, task_config))

    logging.info(f"{len(tasks)} tasks to process")

    responses = []
    with multiprocessing.Pool(args.num_procs) as pool:
        for sub_responses, sub_problems in pool.imap_unordered(process_single_problem, tasks):
            if len(sub_responses) > 0:
                merge_data = save_response(
                    sub_responses,
                    sub_problems,
                    args.model_name,
                    error_file_path,
                    merge_data=merge_data,
                    response_path=args.response_path if args.response_path else None
                )
                responses.extend(sub_responses)

                problem_id = sub_problems[0]["prompt_path"].split("/")[-1].split(".")[0]
                logging.info(f"saved {len(sub_responses)} responses of {problem_id}, totle {len(merge_data)} responses.")

    return responses, merge_data

def get_from_vllm(args, prompts, problems):
    logging.info("getting response from vLLM!")

    sampling_params = SamplingParams(
        n=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        stop_token_ids=args.stop_token_ids,
    )

    logging.info("Loading model...")
    llm = LLM(
        model=args.model_path, 
        max_model_len=args.max_tokens, 
        tensor_parallel_size=args.gpu_num,
    )

    logging.info("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)

    tmp_problems = []
    responses = []
    for output, problem in zip(outputs, problems):
        for i, sample in enumerate(output.outputs):
            responses.append(sample.text)
            problem_copy = copy.deepcopy(problem)
            problem_copy["sample_number"] = i
            tmp_problems.append(problem_copy)

    logging.info(f"Generated {len(responses)} responses.")
    logging.info("Saving responses and code...")
    problems = save_response(responses, problems, args.model_name, error_file_path)
    logging.info(f"Generation complete! Results saved to response.json. Totle response: {len(problems)}")

    return responses, tmp_problems

def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.logging),
            # logging.StreamHandler()
        ]
    )

    start_time = int(time.time())
    logging.info(f"start time: {start_time}")
    model_path = args.model_path
    model_name = args.model_name

    files = ["prompt", "response", "code"]
    for file in files:
        os.makedirs(f"data/{model_name}/{file}", exist_ok=True)

    tokenizer = None
    if args.close is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    logging.info("Generating prompts...")
    aipc_path = args.aipc_path
    prompts, problems = generate_prompts(aipc_path, tokenizer, model_name)
    logging.info(f"Generated {len(prompts)} prompts and {len(problems)} problems.")

    if args.close is None:
        responses, problems = get_from_vllm(args, prompts, problems)
    else:
        responses, problems = get_from_api(args, prompts, problems)

    logging.info(f"Finished, generated {len(responses)} responses, total {len(problems)} problems.")
    end_time = int(time.time())
    logging.info(f"end time: {end_time}")
    logging.info(f"total time: {end_time - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data/ly/qwen_qwq")
    parser.add_argument("--model-name", type=str, default="QwQ-32B-Preview")
    parser.add_argument("--probench-path", type=str, default="pred/problem_list.json")
    parser.add_argument("--gpu-num", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=float, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--stop_token_ids", type=int, nargs='+', default=[151645, 151643])
    parser.add_argument("--close", type=str, choices=["openai"])
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--sleep", type=int, default=1)
    parser.add_argument("--merge", type=str)
    parser.add_argument("--num-procs", type=int, default=8)
    parser.add_argument("--logging", type=str, default="output.log")
    parser.add_argument("--response-path", type=str)
    args = parser.parse_args()
    main(args)