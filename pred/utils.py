import re
import copy
import json


system_message_zh = {
    "role": "system",
    "content": "你是一位经验丰富的程序设计竞赛选手，擅长分析复杂问题并设计高效的解决方案。你的任务是解决以下编程题目，请始终保持清晰的逻辑并逐步推导解题过程。"
}

system_message_en = {
    "role": "system",
    "content": "You are a highly skilled competitive programming expert, adept at analyzing complex problems and designing efficient solutions. Your task is to solve the following programming challenge. Always maintain clear logic and think step by step."
}

template_zh = '''
你的任务是仔细阅读以下题目描述，逐步分析问题并清晰地阐述你的思考过程。最后，请使用 {lang} 语言编写解题代码，并确保代码的正确性和可读性。将代码用以下格式包裹： 

```{lang_type}
{lang} 解题代码
```

接下来是题目描述：

{description}
'''

template_en = '''
Your task is to carefully read the following problem description, analyze the problem step by step, and clearly explain your thought process. Finally, write the solution in {lang} and ensure the code is correct and readable. Wrap your code in the following format:

```{lang_type}
{lang} solution code
```

Here is the problem description:

{description}
'''

langs = {
    'C++': 'cpp',
    # 'Java': 'java',
    # 'Python': 'python'
}
lang_file = {
    'C++': 'cpp',
    'Java': 'java',
    'Python': 'py'
}

def generate_prompts(probench_path, tokenizer, model_name):
    with open(probench_path, 'r', encoding='utf-8') as f:
        aipc_data = json.load(f)

    prompts = []
    problems = []
    for i, problem in enumerate(aipc_data):
        description_path = problem['description_path']
        with open(description_path, 'r', encoding='utf-8') as f:
            description = f.read()

        for lang, lang_type in langs.items():
            if problem['language'] == 'zh':
                prompt = template_zh.format(lang=lang, lang_type=lang_type, description=description)
                messages = [
                    system_message_zh,
                    {"role": "user", "content": prompt}
                ]
            else:
                prompt = template_en.format(lang=lang, lang_type=lang_type, description=description)
                messages = [
                    system_message_en,
                    {"role": "user", "content": prompt}
                ]

            if tokenizer is not None:
                messages = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

            prompts.append(messages)
            problem_copy = copy.deepcopy(problem)
            problem_copy['code_lang'] = lang
            problems.append(problem_copy)

    for i, (prompt, problem) in enumerate(zip(prompts, problems)):
        problem_id = problem["description_path"].split("/")[-1].split(".")[0]
        prompt_path = f"data/{model_name}/prompt/{problem_id}{problem['code_lang']}.md"
        with open(prompt_path, "w", encoding='utf-8') as f:
            f.write(str(prompt))
        problems[i]["prompt_path"] = prompt_path

    return prompts, problems

def save_response(responses, problems, model_name, error_file_path, merge_data=None, response_path=None):
    for i, (response, problem) in enumerate(zip(responses, problems)):
        problem_id = problem["description_path"].split("/")[-1].split(".")[0]
        response_file = f"data/{model_name}/response/{problem_id}{problem['code_lang']}_{problem['sample_number']}.md"
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write(response)
        problems[i]['response_path'] = response_file

        pattern = rf'```{langs[problem["code_lang"]]}.*?\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            code = matches[-1]
            code_file = f"data/{model_name}/code/{problem_id}{problem['code_lang']}_{problem['sample_number']}.{lang_file[problems[i]['code_lang']]}"
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            problems[i]['code_path'] = code_file
        else:
            print(f"No code found in output for response {problem['response_path']}.")
            with open(error_file_path, 'a', encoding='utf-8') as f:
                f.write(f"No code found in output for response {problem['response_path']}.\n")
            problems[i]['code_path'] = ""

    if merge_data is not None:
        merge_data.extend(problems)
        problems = merge_data

    if response_path is None:
        response_path = f"data/{model_name}/response.json"
    with open(response_path, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=4, ensure_ascii=False)

    return problems
