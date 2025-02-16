from utils import *
import torch
from transformers import AutoTokenizer, pipeline
from transformers import LlamaForCausalLM
from transformers import GPT2Tokenizer
import torch
import openai
import re
import time
from openai import OpenAI
from collections import defaultdict
import json
import requests
import difflib
import tokenize
from io import BytesIO
from tenacity import retry, stop_after_attempt, wait_random_exponential

import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, GenerationConfig
from transformers import LlamaForCausalLM, LlamaTokenizer, CodeLlamaTokenizer

import hashlib

SERVER_BASE_URL = "http://localhost:12358"
TOKEN = "314159"
TOKEN = hashlib.sha256(TOKEN.encode("utf-8")).hexdigest()

# for llama models
# B_INST, E_INST = "[INST]", "[/INST]"
# B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
B_SYS, E_SYS = "<|system|>", "</s>"
B_INST, E_INST = "<|prompter|>", "</s><|assistant|>"

key = 'sk-MT7gdbB9BCIfBPwT8f9eB0A2FdEc4215811a83Ac5fA9F2C0'


def format_tokens(dialogs, tokenizer):
    prompt_tokens = []
    for dialog in dialogs:
        if dialog[0]["role"] == "system":
            dialog = [
                         {
                             "role": dialog[1]["role"],
                             "content": B_SYS
                                        + dialog[0]["content"]
                                        + E_SYS
                                        + dialog[1]["content"],
                         }
                     ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system','user' and 'assistant' roles, "
            "starting with user and alternating (u/a/u/a/u...)"
        )

        dialog_tokens: List[int] = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                ) + [tokenizer.eos_token_id]
                for prompt, answer in zip(dialog[::2], dialog[1::2])
            ],
            [],
        )
        assert (
                dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )
        prompt_tokens.append(dialog_tokens)
    return prompt_tokens


def extract_code(response):
    code_blocks = re.findall(r'```c\+\+(.+?)```', response, re.DOTALL)
    if len(code_blocks) > 0:
        code = '\n'.join(code_blocks)
        return code
    else:
        return None


def check_json_format(str):
    try:
        json.loads(str)
        return str
    except json.JSONDecodeError as e:
        print('1 ', e)
        pass
    fixed_string = str
    fixed_string = fixed_string.replace("，", ",")
    fixed_string = fixed_string.replace("，", ",")
    escaped_string = fixed_string.replace('\\', '\\\\').replace('\n', '\\n').replace('\t', '\\t').replace('"', '\\"')
    try:
        json.loads(escaped_string)
        return escaped_string
    except json.JSONDecodeError as e:
        print(e, escaped_string)
        exit()

        return fixed_string


@retry(wait=wait_random_exponential(multiplier=1, max=10), stop=stop_after_attempt(5))
def load_model_tokenizer(model_name, device='cuda:3'):
    if model_name == 'qwen2-7b':
        model_path = '/home/CodeLLM/Qwen/Qwen2___5-Coder-7B-Instruct'
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device)
        return tokenizer, model

    elif model_name == 'DeepSeek-1.3b-it':
        model_path = '/home/llm_model/deepseek-coder/deepseek-coder-1.3b-instruct'
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device)
        return tokenizer, model

    elif model_name == 'DeepSeek-7b-it':
        model_path = '/home/llm_model/deepseek-coder/deepseek-coder-7b-instruct-v1.5'
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device)
        return tokenizer, model

    else:
        return None, None


def generate_response_api(prompt, tokenizer, model, model_name, system_message=None, device='cuda:3'):
    sys_msg = "You are a helpful coding teaching assistant."
    if system_message:
        sys_msg = system_message

    if model_name == 'gpt-4o-mini':
        client = openai.OpenAI(
            api_key="Bu6ORgxIqmn1a",
            base_url='https://ai.liaobots.work/v1'
        )

        message = recursive_call_openai('gpt-4o-mini', client, sys_msg, prompt)
        return message.strip()

    elif model_name == 'DeepSeekCoder-v2':
        client = openai.OpenAI(
            api_key="token-abc123",
            base_url='http://localhost:12343/v1'
        )

        message = recursive_call_openai('/home/CodeLLM/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct', client, sys_msg,
                                        prompt)
        return message.strip()

    elif model_name == 'DeepSeek-33b-it':
        url = "http://localhost:8000/v1/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": "/home/llm_model/deepseek-coder/deepseek-coder-33b-instruct",
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(response.status_code)
            print(response.text)

    elif 'DeepSeek' in model_name:
        messages = [
            {'role': 'user', 'content': sys_msg},
            {'role': 'user', 'content': prompt}
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
            model.device)
        outputs = model.generate(
            inputs,
            max_new_tokens=1024,
            do_sample=False,
            top_k=1,
            # top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return response

def recursive_call_openai(model_name, client, system_prompt, prompt):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        time.sleep(2)
        return response.choices[0].message.content.strip()
    except openai.BadRequestError:
        print(prompt)
        print('BadRequestError, re-trying')
        time.sleep(10)
        return recursive_call_openai(model_name, client, system_prompt, prompt)
    except openai.AuthenticationError:
        print('AuthenticationError, re-trying')
        time.sleep(10)
        return recursive_call_openai(model_name, client, system_prompt, prompt)
    except openai.InternalServerError:
        print('InternalServerError, re-trying')
        time.sleep(10)
        return recursive_call_openai(model_name, client, system_prompt, prompt)
    except:
        print('OtherError, re-trying')
        time.sleep(10)
        return recursive_call_openai(model_name, client, system_prompt, prompt)


def getTextDB(question_id):
    question = "这是一道编程题，请你用C++实现。高精度加法，相当于a+b problem，不用考虑负数。分两行输入。a, b ≤ 10^10000.输出只有一行，代表a+b的值。\n样例输入：1001 9999\n 样例输出：10100。"
    answer = "#include <bits/stdc++.h>\n#define N 11111\nusing namespace std;\n\nint a_digits[N], b_digits[N];\nint a_len, b_len;\nint ans_digits[N], ans_len;\nchar str1[N], str2[N];\nint main() {\n    cin >> str1;\n    // 获取高精度整数长度\n    int a_len = strlen(str1);\t\n    for (int i = 0; i < a_len; ++i)\n        // TODO 请补全下述代码\n        a_digits[i] = str1[a_len - i - 1] - '0'; // 将字符转换成数字，倒着存进数位数组\n    \n    cin >> str2;\n    // 获取高精度整数长度\n    int b_len = strlen(str2);\t\n    for (int i = 0; i < b_len; ++i)\n        // TODO 请补全下述代码\n        b_digits[i] = str2[b_len - i - 1] - '0'; // 将字符转换成数字，倒着存进数位数组\n    \n    ans_len = max(a_len, b_len); \t// 初始长度\n    int k = 0;\t\t\t\t\t\t// 记录进位的变量\n    for (int i = 0; i < ans_len; ++i) {\n        // 假设a_len > b_len，这里需要保证b[b_len]到b[a_len - 1]的位置都是0，否则可能会出错。\n        ans_digits[i] = a_digits[i] + b_digits[i] + k; // 相加计算\n        k = ans_digits[i] / 10;     // 更新进位\n        ans_digits[i] %= 10;\n    }\n    \n    \n    if (k) \n        ans_digits[ans_len++] = k;\t// 最高位进位\n    \n    // 3. 输出\n    // 按照打印顺序输出，从高位到低位。\n    \n    for (int i = ans_len - 1; i >= 0; --i) \n        cout << ans_digits[i];\n    cout << endl;\n    \n    return 0;\n}"
    return question, answer


def check_result(result):
    ac_num = 0
    total_num = 0
    if result['err'] is None:
        for testcase in result['data']:
            total_num += 1
            if testcase['result'] == 0:
                ac_num += 1
        return ac_num / total_num, 0

    else:
        return 0, 1  # compiler error


def check_OJ_result(result):
    if result['status'] == 'accepted':
        return 1, 0
    elif result['status'] == "compile_error":
        return result['score'] / 100, 1  # compiler error
    else:
        return result['score'] / 100, 0


def OJTest(code, question_id):
    from local_oj_test import judge
    import numpy as np
    id_uuid_dict = np.load('/home/code_generate_new/id_uuid_dict.npy', allow_pickle=True)
    default_env = ["LANG=en_US.UTF-8", "LANGUAGE=en_US:en", "LC_ALL=en_US.UTF-8"]

    cpp_lang_config = {
        "compile": {
            "src_name": "main.cpp",
            "exe_name": "main",
            "max_cpu_time": 10000,
            "max_real_time": 10000,
            "max_memory": 256 * 1024 * 1024,
            "compile_command": "/usr/bin/g++ -DONLINE_JUDGE -O2 -w -fmax-errors=3 -std=c++14 {src_path} -lm -o {exe_path}",
        },
        "run": {"command": "{exe_path}", "seccomp_rule": "c_cpp", "env": default_env},
    }
    if str(question_id) not in id_uuid_dict[0]:
        return None
    uuid = id_uuid_dict[0][str(question_id)]
    result = judge(
        src=code,
        language_config=cpp_lang_config,
        max_cpu_time=10000,
        max_memory=1024 * 1024 * 256,
        test_case_id=uuid
    )
    return result


def _OJ_get_status(submission_id):
    url = f'https://acm.sjtu.edu.cn/OnlineJudge/api/v1/submission/{submission_id}'

    headers = {
        'accept': 'application/json',
        'Authorization': 'Bearer acmoj-8a6eb0fd81fb939efde1fd7813edd20c',
    }

    response = requests.get(url.format(submission_id=submission_id), headers=headers)

    if response.status_code == 200:
        submission_status = response.json()
        return submission_status
    else:
        print(response.status_code, response.text)
        return None


import requests
import ssl
from requests.adapters import HTTPAdapter


class SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.set_ciphers('ALL:@SECLEVEL=1')
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)


session = requests.Session()
adapter = SSLAdapter()
session.mount('https://', adapter)


def _OJ_submit(problem_id, code):
    # print(problem_id, code)
    # print(isinstance(code, str))
    base_url = "https://acm.sjtu.edu.cn/OnlineJudge/api/v1"
    submit_url = f'''{base_url}/problem/{problem_id}/submit'''
    access_token = "acmoj-8a6eb0fd81fb939efde1fd7813edd20c"
    headers = {
        'accept': 'application/json',
        'Authorization': 'Bearer acmoj-8a6eb0fd81fb939efde1fd7813edd20c',
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    data = {
        'public': 'false',
        'language': 'cpp',
        'code': code
    }
    response = session.post(submit_url, headers=headers, data=data)
    cnt = 0
    while True:
        if response.status_code == 201:
            response_info = response.text.replace("\'", "\"")
            return json.loads(response_info)

        else:
            cnt += 1
            if cnt >= 10:
                break
            time.sleep(2)
            response = requests.post(submit_url, headers=headers, data=data)
    return None


def OJ_judge(code, question_id):
    match = re.search(r'```cpp(.*?)```', code, re.DOTALL)
    if match:
        code = match.group(1)

    # print('OJ_judging {}.......'.format(question_id))
    response = _OJ_submit(question_id, code)
    # print('OJ_submitted {}'.format(question_id))
    if response == None:
        return None
    submission_id = response['id']

    while True:
        submission_status = _OJ_get_status(submission_id)
        # print(submission_status)
        if not (submission_status['status'] == 'pending' or submission_status['status'] == "compiling" or
                submission_status['status'] == "judging"):
            break

        time.sleep(2)

    message = None
    if submission_status['status'] == "compile_error":
        for group in submission_status['details']['groups']:
            for testpoint in group['testpoints']:
                if testpoint['result'] == 'compile_error':
                    message = testpoint['message']
                    break
        # print(message)

    res = {
        'score': submission_status['score'],
        'status': submission_status['status'],
        'message': message
    }
    # print(f"submission_status: {submission_status}")
    return res


def request(url, data=None):
    kwargs = {"headers": {"X-Judge-Server-Token": TOKEN, "Content-Type": "application/json"}}
    if data:
        kwargs["data"] = json.dumps(data)
    try:
        return requests.post(url, **kwargs).json()
    except Exception as e:
        print(str(e))
        raise e


def judge(
        src,
        language_config,
        max_cpu_time,
        max_memory,
        test_case_id=None,
        test_case=None,
        spj_version=None,
        spj_config=None,
        spj_compile_config=None,
        spj_src=None,
        output=False,
):
    if not (test_case or test_case_id) or (test_case and test_case_id):
        raise ValueError("invalid parameter")

    data = {
        "language_config": language_config,
        "src": src,
        "max_cpu_time": max_cpu_time,
        "max_memory": max_memory,
        "test_case_id": test_case_id,
        "test_case": test_case,
        "spj_version": spj_version,
        "spj_config": spj_config,
        "spj_compile_config": spj_compile_config,
        "spj_src": spj_src,
        "output": output,
    }
    return request(SERVER_BASE_URL + "/judge", data=data)


def codeapex_judge(id, generated_code):
    match = re.search(r'```cpp(.*?)```', generated_code, re.DOTALL)
    if match:
        generated_code = match.group(1)

    json_file_path = '/home/codeapex_teachagent/id_uuid_testcase.json'
    id_to_query = id
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    uuid = data.get(id_to_query, None)
    uuid = uuid['uuid']
    if not uuid:
        print(f"No uuid found for id {id_to_query}")

    cpp_lang_config = {
        "compile": {
            "src_name": "main.cpp",
            "exe_name": "main",
            "max_cpu_time": 10000,
            "max_real_time": 10000,
            "max_memory": 256 * 1024 * 1024,
            "compile_command": "/usr/bin/g++ -DONLINE_JUDGE -O2 -w -fmax-errors=3 -std=c++14 {src_path} -lm -o {exe_path}",
        },
        "run": {"command": "{exe_path}", "seccomp_rule": "c_cpp",
                "env": ["LANG=en_US.UTF-8", "LANGUAGE=en_US:en", "LC_ALL=en_US.UTF-8"]},
    }

    result = judge(
        src=generated_code,
        language_config=cpp_lang_config,
        max_cpu_time=10000,
        max_memory=1024 * 1024 * 256,
        test_case_id=uuid,
    )
    # print(f"generated_code: {generated_code}")
    # print(f"result: {result}")
    if result['err'] == "CompileError":
        average_score = 0
        err = 1
    elif not result['err']:
        err = 0
        test_cases = result['data']
        scores = [1 if case['result'] == 0 else 0 for case in test_cases]
        average_score = sum(scores) / len(test_cases) if test_cases else 0
    else:
        print(f"-----------result['err'] error!!!!: {result['err']}")
        err = 0
        test_cases = result['data']
        scores = [1 if case['result'] == 0 else 0 for case in test_cases]
        average_score = sum(scores) / len(test_cases) if test_cases else 0
    return average_score, err


def codeforce_judge(id, generated_code):
    match = re.search(r'```cpp(.*?)```', generated_code, re.DOTALL)
    if match:
        generated_code = match.group(1)

    json_file_path = '/home/codeforce_data/id_uuid_testcase.json'
    id_to_query = id
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    uuid = data.get(id_to_query, None)
    uuid = uuid['uuid']
    if not uuid:
        print(f"No uuid found for id {id_to_query}")

    cpp_lang_config = {
        "compile": {
            "src_name": "main.cpp",
            "exe_name": "main",
            "max_cpu_time": 10000,
            "max_real_time": 10000,
            "max_memory": 256 * 1024 * 1024,
            "compile_command": "/usr/bin/g++ -DONLINE_JUDGE -O2 -w -fmax-errors=3 -std=c++14 {src_path} -lm -o {exe_path}",
        },
        "run": {"command": "{exe_path}", "seccomp_rule": "c_cpp",
                "env": ["LANG=en_US.UTF-8", "LANGUAGE=en_US:en", "LC_ALL=en_US.UTF-8"]},
    }

    result = judge(
        src=generated_code,
        language_config=cpp_lang_config,
        max_cpu_time=10000,
        max_memory=1024 * 1024 * 256,
        test_case_id=uuid,
    )
    # print(f"generated_code: {generated_code}")
    # print(f"result: {result}")
    if result['err'] == "CompileError":
        average_score = 0
        err = 1
    elif not result['err']:
        err = 0
        test_cases = result['data']
        scores = [1 if case['result'] == 0 else 0 for case in test_cases]
        average_score = sum(scores) / len(test_cases) if test_cases else 0
    else:
        print(f"-----------result['err'] error!!!!: {result['err']}")
        err = 0
        test_cases = result['data']
        scores = [1 if case['result'] == 0 else 0 for case in test_cases]
        average_score = sum(scores) / len(test_cases) if test_cases else 0
    return average_score, err


class JsonlLoggingHandler:
    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def info(self, record):
        try:
            with open(self.filename, 'a', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')

        except Exception:
            self.handleError(record)


def del_dup(mapping):
    # delete key==value items
    res = {}
    for k, v in mapping.items():
        if k == v:
            continue
        res[k] = v
    return res


def clean_code_comments(code_string):

    code_string = re.sub(r'/\*[\s\S]*?\*/', '', code_string)
    code_string = re.sub(r'//.*', '', code_string)
    code_string = re.sub(r'\n\s*\n', '\n', code_string)

    return code_string.strip()


def tokenize_code(c_code):
    tokens = []
    code = clean_code_comments(c_code)
    for toknum, tokval, _, _, _ in tokenize.tokenize(BytesIO(code.encode('utf-8')).readline):
        if toknum not in {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.COMMENT}:
            tokens.append(tokval)
    return tokens


def compare_code(standard, wrong_code, final_code, similarity_threshold=0.8, difference_threshold=0.05):
    try:
        standard_token = tokenize_code(standard)
    except tokenize.TokenError as e:
        lines = standard.splitlines()
        lnum, _ = e.args[1]
        context = "\n".join(lines[max(0, lnum - 3):lnum + 2])
        print(f"Sandard code, Error on line {lnum}:\n{context}")
        raise e

    try:
        wrong_token = tokenize_code(wrong_code)
    except tokenize.TokenError as e:
        lines = wrong_code.splitlines()
        lnum, _ = e.args[1]
        context = "\n".join(lines[max(0, lnum - 3):lnum + 2])
        print(f"Wrong code, Error on line {lnum}:\n{context}")
        raise e
    try:
        final_token = tokenize_code(final_code)
    except tokenize.TokenError as e:
        lines = final_code.splitlines()
        lnum, _ = e.args[1]
        context = "\n".join(lines[max(0, lnum - 3):lnum + 2])
        print(f"Final code, Error on line {lnum}:\n{context}")
        raise e

    try:
        A_B_similarity = difflib.SequenceMatcher(None, standard_token, final_token).ratio()
        A_A_similarity = difflib.SequenceMatcher(None, wrong_token, final_token).ratio()
        A_C_similarity = difflib.SequenceMatcher(None, standard_token, wrong_token).ratio()
        if A_C_similarity > similarity_threshold:
            return False, [A_B_similarity, A_A_similarity, A_C_similarity]
        if A_A_similarity > similarity_threshold or A_A_similarity > A_B_similarity:
            return False, [A_B_similarity, A_A_similarity, A_C_similarity]
        if A_B_similarity > similarity_threshold or A_B_similarity > A_A_similarity + difference_threshold:
            return True, [A_B_similarity, A_A_similarity, A_C_similarity]
        else:
            return False, [A_B_similarity, A_A_similarity, A_C_similarity]
    except Exception as e:
        return False, [A_B_similarity, A_A_similarity, A_C_similarity]


def is_valid_cpp_variable(name):
    cpp_keywords = {
        "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor",
        "bool", "break", "case", "catch", "char", "char8_t", "char16_t", "char32_t",
        "class", "compl", "const", "constexpr", "const_cast", "continue", "decltype",
        "default", "delete", "do", "double", "dynamic_cast", "else", "enum", "explicit",
        "export", "extern", "false", "float", "for", "friend", "goto", "if", "inline",
        "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq",
        "nullptr", "operator", "or", "or_eq", "private", "protected", "public", "register",
        "reinterpret_cast", "return", "short", "signed", "sizeof", "static", "static_assert",
        "static_cast", "struct", "switch", "template", "this", "thread_local", "throw",
        "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using",
        "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq"
    }
    if not isinstance(name, str):
        return False
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
        return False

    if name in cpp_keywords:
        return False

    return True


if __name__ == '__main__':
    code = ''' '''
