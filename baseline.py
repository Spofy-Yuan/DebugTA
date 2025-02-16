from utils import *
import os
import json
import re


class BaseAgent(object):
    def __init__(self, args, question, tokenizer, model):
        self.args = args
        self.question = question
        self.tokenizer, self.model = tokenizer, model
        if self.args.dataset == 'oj' or 'codeapex' or 'codeforce':
            self.question_text = '\n'.join([
                'problem description：{}'.format(self.question["description"]),
            ])
        else:
            self.question_text = '\n'.join([
                'problem description：{}'.format(self.question["description"]),
                'input description：{}'.format(self.question["input_description"]),
                'output description：{}'.format(self.question["output_description"]),
            ])

        self.answer_text = question["standard_code"]
        self.model_name = args.teacher_model
        self.question_id = question["id"]

    def debug_only(self, code, question_doc):
        # directly debug and return the answer
        if code.startswith("```cpp") and code.endswith("```"):
            code = code[6:-3]

        if self.args.dataset == 'oj':
            result = OJ_judge(code, question_id=question_doc["id"])
            res, err = check_OJ_result(result)
        elif self.args.dataset == 'codeapex':
            result = codeapex_judge(question_doc["id"], code)
            res, err = result
        elif self.args.dataset == 'codeforce':
            result = codeforce_judge(question_doc["id"], code)
            res, err = result
        else:
            result = OJTest(code, question_id=question_doc["id"])
            res, err = check_result(result)

        if res == 1:
            new_code = code
            return new_code

        system_prompt = '''你是一个有经验的编程专家，请你直接在学生的错误代码基础上进行修改，改动尽量小，不改变思路方法和变量名称，直接给出修改后的正确的代码。要求：只输出代码，不输出其他任何文字说明。
                    '''
        prompt = f'''任务要求：
                    ```
                    {self.question_text}
                    ```
                    错误代码：
                    ```cpp
                    {code}
                    ```
                    '''

        new_code = generate_response_api(prompt, self.tokenizer, self.model, system_message=system_prompt,
                                         model_name=self.model_name, device="cuda:{}".format(self.args.device))

        if new_code.startswith("```cpp") and new_code.endswith("```"):
            new_code = new_code[6:-3]

        return new_code

    def act_with_standard_code(self, student, code, question_doc):
        # directly debug with standard code and return the answer
        if code.startswith("```cpp") and code.endswith("```"):
            code = code[6:-3]

        if self.args.dataset == 'oj':
            result = OJ_judge(code, question_id=question_doc["id"])
            res, err = check_OJ_result(result)
        elif self.args.dataset == 'codeapex':
            result = codeapex_judge(question_doc["id"], code)
            res, err = result
        elif self.args.dataset == 'codeforce':
            result = codeforce_judge(question_doc["id"], code)
            res, err = result
        else:
            result = OJTest(code, question_id=question_doc["id"])
            res, err = check_result(result)

        if res == 1:
            new_code = code
            return new_code

        system_prompt = '''你是一个有经验的编程专家，请你直接在学生的错误代码基础上进行修改，改动尽量小，不改变思路方法和变量名称，直接给出修改后的正确的代码。要求：只输出代码，不输出其他任何文字说明。
                    '''
        prompt = f'''任务要求：
                    ```
                    {self.question_text}
                    ```
                    错误代码：
                    ```cpp
                    {code}
                    ```
                    参考答案：
                    ```cpp
                    {self.answer_text}
                    ```
                    '''

        new_code = generate_response_api(prompt, self.tokenizer, self.model, system_message=system_prompt,
                                         model_name=self.model_name, device="cuda:{}".format(self.args.device))

        if new_code.startswith("```cpp") and new_code.endswith("```"):
            new_code = new_code[6:-3]

        return new_code
