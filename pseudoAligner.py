from utils import *
import os
import json
import re
from retriever import CodeRetriever
from utils import is_valid_cpp_variable


class PseudoAligner(object):
    def __init__(self, args, question, tokenizer, model, logger, wrong_code=None):
        self.args = args
        self.question = question
        self.logger = logger
        self.tokenizer, self.model = tokenizer, model
        if self.args.dataset == 'oj' or 'codeapex' or 'codeforce':
            question_text = '\n'.join([
                '题目描述：{}'.format(self.question["description"]),
            ])
        else:
            question_text = '\n'.join([
                '题目描述：{}'.format(self.question["description"]),
                '输入描述：{}'.format(self.question["input_description"]),
                '输出描述：{}'.format(self.question["output_description"]),
            ])
        if args.use_retrieve:
            print("Using retriever!")
            self.retriever = CodeRetriever(args.dataset)
            self.answer_text = self.retriever.get_code(question["id"], wrong_code)
        else:
            self.answer_text = question["standard_code"]
        # print('===========Pseudo Aligner INITIALIZATION============')
        # print(f'''Question ID: {question["id"]}''')
        # print(f'''Answer_text: {self.answer_text}''')

        self.model_name = args.teacher_model
        self.question_id = question["id"]
        self.question_text = question_text
        self.pseudo_code = self.get_pseudoCode(self.answer_text, "correct code")

        self.log_file = f'''log-{self.args.dataset}-{self.args.strategy_name}-{self.args.teacher_model}-{self.args.student_model}-{self.args.round_num}'''
        self.logger.info({"actor": "Retriever", "message": f'''standard Answer:{self.answer_text}\n'''})

    def get_description(self, question_id, question_text, answer_text):
        path = os.path.join('data/description', f'''{question_id}.txt''')
        if os.path.exists(path):
            with open(path, 'r') as f:
                desc = f.read()
                return desc

        system_prompt = ''' You're a well-experienced programmer. You received a task to simplify and clearify the question description.
Please generate a clear and concise description for the following question. Please delete the scence description and story line, and only keep the necessary information (program requirements, input format, output format, etc.).
Sample input:
```
question:
小米接到了一个任务，需要帮助他的朋友小明解决一个问题。小明有一个长度为n的数组，他想知道这个数组中的所有元素的和。请你帮助小明编写一个程序，计算数组中所有元素的和。
输入描述：
第一行包含一个整数n，表示数组的长度。
第二行包含n个整数，表示数组中的元素。
输出描述：
一个整数，表示数组中所有元素的和。
```
Sample output:
```
输入一个长度为n的数组，输出数组中所有元素的和。
输入描述：
第一行包含一个整数n，表示数组的长度。
第二行包含n个整数，表示数组中的元素。
输出描述：
一个整数，表示数组中所有元素的和。
```
'''
        prompt = f'''question:
{question_text}
'''
        description = generate_response_api(prompt, self.tokenizer, self.model, system_message=system_prompt,
                                            model_name=self.model_name, device="cuda:{}".format(self.args.device))
        with open(path, 'w') as f:
            f.write(description)
        return description

    def get_pseudoCode(self, code, name='correct code'):
        #
        system_prompt = '''Please generate pseudo code for the following C++ code.  
Format:\{algorithm2e\} package in latex. 
注意：只输出伪代码本身，不要输出额外信息。（包括\\documentclass,\\usepackage,\\begin\{document\}等内容）。
请将algorithm name写到伪代码的caption里。
'''
        prompt = f'''
code:
```cpp
{code}
```
algorithm name: {name}.
'''
        pseudo = generate_response_api(prompt, self.tokenizer, self.model, system_message=system_prompt,
                                       model_name=self.model_name, device="cuda:{}".format(self.args.device))
        return pseudo

    def aligner(self, wrong_code):
        wrong_pse = self.get_pseudoCode(wrong_code, "wrong code")

        system_prompt = '''你是一个有经验的C++编程专家，你收到了一个任务的两份伪代码，一份正确代码和一份错误代码。现在，请你找出正确代码和错误代码中变量名的对应关系。两份代码中的对应变量具有相似的作用，但是注意以下几点：1. 避免仅仅调换两个变量的顺序； 2. 仅关注变量名称，避免混淆不同变量类型，检查变量类型，尤其要注意避免修改后发生变量命名冲突； 3. 不改变变量含义，保证代码仍然正确，仅仅对应变量名称。
要求只以 JSON 格式输出正确代码和错误代码的变量对应关系，无需输出任何文字和说明，具体结构如下：```
{
  "正确代码变量":"错误代码变量"
}
```
请你分析代码变量的用途后，参照格式只输出这个对应关系的JSON。

样例：
正确代码：
```
\\begin\{algorithm\}
\caption{algorithm name}
\KwIn{$n$ is the range.}
\KwOut{$sum$ is the sum of the range.}
set sum = 0
\For{$t = 1$ \KwTo $n$}{
    sum += t\;
}
Print $sum$;
\end{algorithm}
```

错误代码：
```
\\begin\{algorithm\}
\caption{algorithm name}
\KwIn{$M$ is the range.}
\KwOut{$s$ is the sum of the range.}
set s = 0
\For{$i = 1$ \KwTo $M$}{
    s += i;
}
Print $s$;
\end{algorithm}
```

输出如下：
```
{
    "n":"M",
    "t":"i",
    "sum":"s"
}
```
'''

        prompt = f'''
错误代码：
```
{wrong_pse}
```

正确代码：
```
{self.pseudo_code}
```
'''

        res = generate_response_api(prompt, self.tokenizer, self.model, system_message=system_prompt,
                                    model_name=self.model_name, device="cuda:{}".format(self.args.device))

        pattern = r'{(.*?)}'
        matches = re.findall(pattern, res, re.DOTALL)
        for match in matches:
            try:
                json_str = match.strip()
                json_obj = json.loads('{' + json_str + '}')
                self.logger.info(
                    {"actor": "aligner", "message": f'''wrong pse:{wrong_pse}\n correct pse:{self.pseudo_code}\n'''})
                self.logger.info({"actor": "aligner", "message": str(json_obj)})
                return del_dup(json_obj)
            except json.JSONDecodeError as e:
                continue
        return {}

    def dir_aligner(self, wrong_code):
        wrong_pse = self.get_pseudoCode(wrong_code, "wrong code")

        system_prompt = '''你是一个有经验的C++编程专家，你收到了一个任务的两份伪代码，一份正确代码和一份错误代码。现在，请你找出正确代码和错误代码中变量名的对应关系。两份代码中的对应变量具有相似的作用，但是注意以下几点：1. 避免仅仅调换两个变量的顺序； 2. 仅关注变量名称，避免混淆不同变量类型，检查变量类型，尤其要注意避免修改后发生变量命名冲突； 3. 不改变变量含义，保证代码仍然正确，仅仅对应变量名称。
要求只以 JSON 格式输出正确代码和错误代码的变量对应关系，无需输出任何文字和说明，具体结构如下：```
{
  "正确代码变量":"错误代码变量"
}
```
请你分析代码变量的用途后，参照格式只输出这个对应关系的JSON。

样例：
正确代码：
```
#include<iostream>
int main(){
    int n, m;
    cin >> n >> m;
    int sum;
    for (int i = m; i< n; i++)
        sum += i;
    cout << sum << endl;
}
```

错误代码：
```
#include<iostream>
int main(){
    int start, end;
    cin >> end >> start;
    int res;
    int m;
    for (m = start; m <= end - 1; m++)
        res += m;
    cout << res << endl;
}
```

输出如下：
```
{
    "n":"end",
    "m":"start",
    "i":"m",
    "sum":"res"
}
```
'''

        prompt = f'''
错误代码：
```
{wrong_code}
```

正确代码：
```
{self.answer_text}
```
'''
        res = generate_response_api(prompt, self.tokenizer, self.model, system_message=system_prompt,
                                    model_name=self.model_name, device="cuda:{}".format(self.args.device))

        pattern = r'{(.*?)}'
        matches = re.findall(pattern, res, re.DOTALL)
        for match in matches:
            try:
                json_str = match.strip()
                json_obj = json.loads('{' + json_str + '}')
                self.logger.info(
                    {"actor": "aligner", "message": f'''wrong pse:{wrong_pse}\n correct pse:{self.pseudo_code}\n'''})
                self.logger.info({"actor": "aligner", "message": str(json_obj)})
                return del_dup(json_obj)
            except json.JSONDecodeError as e:
                continue
        return {}

    def compile_error(self, student_response, error_message):
        system_prompt = '''你是一个有经验的程序员，请你根据给定的编译错误信息，找到代码的错误原因，并分点给出修改建议。
'''

        prompt = f'''错误代码：
{student_response}
编译器报错：
{error_message}
'''

        res = generate_response_api(prompt, self.tokenizer, self.model, system_message=system_prompt,
                                    model_name=self.model_name, device="cuda:{}".format(self.args.device))

        self.logger.info({"actor": "teacher", "info": error_message, "message": res})
        return res

    def substitutor(self, align_dict, question_doc):
        new_correct_code = self.answer_text

        if self.args.dataset == 'oj':
            result = OJ_judge(new_correct_code, question_id=question_doc["id"])
            res, err = check_OJ_result(result)
            if res != 1:
                print(f"WARNING: Q-{self.question_id} correct code cannot pass the test")
        elif self.args.dataset == 'codeapex':
            result = codeapex_judge(question_doc["id"], new_correct_code)
            res, err = result
            if res != 1:
                print(f"WARNING: Q-{self.question_id} correct code cannot pass the test")
        elif self.args.dataset == 'codeforce':
            result = codeforce_judge(question_doc["id"], new_correct_code)
            res, err = result
            if res != 1:
                print(f"WARNING: Q-{self.question_id} correct code cannot pass the test")
        else:
            result = OJTest(new_correct_code, question_id=question_doc["id"])
            res, err = check_result(result)

        for old_var, new_var in align_dict.items():
            if (new_var is not None) and is_valid_cpp_variable(old_var) and is_valid_cpp_variable(new_var):
                new_correct_code = re.sub(r'\b{}\b'.format(re.escape(new_var)), old_var, new_correct_code)

        if self.args.dataset == 'oj' or 'codeapex' or 'codeforce':
            if self.args.dataset == 'oj':
                result = OJ_judge(new_correct_code, question_id=question_doc["id"])
                res, err = check_OJ_result(result)
            elif self.args.dataset == 'codeapex':
                result = codeapex_judge(question_doc["id"], new_correct_code)
                res, err = result
            elif self.args.dataset == 'codeforce':
                result = codeforce_judge(question_doc["id"], new_correct_code)
                res, err = result
            if res != 1:
                # print(f"WARNING: Q-{self.question_id} substitute: \n{new_correct_code}")
                system_prompt = f'''请你根据给定的变量对应关系，将给出的code中的变量名称key替换成value。请注意，不要做其他的修改。
样例输入：
```
变量对应：{{"key":"value"}}
代码：
int main(){{
    int key;
    cin >> key;
    cout << key;
}}
```
样例输出：
```
int main(){{
    int value;
    cin >> value;
    cout << value;
}}
```
'''
                prompt = f'''变量对应：{align_dict}
代码：{self.answer_text}
'''

                gpt_new_code = generate_response_api(prompt, self.tokenizer, self.model, system_message=system_prompt,
                                                     model_name=self.model_name,
                                                     device="cuda:{}".format(self.args.device))
                if self.args.dataset == 'oj':
                    result = OJ_judge(gpt_new_code, question_id=question_doc["id"])
                    res, err = check_OJ_result(result)
                elif self.args.dataset == 'codeapex':
                    result = codeapex_judge(question_doc["id"], gpt_new_code)
                    res, err = result
                elif self.args.dataset == 'codeforce':
                    result = codeforce_judge(question_doc["id"], gpt_new_code)
                    res, err = result

                new_code = gpt_new_code if res else self.answer_text
                substitute_file = os.path.join(self.args.output_dir, self.log_file,
                                               'substitute.csv') if not self.args.debug else 'debug/substitute.csv'

                if os.path.exists(substitute_file):
                    with open(substitute_file, '+a') as f:
                        f.write(f'{self.question_id},{align_dict},{self.answer_text},{new_code}\n')
                else:

                    with open(substitute_file, 'w') as f:
                        f.write('question_id,align_dict,answer_text,new_correct_code\n')
                        f.write(f'{self.question_id},{align_dict},{self.answer_text},{new_code}\n')


        else:
            result = OJTest(new_correct_code, question_id=question_doc["id"])
            res, err = check_result(result)

        return new_correct_code

    def act(self, student, code, question_doc) -> str:
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
        # print(f"result: {result}, error: {error}")

        if res == 1:
            return "Exit"

        if self.args.dataset == 'mongodb':
            if result["err"] == "CompileError":
                error_message = result["data"]
                response = self.compile_error(code, error_message)
                return response
        elif self.args.dataset == 'oj':
            if result["status"] == "compile_error":
                error_message = result["message"]
                response = self.compile_error(code, error_message)
                return response
        elif self.args.dataset == 'codeapex' or 'codeforce':
            if err == 1:
                error_message = "CompileError"
                response = self.compile_error(code, error_message)
                return response

        response = self.aligner(code)
        new_correct_code = self.substitutor(response, question_doc)

        system_prompt = '''你是一个有经验的编程教师，请你比对学生的错误代码和参考答案，找到错误的code_snippet，并在issue中说明错误原因，给出对应的suggestion。
注意：用下面的JSON格式输出，只需要给出错误的地方和错误原因，不需要给出正确的代码。如果代码已经正确，请输出'Exit'即可。
样例输出：
{{
    "errors": [
      {{
        "code_snippet": "for(i=0; i<n; i++)",
        "issue": "变量i未声明",
        "suggestion": "应在for循环前声明int i"
      }},
      {{
        "code_snippet": "for(j=0; j<m; j++)",
        "issue": "变量j未声明",
        "suggestion": "应在for循环前声明int j"
      }}
    ]
}}
'''

        if self.args.dataset == 'oj':
            prompt = f'''任务要求:
```
{self.question_text}
```

错误代码：
```cpp
{code}
```

参考答案：
```cpp
{new_correct_code}
```

测试报错：{result["status"]}


'''
        else:
            prompt = f'''任务要求:
            ```
            {self.question_text}
            ```

            错误代码：
            ```cpp
            {code}
            ```

            参考答案：
            ```cpp
            {new_correct_code}
            ```

            '''

        self.logger.info({"actor": "teacher", "message": f'''wrong_code:{code}\n correct_code:{new_correct_code}\n'''})

        response = generate_response_api(prompt, self.tokenizer, self.model, system_message=system_prompt,
                                         model_name=self.model_name, device="cuda:{}".format(self.args.device))
        self.logger.info({"actor": "teacher", "message": response})
        return response

    def one_round_baseline(self, student, code, question_doc) -> str:
        if code.startswith("```cpp") and code.endswith("```"):
            code = code[6:-3]
        system_prompt = f'''你是一个有经验的程序员，现在你正在检查一份有问题的C++算法代码，我会给你任务描述，错误代码和正确代码。请你找出错误代码中的错误，并给出修改建议。要求：只输出修改建议，不输出正确答案。
'''
        prompt = f'''任务要求：
{self.question_text}
错误代码：
```cpp
{code}
```
参考答案：
```cpp
{self.answer_text}
```
'''
        res = generate_response_api(prompt, self.tokenizer, self.model, system_message=system_prompt,
                                    model_name=self.model_name, device="cuda:{}".format(self.args.device))
        return res
