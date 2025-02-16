import os
from utils import *
import json


class StudentSim(object):
    def __init__(self, args, target_question, tokenizer, model):
        self.args = args
        self.question = target_question

        if self.args.dataset == 'oj' or 'codeapex' or 'codeforce':
            self.target_question = '\n'.join([
                '题目描述：{}'.format(self.question["description"]),
            ])
        else:
            self.target_question = '\n'.join([
                '题目描述：{}'.format(self.question["description"]),
                '输入描述：{}'.format(self.question["input_description"]),
                '输出描述：{}'.format(self.question["output_description"]),
            ])
        
        self.model_name = args.student_model
        
        self.args = args
        self.tokenizer, self.model = tokenizer, model
        self.init_answer()

    def update_thought(self, thought):
        try:
            code = extract_code(thought)
            if code == None:
                self.thought = thought
            else:
                self.thought = code
        except:
            self.thought = thought

    def init_answer(self):
        with open(os.path.join(self.args.prompt_dir,'student', "StudentInit.json"), 'r')as f:
            prompt_dir = json.load(f)
        if not os.path.exists(os.path.join('data', 'initCode', self.args.student_model)):
            os.mkdir(os.path.join('data', 'initCode', self.args.student_model))
        # path = os.path.join('data', 'initCode', self.args.student_model, f'{self.question["id"]}.cpp')
        if self.args.dataset == 'oj':
            path = os.path.join('/home/shared/EduData/oj_2023_04_23/wrongCode', f'{self.question["id"]}.cpp')
        elif self.args.dataset == 'codeapex':
            path = os.path.join('/home/shared/EduData/codeapex_teachagent/wrongCode', f'{self.question["id"]}.cpp')
        elif self.args.dataset == 'codeforce':
            path = os.path.join('/home/shared/EduData/codeforce_data/wrong_code', f'{self.question["id"]}.cpp')
        else:
            path = os.path.join('data', 'initCode', 'doubao', f'{self.question["id"]}.cpp')

        if os.path.exists(path):
            with open(path, 'r') as f:
                self.thought = f.read()
                
        else:
            system_prompt = '\n'.join(prompt_dir["system"])
            prompt = '\n'.join(prompt_dir["prompt"])
            prompt = prompt.replace("{QuestionText}", self.target_question)
            prompt = prompt.replace("{Code}", self.thought)
            response = generate_response_api(prompt, self.tokenizer, self.model, self.model_name, system_message=system_prompt)
            self.update_thought(response)
            with open(path, 'w') as f:
                f.write(self.thought)
        
            
    def think(self, teacher_prompt):
        with open(os.path.join(self.args.prompt_dir, 'student', 'Student.txt'), 'r') as f:
            system_prompt = f.read()
        with open(os.path.join(self.args.prompt_dir, 'student', 'StudentThinkPrompt.txt'), 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        prompt_template = prompt_template.replace("{Code}", self.thought)
        prompt_template = prompt_template.replace("{TeacherPrompt}", teacher_prompt)
        prompt_template = prompt_template.replace("{QuestionText}", self.target_question)
        
        response = generate_response_api(prompt_template, self.tokenizer, self.model, self.model_name,system_message=system_prompt)
        self.update_thought(response)
        return prompt_template, response

