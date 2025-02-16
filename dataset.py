import torch
from torch.utils import data
import json
import os


class DataSet(data.Dataset):
    def __init__(self, args):
        self.args = args
        if self.args.dataset == 'oj':
            self.data_path = '/home/shared/EduData/oj_2023_04_23/OJ_debug.jsonl'
            self.code_path = '/home/shared/EduData/oj_2023_04_23/StandardCode.jsonl'
            self.id_list = []
            self.descriptions = []
            self.standard_code = {}
            self.wrong_code = {}
            with open(self.data_path, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    self.id_list.append(data["ID"])
                    self.descriptions.append(data["description"])

            with open(self.code_path, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    self.standard_code[data['ID']] = data["Code"]

            for id in self.id_list:
                if os.path.exists(os.path.join('/home/shared/EduData/oj_2023_04_23/wrongCode', f'{id}.cpp')):
                    path = os.path.join('/home/shared/EduData/oj_2023_04_23/wrongCode', f'{id}.cpp')
                    with open(path, 'r') as f:
                        self.wrong_code[id] = f.read()
                else:
                    self.wrong_code[id] = None

        elif self.args.dataset == 'codeapex':
            self.data_path = '/home/shared/EduData/codeapex_teachagent/OJ_debug.jsonl'
            self.code_path = '/home/shared/EduData/codeapex_teachagent/StandardCode.jsonl'
            self.id_list = []
            self.descriptions = []
            self.standard_code = {}
            self.wrong_code = {}
            with open(self.data_path, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    self.id_list.append(data["ID"])
                    self.descriptions.append(data["description"])

            with open(self.code_path, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    self.standard_code[data['ID']] = data["Code"]

            for id in self.id_list:
                if os.path.exists(
                        os.path.join('/home/shared/EduData/codeapex_teachagent/wrongCode', f'{id}.cpp')):
                    path = os.path.join('/home/shared/EduData/codeapex_teachagent/wrongCode', f'{id}.cpp')
                    with open(path, 'r') as f:
                        self.wrong_code[id] = f.read()
                else:
                    self.wrong_code[id] = None

        elif self.args.dataset == 'codeforce':
            self.data_path = '/home/shared/EduData/codeforce_data/OJ_debug_new.jsonl'
            self.code_path = '/home/shared/EduData/codeforce_data/StandardCode.jsonl'
            self.id_list = []
            self.descriptions = []
            self.standard_code = {}
            self.wrong_code = {}
            with open(self.data_path, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    self.id_list.append(data["ID"])
                    self.descriptions.append(data["description"])

            with open(self.code_path, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    self.standard_code[data['ID']] = data["Code"]

            for id in self.id_list:
                if os.path.exists(
                        os.path.join('/home/shared/EduData/codeforce_data/wrong_code', f'{id}.cpp')):
                    path = os.path.join('/home/shared/EduData/codeforce_data/wrong_code', f'{id}.cpp')
                    with open(path, 'r') as f:
                        self.wrong_code[id] = f.read()
                else:
                    self.wrong_code[id] = None

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, i):
        if self.args.dataset == 'oj' or self.args.dataset == 'codeapex' or self.args.dataset == 'codeforce':
            problem_id = self.id_list[i]
            return {
                "id": self.id_list[i],
                "description": self.descriptions[i],
                "standard_code": self.standard_code[str(problem_id)][0],
                'wrong_code': self.wrong_code[self.id_list[i]]
            }

    def collate_fn(self, batch):
        return batch
