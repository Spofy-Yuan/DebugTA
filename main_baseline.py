from studentSim import StudentSim
import argparse
import openai
import os
from torch.utils.data import DataLoader
from utils import *
from dataset import DataSet
from tqdm import tqdm
from baseline import BaseAgent
import re

# origin
# python main_baseline.py --strategy_name origin --teacher_model gpt-4o-mini --dataset codeforce --mode all --round 0 --origin --test_num 110 --Continue
# baseline direct_debug
# python main_baseline.py --strategy_name baseline_direct_debug --teacher_model gpt-4o-mini --dataset codeforce --mode all --round 0 --test_num 110 --Continue
# baseline debug with standard_code
# python main_baseline.py --strategy_name baseline_debug_with_standard_code --teacher_model gpt-4o-mini --dataset codeforce --mode all --test_num 110 --round 0 --Continue


API_KEY = "<API_KEY>"
openai.api_key = API_KEY

# os.environ["http_proxy"] = "http://127.0.0.1:8888"
# os.environ["https_proxy"] = "http://127.0.0.1:8888"
# os.environ["all_proxy"] = "socks5://127.0.0.1:8889"
# os.environ["OPENAI_API_KEY"] = API_KEY

parser = argparse.ArgumentParser(description='TeachAgent')
parser.add_argument('--debug', action='store_true', help='log debug messages or not')
parser.add_argument('--strategy_name', type=str, default='direct_debug', help='describe the strategy')
parser.add_argument('--prompt_dir', type=str, default='Prompts/', help='dir to save log and models')
parser.add_argument('--teacher_model', type=str, default='doubao')
parser.add_argument('--student_model', type=str, default='gpt-4o-mini')
parser.add_argument('--output_dir', type=str, default='outputs')
parser.add_argument('--test_num', type=int, default=300)
parser.add_argument('--device', type=int, default=5)
parser.add_argument('--Continue', action='store_true', help='continue from the last run with the same parameters')
parser.add_argument('--dataset', type=str, choices=['mongodb', 'oj', 'codeapex', 'codeforce', 'humaneval', 'apps'],
                    default='codeapex')
parser.add_argument('--round_num', type=int, default=3)
parser.add_argument('--mode', type=str, default='all', choices=['all', 'evaluate'])
parser.add_argument('--test_id', type=str, nargs='*', default=[])
parser.add_argument('--origin', action='store_true', help='use the original code as the input of the student model')

args = parser.parse_args()
oj_unknown_id = [1016, 1038, 1040, 1048, 1072, 1079, 1080, 1107, 1108, 1128, 1135, 1136, 1137, 1173, 1178, 1179]


def test(args, file_name, codefile, question_doc, models):
    logger = JsonlLoggingHandler(os.path.join(args.output_dir, file_name, str(question_doc["id"]) + '.jsonl'))
    logger.info(
        {"actor": "system", "info": "question doc", "message": question_doc["id"] + '\n' + question_doc["description"]})
    logger.info({"actor": "system", "info": "standard code", "message": question_doc["standard_code"]})

    student = StudentSim(args, question_doc, models['student_tokenizer'], models['student_model'])

    if args.origin:
        code = student.thought
    else:
        teacher = BaseAgent(args, question_doc, models['teacher_tokenizer'], models['teacher_model'])
        if args.strategy_name == 'baseline_direct_debug':
            code = teacher.debug_only(student, student.thought, question_doc)
        elif args.strategy_name == 'baseline_debug_with_standard_code':

            code = teacher.act_with_standard_code(student, student.thought, question_doc)
        else:
            assert False

    cpp_code_blocks = re.findall(r'```cpp\n(.*?)```', code, re.DOTALL)
    if cpp_code_blocks:
        code = max(cpp_code_blocks, key=len)
    if args.dataset == 'oj':
        result = OJ_judge(code, question_id=question_doc["id"])
        res, err = check_OJ_result(result)
    elif args.dataset == 'codeapex':
        result = codeapex_judge(question_doc["id"], code)
        res, err = result
    elif args.dataset == 'codeforce':
        result = codeforce_judge(question_doc["id"], code)
        res, err = result
    else:
        pass

    if result == None:
        print('OJTest Error {}'.format(question_doc["id"]))
        return "Error", 0

    with open(os.path.join(args.output_dir, file_name, codefile, str(question_doc["id"]) + '.json'), 'w',
              encoding='utf-8') as f:
        dump_dict = {"id": question_doc["id"], "code": code, "question": question_doc["description"], "result": result,
                     'ac': res}
        json.dump(dump_dict, f, ensure_ascii=False, indent=4)

    print(f"res_ac: {res}")
    print(f"errtype_ce: {err}")
    return res, err  # err为1表示compiler error

def get_tested_questions(continue_dir):
    tested_questions = set()
    for filename in os.listdir(continue_dir):
        if filename.endswith('.json'):
            question_id = int(filename.split('.')[0].split('-')[-1])
            tested_questions.add(question_id)
    return tested_questions


if __name__ == '__main__':
    if args.mode == 'evaluate':
        args.Continue = True
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.debug:
        file_name = 'debug'
        code_file_name = 'debug-code'
    else:
        params_str = f'''{args.dataset}-{args.strategy_name}-{args.teacher_model}-{args.student_model}-{args.round_num}'''
        file_name = f"log-{params_str}"
        args.log_file = file_name
        if args.Continue:
            print(f"Continue from {file_name}")
            assert os.path.exists(os.path.join(args.output_dir, file_name))
            code_file_name = "codelog"
        else:
            assert not os.path.exists(os.path.join(args.output_dir, file_name))
            os.makedirs(os.path.join(args.output_dir, file_name))
            code_file_name = "codelog"
            os.makedirs(os.path.join(args.output_dir, file_name, code_file_name))

    if args.Continue:
        continue_dir = os.path.join(args.output_dir, file_name, code_file_name)
        tested_questions = get_tested_questions(continue_dir)
    else:
        tested_questions = set()

    data_set = DataSet(args)
    dataloader = DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        collate_fn=data_set.collate_fn
    )

    teacher_tok, teacher_model = load_model_tokenizer(args.teacher_model, device='cuda:{}'.format(args.device))
    stu_tok, stu_model = load_model_tokenizer(args.student_model, device='cuda:{}'.format(args.device))
    models = {
        'teacher_tokenizer': teacher_tok,
        'teacher_model': teacher_model,
        'student_tokenizer': stu_tok,
        'student_model': stu_model
    }
    result_list = []
    ce = 0  # compiler error
    wa = 0  # wrong answer
    total_num = len(tested_questions)
    if args.mode == 'all':
        for question in dataloader:
            question_id = int(question[0]["id"])
            if len(tested_questions) >= args.test_num:
                break
            if len(args.test_id) > 0 and str(question_id) not in args.test_id:
                continue

            if args.dataset == 'oj':
                if question_id in tested_questions or question_id in oj_unknown_id:
                    continue
            total_num += 1
            ac_rate, error_type = test(args, file_name, code_file_name, question[0], models)
            if ac_rate == "Error":
                assert False

            tested_questions.add(question_id)
            result_list.append(ac_rate)
            print(
                f"Number {total_num}: {question_id} ------------------------------------------------------------------------------------------")
            if ac_rate == 0:
                if error_type == 1:  # compiler error
                    ce += 1
                else:
                    wa += 1

        if len(result_list):
            print(result_list)
            print('Average Accuracy:', sum(result_list) / len(result_list))
        else:
            print("result_list is empty!")
        print(f"Wrong Answer: {wa}")
        # save log
        with open(os.path.join(args.output_dir, file_name, 'final_log.txt'), 'w', encoding='utf-8') as f:
            f.write(', '.join([f'{key}={value}' for key, value in vars(args).items()]))
            f.write("Wrong Answer: {}\n".format(wa))
            f.write("Compiler Error: {}\n".format(ce))
            f.write("Total Number: {}\n".format(total_num))
            f.write("Average Accuracy: {}\n".format(sum(result_list) / len(result_list))
                    if len(result_list) else "No result")

    elif args.mode == 'evaluate':
        print('Evaluating...')
        wa = 0
        ce = 0
        ac = 0
        code_error = 0
        score_list = []
        res_record = {}

        for question in tqdm(dataloader, desc="Evaluating"):

            if len(score_list) >= total_num:
                break
            question_id = int(question[0]["id"])
            wrong_code = question[0]['wrong_code']
            origin_code = question[0]['standard_code']
            # print('Origin code:', origin_code)
            standard_code = origin_code
            if len(score_list) >= args.test_num:
                break
            # print(tested_questions)
            if question_id in tested_questions:
                # print(question_id)
                with open(os.path.join(args.output_dir, file_name, code_file_name, str(question_id) + '.json'), 'r',
                          encoding='utf-8') as f:
                    data = json.load(f)

                final_code = data['code']

                result = data['result']

                if args.dataset == 'codeapex' or args.dataset == 'codeforce':
                    res, err = result
                else:
                    res, err = check_result(result) if args.dataset == 'mongodb' else check_OJ_result(result)

                error_message = 'Correct'
                if err == 1:
                    ce += 1
                    error_message = 'Compiler Error'
                else:
                    if args.dataset == 'oj':
                        s_t = 0.8
                        d_t = 0.15
                    elif args.dataset == 'codeapex':
                        s_t = 0.9
                        d_t = 0.05
                    else:
                        s_t = 0.8
                        d_t = 0.15

                    dc_res, score = compare_code(standard_code, wrong_code, final_code, similarity_threshold=s_t,
                                                 difference_threshold=d_t)
                    # dc_res = False
                    if dc_res:
                        res = 0
                        code_error += 1
                        error_message = 'Code Error'
                        print(f"Question: {question_id}, score:{score}")
                        # print(f"standard_code: {standard_code}")
                        # print(f"wrong_code: {wrong_code}")
                        # print(f'Question{question_id}, res: {res}, err: {err}, dc_res: {dc_res}, score:{score}')
                    elif res < 1:
                        wa += 1
                        error_message = 'Wrong Answer'
                        # print(f'Question{question_id}, res: {res}, err: {err}')
                    else:
                        ac += 1
                        # print(f'Question{question_id}, AC!, score:{score}')

                score_list.append(res)
                res_record[question_id] = {"res": res, "err": error_message, "code": final_code}

        total_num = len(score_list)
        print(f"Avg Accuracy: {sum(score_list) / len(score_list)}")
        print(f"Accepted: {ac}")
        print(f"Wrong Answer: {wa}")
        print(f"Compiler Error: {ce}")
        print(f"Code Error: {code_error}")
        print(f"Total Number: {total_num}")
        print(f"Plagiarism Rate: {code_error / total_num}")
        print(f"AC@all: {ac / total_num}")
        print(f"Compile Rate: {1 - ce / total_num}")

        with open(os.path.join(args.output_dir, file_name, 'final_log.txt'), 'w', encoding='utf-8') as f:
            f.write(', '.join([f'{key}={value}' for key, value in vars(args).items()]) + '\n')
            f.write("Accepted: {}\n".format(ac))
            f.write("Wrong Answer: {}\n".format(wa))
            f.write("Compiler Error: {}\n".format(ce))
            f.write("Code Error: {}\n".format(code_error))
            f.write("Total Number: {}\n".format(total_num))
            f.write("Plagiarism Rate: {}\n".format(code_error / total_num))
            f.write("AC@all: {}\n".format(ac / total_num))
            f.write("Compile Rate: {}\n".format(1 - ce / total_num))
            f.write("Average Accuracy: {}\n".format(sum(score_list) / len(score_list))
                    if len(score_list) else "No result")

        import pandas as pd

        res_path = os.path.join(args.output_dir, file_name, 'final_result.csv')
        df = pd.DataFrame.from_dict(res_record, orient='index')
        df.to_csv(res_path, encoding='utf-8')
