from utils import *
from collections import Counter
from math import log
from transformers import AutoTokenizer

class CodeRetriever(object):
    def __init__(self, dataset):   
        self.dataset = dataset  
        self.pool = self.process()
        self.b = 0.75
        self.k1 = 2.0
    
    def clean_code_comments(self, code_string):
        code_string = re.sub(r'/\*[\s\S]*?\*/', '', code_string)
        code_string = re.sub(r'//.*', '', code_string)
        code_string = re.sub(r'\n\s*\n', '\n', code_string)
        
        return code_string.strip()


    def process(self):
        if self.dataset == 'oj':
            path = '/home/share/EduData/oj_2023_04_23/StandardCode.jsonl'
            with open(path, 'r') as f:
                record = [json.loads(line) for line in f]
            pool = {}
            for item in record:
                qid = item['ID']
                code = item['Code']
                pool[qid] = code
            return pool
        elif self.dataset == 'codeapex':
            path = '/home/share/EduData/codeapex_teachagent/StandardCode.jsonl'
            with open(path, 'r') as f:
                record = [json.loads(line) for line in f]
            pool = {}
            for item in record:
                qid = item['ID']
                code = item['Code']
                pool[qid] = code
            return pool
        elif self.dataset == 'codeforce':
            path = '/home/share/EduData/codeforce_data/StandardCode.jsonl'
            with open(path, 'r') as f:
                record = [json.loads(line) for line in f]
            pool = {}
            for item in record:
                qid = item['ID']
                code = item['Code']
                pool[qid] = code
            return pool
        else:
            pass
    
    def get_code(self, qid, wrong_code):

        if qid in self.pool:
            return self.find_most_similar(wrong_code, self.pool[qid])
        elif str(qid) in self.pool:
            return self.find_most_similar(wrong_code, self.pool[str(qid)])
        else:
            return None
    
    def tokenize_code(self, code_string, tokenizer_name='gpt2'):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        cleaned_code = self.clean_code_comments(code_string)
        chunk_size = 1024
        chunks = [cleaned_code[i:i + chunk_size] for i in range(0, len(cleaned_code), chunk_size)]
        result = []
        for chunk in chunks:
            result += tokenizer.encode(chunk)
        return result

    def calculate_bm25_score(self, target_tokens, candidate_tokens, avg_doc_length, doc_freqs, N):
        score = 0
        candidate_term_freq = Counter(candidate_tokens)
        
        for token in set(target_tokens):
            if token in doc_freqs:
                idf = log((N - doc_freqs[token] + 0.5) / (doc_freqs[token] + 0.5) + 1.0)
                tf = candidate_term_freq[token]
                doc_length_norm = (1 - self.b + self.b * len(candidate_tokens) / avg_doc_length)
                term_score = (tf * (self.k1 + 1)) / (tf + self.k1 * doc_length_norm)
                score += idf * term_score
        
        return score

    def find_most_similar(self, target_code, code_pool):
        target_tokens = self.tokenize_code(target_code)
        tokenized_pool = [self.tokenize_code(code) for code in code_pool]
        avg_doc_length = sum(len(tokens) for tokens in tokenized_pool) / len(tokenized_pool)
        doc_freqs = Counter()
        for tokens in tokenized_pool:
            doc_freqs.update(set(tokens))
        scores = []
        for i, candidate_tokens in enumerate(tokenized_pool):
            score = self.calculate_bm25_score(
                target_tokens,
                candidate_tokens,
                avg_doc_length,
                doc_freqs,
                len(code_pool)
            )
            scores.append((score, i))

        best_score, best_idx = max(scores)
        return code_pool[best_idx]
