# -*- coding: utf-8 -*-
"""领域知识库：加载 JSONL 切片 + 纯 Python BM25 检索（零第三方依赖）。

设计说明
--------
* 检索用 BM25（CJK 采用字符二元组 + ASCII 词元，无需分词器与外部模型）。
* 每条切片带 source / section，检索结果天然可回溯到出处 —— 这是"高保真知识溯源"
  得以实现的底座：生成与审核的每一步都拿到 (doc_id, 原文, 出处)。
"""
import json
import math
import os
import re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_TOKEN_RE = re.compile(r'[A-Za-z][A-Za-z0-9_\-\.]*|\d+(?:\.\d+)?|[一-鿿]')


def tokenize(text):
    """ASCII 词 + 单字 + 相邻汉字二元组。返回 list。"""
    toks = _TOKEN_RE.findall(text)
    cjk = [t for t in toks if '一' <= t[0] <= '鿿']
    grams = ['%s%s' % (cjk[i], cjk[i + 1]) for i in range(len(cjk) - 1)]
    return [t.lower() for t in toks] + grams


class KnowledgeBase:
    """JSONL 知识库切片 + BM25 索引。"""

    def __init__(self, path=None):
        self.path = path or os.path.join(ROOT, 'kb', 'domain_kb.jsonl')
        self.docs = []
        with open(self.path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.docs.append(json.loads(line))
        self.by_id = {d['doc_id']: d for d in self.docs}
        self._build_index()

    # ---------------------------------------------------------------- 索引
    def _build_index(self):
        self.tf, self.df, self.doc_len = [], Counter(), []
        for d in self.docs:
            toks = tokenize('%s %s %s %s' % (d.get('title', ''), d.get('text', ''),
                                             ' '.join(d.get('knowledge_points', [])),
                                             d.get('module', '')))
            c = Counter(toks)
            self.tf.append(c)
            self.doc_len.append(sum(c.values()))
            for t in c:
                self.df[t] += 1
        self.N = len(self.docs)
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.k1, self.b = 1.5, 0.75

    def _score(self, q_toks, i):
        score = 0.0
        for t in q_toks:
            f = self.tf[i].get(t)
            if not f:
                continue
            n = self.df.get(t, 0)
            idf = math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))
            score += idf * (f * (self.k1 + 1)) / (
                f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl))
        return score

    # ---------------------------------------------------------------- 检索
    def retrieve(self, query, k=3, module=None, doc_ids=None):
        """返回 [(score, doc), ...]，已按分数降序。"""
        q = tokenize(query)
        cand = range(self.N)
        if module:
            cand = [i for i in range(self.N) if self.docs[i]['module'] == module]
        if doc_ids:
            cand = [i for i in range(self.N) if self.docs[i]['doc_id'] in set(doc_ids)]
        scored = [(self._score(q, i), i) for i in cand]
        scored.sort(key=lambda x: (-x[0], self.docs[x[1]]['doc_id']))
        out = [(s, self.docs[i]) for s, i in scored[:k] if s > 0]
        return out or [(0.0, self.docs[i]) for _, i in scored[:k]]

    def get(self, doc_id):
        return self.by_id.get(doc_id)

    def knowledge_points(self):
        """核心知识点集合（每条切片一个主知识点 = 其 title），用于覆盖率统计。"""
        return ['%s（%s）' % (d['title'], d['doc_id']) for d in self.docs]

    def stats(self):
        from collections import Counter as C
        return {
            '条目数': self.N,
            '模块数': len(set(d['module'] for d in self.docs)),
            '知识点标签数': len(set(k for d in self.docs
                                for k in d.get('knowledge_points', []))),
            '来源数': len(set(d.get('source', '') for d in self.docs)),
            '难度分布': dict(C(d.get('difficulty', '未标注') for d in self.docs)),
        }

    def citation(self, doc):
        """生成人类可读的溯源出处串。"""
        return '%s §%s（%s，%s）' % (doc['title'], doc.get('section', '-'),
                                    doc['source'], doc['doc_id'])


def load(path=None):
    return KnowledgeBase(path)


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    kb = load()
    print(json.dumps(kb.stats(), ensure_ascii=False, indent=2))
    for q in ['折扣因子应该取多大', '门控通信的稀疏度怎么算', '约束不可行怎么办']:
        print('\nQ:', q)
        for s, d in kb.retrieve(q, k=2):
            print('  %.2f  %s' % (s, kb.citation(d)))
