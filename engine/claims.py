# -*- coding: utf-8 -*-
"""声明级审核：把生成文本切分为**原子声明**，逐条检索取证并判定三态。

三态定义（与方案文档 §5.3 一致）
--------------------------------
SUPPORTED     —— 知识库存在支持证据，且无冲突证据
CONTRADICTED  —— 知识库存在与之矛盾的内容
UNVERIFIED    —— 知识库无足够证据支持亦无反驳

每一条声明都携带它取证用到的证据（doc_id + 出处），这是"高保真知识溯源"的实现：
审核结论不是模型的一句自我评价，而是可回溯到具体文档段落的结构化记录。
"""
import re

# 非断言行（标题、元信息、溯源行、审核标注）不参与审核
_SKIP_PREFIX = ('#', '|', '```', '>', '适用对象', '本资源面向', '面向学习者',
                '**溯源**', '溯源', '*文档版本', '证据出处', '【证据不足',
                '（证据不足', '本资源中有')
_SKIP_SECTIONS = ('证据出处', '溯源', '学习提示', '使用说明', '附', '审核说明')
_META_PAT = re.compile(r'^(适用对象|本资源面向|关于|说明)[：:]')
# 引用行（形如「标题 §3.3」）与占位句，不是断言
_CITE_PAT = re.compile(r'^[^。]{0,60}§[^。]{0,20}$')
# 以文档编号收尾的**出处行**（形如"解析：出处：《值分解方法（VDN / QMIX）》§4（KB-M1-05）。"）。
# 它指向的是元数据、不是领域事实，同样不是断言。
# 依据是实测：分阶测试题的解析行每份都这样收尾，共 21 条被当成"无法溯源的断言"
# 计入严口径幻觉率（14.02% 里有相当一部分是它们）—— 那衡量的是"我们没能给
# 自己的引用行找到证据"，而不是"资源里有不可溯源的知识"。
# 形状要求"冒号开头的字段 + 句末的文档编号"：即整句就是一个出处字段，
# 没有别的内容。所以"证据 KB-M1-05 指出 QMIX 使用单调性约束（KB-M1-05）。"
# 这类**在引用中作断言**的句子不受影响（它没有字段冒号）。
_CITE_TAIL_PAT = re.compile(r'^[^。：]{0,20}[：:][^。]{0,80}[（(]KB-M\d+-\d+[）)]。?$')


def _is_citation(s):
    """该片段是否只是一个出处字段（引用行），而不是断言。

    **必须按句判、不能只按行判**：生成器把解析与出处写在同一行
    （"**解析**：出处：《…》§4（KB-M1-05）。答案逐字取自知识库原文，可直接核对。"），
    按行判会整行漏过，那句出处照样进审核 —— 实测上一版就是这么漏掉的 21 条。
    """
    return bool(_CITE_PAT.match(s) or _CITE_TAIL_PAT.search(s))
_PLACEHOLDER = ('见上文核心概念', '无检索到证据', '暂无记录的典型误区')

# 短句下限：标题、标签、字段名等碎片不是断言，不参与审核。
# **但绝对化表述例外**：先按绝对化词表放行，再按长度过滤。
# 依据是单元测试暴露的真实缺口 —— "γ 越大越好。" 只有 7 个字符，
# 早期版本被长度规则**静默丢弃**，于是这条最典型的错误断言
# 既不进分子也不进分母，幻觉率被系统性低估；
# 而它恰恰是投影算子最该处置的那一类句子（命中 ABS 词表 ⇒ 必然被判冲突）。
MIN_CLAIM_LEN = 8


class Claim(object):
    __slots__ = ('cid', 'text', 'section', 'verdict', 'reason', 'evidence',
                 'origin', 'operator', 'final_text', 'edit_distance')

    def __init__(self, cid, text, section):
        self.cid = cid
        self.text = text
        self.section = section
        self.verdict = 'UNVERIFIED'
        self.reason = ''
        self.evidence = []          # [(doc_id, citation, score)]
        self.origin = text          # 投影前原文（审计链）
        self.operator = 'KEEP'      # KEEP / REPLACE / LIMIT / DELETE
        self.final_text = text
        self.edit_distance = 0

    @property
    def ok(self):
        return self.verdict == 'SUPPORTED'

    def to_dict(self):
        return {
            'cid': self.cid, 'section': self.section, 'text': self.text,
            'verdict': self.verdict, 'reason': self.reason,
            'evidence': [{'doc_id': e[0], 'citation': e[1], 'score': round(e[2], 2)}
                         for e in self.evidence],
            'operator': self.operator, 'final_text': self.final_text,
            'edit_distance': self.edit_distance,
        }


def split_claims(text):
    """把资源文本切分为原子声明列表。返回 [(section, claim_text), ...]。"""
    from .llm import ABS_WORDS
    out, section = [], '正文'
    for raw in text.split('\n'):
        line = raw.strip()
        if not line:
            continue
        if line.startswith('#'):
            section = line.lstrip('#').strip()
            continue
        if line.startswith(_SKIP_PREFIX) or _META_PAT.match(line):
            continue
        if any(sec in section for sec in _SKIP_SECTIONS):
            continue
        line = re.sub(r'^[-*]\s+', '', line)
        line = re.sub(r'^\d+[\.、]\s+', '', line)
        line = line.replace('**', '').strip()
        if not line or (len(line) < MIN_CLAIM_LEN
                        and not any(w in line for w in ABS_WORDS)):
            continue
        if _is_citation(line) or any(p in line for p in _PLACEHOLDER):
            continue
        for s in re.split(r'(?<=[。；!?！？])', line):
            s = s.strip()
            if not s or _is_citation(s):
                continue
            # 与整行同一准则：够长，或者虽短但命中绝对化词表（见 MIN_CLAIM_LEN 注释）
            if len(s) >= MIN_CLAIM_LEN or any(w in s for w in ABS_WORDS):
                out.append((section, s))
    return out


# 领域断言与"操作性/流程性表述"的分界：与证据的最高词元重叠率。
# 依据是对本系统**实际生成文本**的实测标定
# （回归测试见 tests/test_engine.py::TestClaimSplitting.test_domain_floor_calibration）：
#   流程性表述   0.19 ~ 0.39（"确认 Python 版本与依赖…"0.39、"按本知识点的最小示例…"0.31）
#   领域事实断言 0.54 ~ 0.65（"折扣因子 γ 应当取接近 1 的值…"0.54、"超参直接照搬…"0.65）
# 0.50 取在两组之间的空档里，两侧各留 ≥0.11 的余量。
DOMAIN_OVERLAP_FLOOR = 0.50

# 操作性祈使句的起首动词：句子以这些词起句 ⇒ 它是一条**流程指令**，
# 不是对领域事实的断言。这是"言语行为"层面的判据，不是词元层面的，
# 因此不会被"句子里恰好出现了某个知识点名词"所干扰。
# 词表按本系统生成器实际产出的流程句标定（见 llm.py::MockLLM._guide 的三个步骤句），
# 不做无依据的外扩。
_OPERATIONAL_VERBS = ('确认', '安装', '配置', '运行', '执行', '固定', '记录', '检查',
                      '保存', '导出', '启动', '调试', '打开', '查看', '填写', '跑通',
                      '搭好', '按', '先', '请')

# 程度/比较/全称性标记：祈使句里出现这些词，说明它不只是"让你做一步操作"，
# 而是在顺带断言一条（可能错的）领域属性 —— 此时**取消豁免**，照常审核。
# 这堵的是"祈使句式幻觉"这个漏洞："先检查学习率调大一些效果更好。"
# 以操作动词起句、不含绝对化词、重叠也低，早期版本会被当成流程句放过。
# 不含"最"：本系统流程句里有"最小示例""最常见"这类正常用语，会误伤。
_ASSERTIVE_MARKERS = ('更', '越', '完全', '彻底', '必然', '一定', '唯一', '总是', '永远')


def is_domain_claim(ctext, evidence, kb=None):
    """判断一句话是否构成**领域事实断言**，返回 (是否, 最高重叠, 理由)。

    为什么需要这个判定：规范化流程/元信息（"确认 Python 版本与依赖"、"本讲义的每条
    结论均可在文末核对"）不是对领域事实的断言，把它们计入幻觉率分母会让指标失去
    意义。因此它们被单独归为「非断言性文本」并从分母中排除，**且逐条记录、随
    审核明细一并输出**，不做任何隐式丢弃。

    疑问句同理：题目题干（"损失函数里折扣因子 γ 是不是越大越好？"）提出的是问题、
    没有作出任何断言，既不可能"幻觉"，也不该被投影改写 ——
    否则会出现"把题干改写成答案"这种荒谬结果。

    判定的**先后次序**（次序本身就是这套判据的全部，每一步都有实测依据）：
      1. 疑问句 ⇒ 非断言；
      2. 与证据重叠 ≥ 0.50 ⇒ 断言（这一档最硬：句子的词元有一半以上来自知识库原文，
         无论它长什么样，都是在复述知识库内容）；
      3. 命中绝对化词表 ⇒ 断言（兜底一）；
      4. 以操作性动词起句 ⇒ 非断言（见 _OPERATIONAL_VERBS）；
      5. 出现知识库知识点标签 ⇒ 断言（兜底二）。

    **第 4 步为什么必须排在第 5 步之前**（这是实测踩出来的坑）：
    兜底二在**整个知识库**的知识点标签里做子串匹配，而知识点标签多是名词短语
    （"随机种子"、"折扣因子"、"VDN"），于是"固定随机种子以保证可复现"
    "记录与「值分解方法（VDN / QMIX）」相关的读数"这类**纯流程指令**仅仅因为
    提到了一个术语就被拖进审核、判为 UNVERIFIED，最后计入严口径幻觉率 ——
    实测一份实操指南的三个步骤句全部如此，非断言性文本数 = 0。
    那不是"严格的指标"，那是把流程文字算成幻觉。

    第 4 步同时要堵住"**祈使句式幻觉**"：以操作动词起句、却顺带断言领域属性的句子
    （"先检查学习率调大一些效果更好。"、"固定随机种子后收敛曲线就完全一致了。"），
    若只看起首动词就会被当成流程句放过。因此第 4 步要求句子**不含程度/比较/全称
    标记**（_ASSERTIVE_MARKERS）才予豁免，否则照常审核。

    剩余已知边界（**方向明确，只会少算、不会多算幻觉**）：若一条祈使句断言了错误的
    领域属性，却既不含绝对化词、也不含程度标记、又没提任何知识点名词、与证据重叠还
    低于 0.50 —— 它仍会被归入非断言性。该侧的偏差是保守的：只会让幻觉率偏低，
    不会凭空造出幻觉；本系统的生成器里祈使句只用于流程步骤，故当前产物不触发。
    """
    from .kb import tokenize
    from .llm import ABS_WORDS

    if ctext.rstrip().endswith(('？', '?')):
        return False, 0.0, '疑问句：提出问题而不作断言 ⇒ 不计入幻觉率、不参与改写'

    c_tok = set(tokenize(ctext))
    best = 0.0
    for d in evidence:
        d_tok = set(tokenize(d['text']))
        if not c_tok:
            continue
        best = max(best, len(c_tok & d_tok) / float(len(c_tok)))
    if best >= DOMAIN_OVERLAP_FLOOR:
        return True, best, '与知识库证据重叠 %.2f ≥ %.2f' % (best, DOMAIN_OVERLAP_FLOOR)
    hit = [w for w in ABS_WORDS if w in ctext]
    if hit:
        return True, best, '命中绝对化词表 %s（兜底：一律审核）' % '/'.join(hit)
    for v in _OPERATIONAL_VERBS:
        if ctext.startswith(v):
            m = [w for w in _ASSERTIVE_MARKERS if w in ctext]
            if m:
                return True, best, ('以操作性动词「%s」起句但含程度/全称标记 %s ⇒ '
                                    '句子在顺带断言领域属性，取消豁免' % (v, '/'.join(m)))
            return False, best, ('以操作性动词「%s」起句 ⇒ 流程指令，'
                                 '不是对领域事实的断言' % v)
    if kb is not None:
        for d in kb.docs:
            for k in d.get('knowledge_points', []):
                if k in ctext:
                    return True, best, '出现知识点标签「%s」（兜底：一律审核）' % k
    return False, best, '与证据最高重叠仅 %.2f 且无领域术语 ⇒ 属操作性/流程性表述' % best


def audit(text, kb, llm, topk=3, extra_doc_ids=None, split_non_domain=True):
    """对一段资源文本做声明级审核，返回 (claims, non_domain)。

    topk              每条声明检索几条证据
    extra_doc_ids     若指定，先在这些文档内检索（用于"补充检索后重审"）
    split_non_domain  是否区分「领域断言」与「非断言性文本」（默认区分）
    """
    claims, non_domain = [], []
    for i, (section, ctext) in enumerate(split_claims(text), 1):
        cand = kb.retrieve(ctext, k=topk, doc_ids=extra_doc_ids) \
            if extra_doc_ids else kb.retrieve(ctext, k=topk)
        evidence = [d for _, d in cand]
        ok, ov, why = is_domain_claim(ctext, evidence, kb) if split_non_domain \
            else (True, 0.0, '未启用区分')
        if not ok:
            non_domain.append({'section': section, 'text': ctext,
                               '最高重叠': round(ov, 4), '归类理由': why})
            continue
        c = Claim('C%03d' % (len(claims) + 1), ctext, section)
        c.evidence = [(d['doc_id'], kb.citation(d), s) for s, d in cand]
        c.verdict, c.reason = llm.judge(ctext, evidence)
        c.final_text = ctext
        claims.append(c)
    return claims, non_domain


def summarize_delivered(claims):
    """投影**之后**的口径：只在**实际交付出去**的声明上统计。

    这一点必须写清楚：被投影删除的声明已经不在交付文本里了，
    把它算进分母会把"投影把错误删掉了"误读成"错误还在"。
    只有留在交付文本里的声明，才谈得上"读者会读到它"。
    """
    kept = [c for c in claims if c.operator != 'DELETE']
    n = len(kept)
    if not n:
        return {'交付声明数': 0, '投影处置数': len(claims),
                '幻觉率_宽口径(冲突/交付)': 0.0, '幻觉率_严口径(交付)': 0.0}
    from collections import Counter
    cnt = Counter(c.verdict for c in kept)
    return {
        '交付声明数': n,
        '投影处置数': len(claims) - n,
        'SUPPORTED': cnt.get('SUPPORTED', 0),
        'CONTRADICTED': cnt.get('CONTRADICTED', 0),
        'UNVERIFIED': cnt.get('UNVERIFIED', 0),
        '幻觉率_宽口径(冲突/交付)': round(cnt.get('CONTRADICTED', 0) / float(n), 4),
        '幻觉率_严口径(交付)': round(
            (cnt.get('CONTRADICTED', 0) + cnt.get('UNVERIFIED', 0)) / float(n), 4),
    }


def summarize(claims, non_domain=None):
    """投影前的三态汇总。分母 = **领域断言**条数（非断言性文本单独报告，不入分母）。

    无论声明数是否为 0，返回的键集合始终一致 —— 下游（报告、评测、单元测试）
    因此不需要对空资源做特判，也就不会在某个画像上静默崩掉。
    """
    from collections import Counter
    n = len(claims)
    nd = len(non_domain or [])
    cnt = Counter(c.verdict for c in claims)
    return {
        '声明总数': n,
        '非断言性文本_不计入': nd,
        'SUPPORTED': cnt.get('SUPPORTED', 0),
        'CONTRADICTED': cnt.get('CONTRADICTED', 0),
        'UNVERIFIED': cnt.get('UNVERIFIED', 0),
        # 宽口径：仅冲突
        '幻觉率_宽口径(冲突/总数)':
            round(cnt.get('CONTRADICTED', 0) / float(n), 4) if n else 0.0,
        # 严口径：冲突 + 无法溯源
        '幻觉率_严口径((冲突+未溯源)/总数)':
            round((cnt.get('CONTRADICTED', 0) + cnt.get('UNVERIFIED', 0))
                  / float(n), 4) if n else 0.0,
    }


if __name__ == '__main__':
    import sys
    import json
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from .kb import load
    from .llm import get_backend
    kb, llm = load(), get_backend('mock')
    ev = [d for _, d in kb.retrieve('折扣因子的作用与取值', k=2)]
    prof = {'profile_id': 'L001', 'education': '本科大三',
            'theory_score': '85/100', 'hands_on': '有 Python'}
    txt = llm.generate_resource('lecture', '折扣因子的作用与取值', '进阶', ev, prof)
    cl, nd = audit(txt, kb, llm)
    for c in cl:
        print('%-6s %-13s %s' % (c.cid, c.verdict, c.text[:46]))
    for d in nd:
        print('%-6s %-13s %s' % ('--', '非断言性', d['text'][:46]))
    print(json.dumps(summarize(cl, nd), ensure_ascii=False, indent=2))
