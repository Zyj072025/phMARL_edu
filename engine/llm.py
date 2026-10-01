# -*- coding: utf-8 -*-
"""可插拔 LLM 后端。

两个后端，接口完全一致，用 `--backend` 切换：

1. `mock`（默认，**离线**）
   确定性模板后端，零网络依赖。用途有两条，**必须分清**：
   （a）**机制回归夹具**：内置一组"常见专业误区"作为注入项，用于端到端验证
        "声明切分 → 取证判定 → 约束投影 → 溯源"这条链路确实能抓住并修正错误，
        并作为单元测试的 ground truth；
   （b）无网络环境下的演示与调试，保证任何机器上 `python run.py` 都能跑通。
   ⚠️ **mock 后端跑出来的幻觉率不是真实大模型的幻觉率**，它衡量的是"审核与投影
      机制的有效性"。真实指标必须用 `openai` 后端在真实模型上测量，两者在输出中
      会被明确区分标注（见 eval/run_eval.py 与 report）。

2. `openai`（真实大模型，OpenAI 兼容接口）
   通过环境变量提供 key，走 urllib（无需第三方 SDK）。同一套 prompt 与解析逻辑，
   换上它即得到可对外汇报的真实数值。
"""
import json
import os
import re
import urllib.error
import urllib.request
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- 误区夹具
# (doc_id, 注入的错误声明, 修正后的表述)
# 这些是**真实存在于该领域学习者中的典型误区**，用于让审核与投影机制有明确的抓手。
MISCONCEPTIONS = [
    ('KB-M1-08',
     '折扣因子 γ 应当尽可能取接近 1 的值，γ 越大越重视长期收益，因此 γ=0.999 总是最优选择。',
     '折扣因子 γ 的取值需要在"重视长期回报"与"控制价值估计方差与自举偏差"之间权衡：'
     'γ 越接近 1，长时序任务中的训练越容易不稳定，工程实践常见取值为 0.95～0.99，'
     '而不是把 γ 调到最大。'),
    ('KB-M1-04',
     'CTDE（集中训练分散执行）可以彻底消除多智能体环境中的非平稳性，'
     '因为训练时 Critic 能看到所有智能体的信息。',
     'CTDE 并不能消除非平稳性，它只是让集中式 Critic 在训练阶段能感知其他智能体的策略，'
     '从而缓解而非消除环境分布随其他智能体更新而变化的问题。'),
    ('KB-M2-07',
     '软更新系数 τ 建议取 0.5 左右，这样目标网络可以最快跟踪在线网络，训练最稳定。',
     '软更新系数 τ 越大，目标网络跟踪越快但稳定作用越弱；τ 过大（如 0.5 以上）'
     '会显著削弱目标网络的稳定作用，使价值估计明显波动，工程上常取 0.005～0.01。'),
    ('KB-M3-04',
     '消息维度越大，智能体之间传递的信息越完整，通信效果一定越好，因此无需做消息压缩。',
     '消息维度不是越大越好：过度压缩会丢失关键信息，压缩不足则收益有限，'
     '应当通过"压缩率—任务性能"的对照实验来确定消息维度。'),
    ('KB-M4-04',
     '当安全滤波器的约束不可行时，可以直接跳过该步的约束检查、按原动作执行，'
     '这样能避免系统报错中断。',
     '约束不可行时必须定义明确且可审计的降级策略（放松约束、退化为保守动作、'
     '或拒绝该步决策并报错），绝不能静默返回未经验证的结果。'),
    ('KB-M4-05',
     '安全滤波器的介入率越高，说明滤波器工作越积极、安全性能越好。',
     '介入率需要与安全率一并解读：介入率为 100% 说明原策略完全被约束支配、'
     '每一步动作都被修改；介入率为 0 说明滤波器未起作用，'
     '两者都属于需要排查的异常信号，不能单独用介入率衡量滤波器优劣。'),
    ('KB-M1-05',
     'QMIX 使用求和的方式把联合价值函数分解为各智能体局部价值函数之和，'
     '因此它本质上与 VDN 完全相同。',
     'QMIX 并非求和分解：它使用带单调性约束（∂Q_tot/∂Q_i ≥ 0）的混合网络，'
     '在保证可分散执行（全局 argmax 与各智能体局部 argmax 一致）的同时，'
     '表达能力严格强于 VDN 的简单求和分解。'),
    ('KB-M2-05',
     '确定性策略梯度方法（DDPG 等）不需要额外的探索噪声，'
     '因为确定性策略本身在训练初期就会自动探索。',
     '确定性策略本身不产生探索，必须外挂探索噪声（如高斯噪声或 OU 噪声），'
     '否则智能体会长期执行同一动作而无法充分探索状态空间。'),
    ('KB-M3-01',
     '可学习通信中的消息内容是有监督标签的，因此可以直接用监督学习训练通信模块。',
     '可学习通信的消息没有监督标签：消息内容与"是否发送"都是离散或连续的决策，'
     '只能通过任务回报反向传播来间接学习，这也是该方向的核心难点之一。'),
    ('KB-M5-03',
     '比较两个策略时，只要各自跑一次训练、比较曲线终点的平均奖励就足以得出结论。',
     '深度强化学习对随机种子与超参高度敏感，单次运行结果不具代表性；'
     '正确做法是控制环境随机性做配对比较（同一批固定初始条件），'
     '并报告均值±标准差与显著性检验。'),
]

FIXTURE = {d: {'wrong': w, 'corrected': c} for d, w, c in MISCONCEPTIONS}


def _stable_hash(s):
    """跨进程稳定的哈希（不能用内置 hash：PYTHONHASHSEED 会随机化）。"""
    return zlib.crc32(s.encode('utf-8'))


# 条目级对账的最低表述相似度。标定依据（见 tests/test_engine.py::
# TestMockBackend.test_fixture_alignment_separates_wrong_from_corrected）：
#   一条声明与"它自己那条"误区（逐字） 1.00
#   与同一对里**另一条**（错误 vs 修正） 0.04 ~ 0.21
#   与**其他**条目的误区（跨条目最大）   0.12
#   短句（"γ 越大越好。"）与各条误区      最大 0.23
# 0.70 落在空档里，因此"是不是那条误区的另一种说法"可以稳定判定。
FIXTURE_FLOOR = 0.70

# 「与证据实质一致」这条判据（judge 第 2 步）的最低词元数。
# 依据：本系统实际产出的领域断言，最短的是"工程上 τ 常取 0.005～0.01；"（11 个词元，
# 逐字来自知识库条目），而被判错的短句是"γ 越大越好。"（7 个词元）。
# 10 落在两者之间。**这不是为了调指标，是为了不让一个统计量在样本太小时失效**：
# |A∩B|/|A| 在 |A| 很小时必然接近 1（任何一句话的字符都会被长文档覆盖），
# 此时"实质性一致"这个结论根本无从成立 —— 宁可退回 UNVERIFIED 并计入严口径。
MIN_OVERLAP_TOKENS = 10


def _jaccard(a, b):
    """**对称**的表述相似度：词元 Jaccard 系数 ∈ [0,1]。

    必须对称，这是一个被单元测试抓到的真实缺陷。早期版本用的是
    "a 的词元有多少比例出现在 b 中"（|A∩B|/|A|），对**短句**会退化：
    "γ 越大越好。" 只有 {γ, 越大, 越好} 几个词元，随手一条含"越大…越好"
    的别的误区（消息压缩那条）就能拿到 0.83，于是这句话被判成"与消息压缩
    条目的修正表述一致"→ SUPPORTED。它明明是折扣因子那条误区，却因为句子短
    而"认领"了另一个学科的正确答案 —— 与 projection._corrective 修掉的是
    同一类"换题目"错误，只是发生在判定侧。
    Jaccard = |A∩B|/|A∪B| 对短句同样严格：上面那句与任何一条误区的相似度
    都不超过 0.23。而当 A ⊇ B（声明=前缀+误区原文逐字）时它退化为原来的
    方向性比值，因此"注入的误区必须被判冲突"这条仍然稳稳成立。
    """
    from .kb import tokenize
    A, B = set(tokenize(a)), set(tokenize(b))
    return len(A & B) / float(len(A | B)) if (A or B) else 0.0


# 绝对化/无条件化表述词表。判定的核心线索，**同时**被 claims.py 用作
# "这句话必须进入审核"的兜底条件（见 claims.is_domain_claim）——
# 两处共用一个定义，避免审核范围被悄悄放宽。
ABS_WORDS = ['越大越好', '总是最优', '一定越好', '彻底消除', '完全相同',
             '最好', '最优选择', '无需', '不需要', '直接跳过',
             '越高越好', '足以得出结论', '最快', '最稳定', '所有情况']


class LLMBackend:
    name = 'base'

    def generate_resource(self, kind, kp, difficulty, evidence, profile):
        raise NotImplementedError

    def revise(self, claim, evidence, corrected):
        raise NotImplementedError

    def judge(self, claim, evidence):
        """返回 (verdict, reason)：verdict ∈ {SUPPORTED, CONTRADICTED, UNVERIFIED}。"""
        raise NotImplementedError

    def info(self):
        return {'backend': self.name}


# ---------------------------------------------------------------- mock 后端
class MockLLM(LLMBackend):
    """离线确定性后端。见模块 docstring 中关于"夹具 vs 真实测量"的说明。"""
    name = 'mock'

    def __init__(self, inject_ratio=3, bank_path=None):
        # inject_ratio: 每 N 个知识点注入 1 条误区（确定性，保证消融各组同题同错）
        self.inject_ratio = max(1, int(inject_ratio))
        self.bank_path = bank_path or os.path.join(
            ROOT, 'data', 'bank', 'questions.json')
        self._bank = None

    # -------------------------------------------------------- 题库
    @property
    def bank(self):
        if self._bank is None:
            try:
                with open(self.bank_path, encoding='utf-8') as f:
                    self._bank = json.load(f)
            except FileNotFoundError:
                self._bank = []
        return self._bank

    def pick_question(self, kp, difficulty):
        for q in self.bank:
            if q['knowledge_point'] in kp and q['difficulty'] == difficulty:
                return q
        for q in self.bank:
            if q['knowledge_point'] in kp:
                return q
        return None

    # -------------------------------------------------------- 生成
    def generate_resource(self, kind, kp, difficulty, evidence, profile):
        if kind == 'lecture':
            return self._lecture(kp, difficulty, evidence, profile)
        if kind == 'guide':
            return self._guide(kp, difficulty, evidence, profile)
        if kind == 'quiz':
            return self._quiz(kp, difficulty, evidence, profile)
        raise ValueError('未知资源形态: %s' % kind)

    def _misconception_for(self, evidence, task_key):
        """确定性注入：优先使用证据里携带的 doc_id，按 task_key 决定是否注入。"""
        if _stable_hash(task_key) % self.inject_ratio:
            return None
        for doc in evidence:
            if doc['doc_id'] in FIXTURE:
                return FIXTURE[doc['doc_id']]
        return None

    def _persona_line(self, profile, difficulty):
        return ('本资源面向学习者 %s（%s｜理论测试 %s｜实操经历：%s），'
                '目标难度档：%s。' % (
                    profile.get('profile_id', '?'),
                    profile.get('education', '未填'),
                    profile.get('theory_score', '未填'),
                    profile.get('hands_on', '未填'),
                    difficulty))

    def _lecture(self, kp, difficulty, evidence, profile):
        ev = evidence[0] if evidence else None
        core = ev['text'] if ev else '（无检索到证据）'
        cite = ('%s §%s' % (ev['title'], ev.get('section', '-')) if ev else '-')
        # 注入位置刻意放在**正文断言**里：这正是大模型幻觉的真实发生形态——
        # 它不会把错误写进"常见误区"，而是当作事实写进正文。
        mis = self._misconception_for(evidence, 'lecture|' + kp)
        lines = ['# 定制讲义：%s' % kp, '',
                 self._persona_line(profile, difficulty), '',
                 '## 一、核心概念', core]
        if mis:
            lines.append(mis['wrong'])
        lines += ['', '## 二、要点速记']
        if ev:
            for k in ev.get('knowledge_points', [])[:5]:
                sents = [s.strip() for s in re.split(r'[。；]', ev['text'])
                         if k in s]
                if sents:
                    lines.append('- **%s**：%s。' % (k, sents[0]))
        else:
            lines.append('· 本知识点证据不足，暂不展开。')
        lines += ['', '## 三、学习提示',
                  '- 本讲义的每条结论均可在文末证据出处的原文中核对；'
                  '若发现与教材表述不一致，以证据原文为准。',
                  '', '## 四、证据出处', '- %s' % cite]
        return '\n'.join(lines)

    def _guide(self, kp, difficulty, evidence, profile):
        ev = evidence[0] if evidence else None
        mis = self._misconception_for(evidence, 'guide|' + kp)
        cite = ('%s §%s' % (ev['title'], ev.get('section', '-')) if ev else '-')
        lines = ['# 实操指南：%s（难度档：%s）' % (kp, difficulty), '',
                 '适用对象：%s' % profile.get('profile_id', '?'), '',
                 '## 步骤 1｜环境准备', '确认 Python 版本与依赖，固定随机种子以保证可复现。',
                 '',
                 '## 步骤 2｜最小可运行实现',
                 '按本知识点的最小示例搭好训练回路，先只跑 1 个随机种子的短程配置，'
                 '确认日志与评估脚本都能产出指标。', '',
                 '## 步骤 3｜观察关键读数',
                 '记录与「%s」直接相关的读数（而不是只看总奖励），'
                 '确认读数随训练的变化方向符合预期。' % kp, '',
                 '## 步骤 4｜常见坑']
        if mis:
            lines.append('- ⚠️ 高频错误：%s' % mis['wrong'])
        lines.append('- 超参直接照搬其他环境而不做标定，是训练失败的最常见根因。')
        lines += ['', '## 溯源', '- %s' % cite]
        return '\n'.join(lines)

    def _quiz(self, kp, difficulty, evidence, profile):
        q = self.pick_question(kp, difficulty)
        if q is None:
            return ('# 分阶测试题：%s（%s）\n\n'
                    '（题库中该知识点暂无对应难度题目，由生成 Agent 出题：'
                    '请说明 %s 的定义、适用条件与常见误区。）'
                    % (kp, difficulty, kp))
        return ('# 分阶测试题：%s（%s）\n\n**题干**：%s\n\n'
                '**参考答案**：%s\n\n**解析**：%s\n\n**溯源**：%s'
                % (kp, difficulty, q['stem'], q['answer'], q['explain'],
                   q['source']))

    # -------------------------------------------------------- 修正与判定
    def revise(self, claim, evidence, corrected):
        return corrected

    def judge(self, claim, evidence):
        """离线判定器。判据的执行顺序本身就是结论的一部分：

          1. 与某条已知误区的表述高度一致      → CONTRADICTED（条目级对账）
             与该条目的修正表述高度一致        → SUPPORTED
          2. 与证据文本实质一致（重叠 ≥ 0.62）→ SUPPORTED
          3. 命中"绝对化/无条件化"表述词表    → CONTRADICTED
          4. 其余                              → UNVERIFIED

        第 2 步必须排在第 3 步前面 —— 详见方法体内注释。
        """
        from .kb import tokenize
        c_tok = set(tokenize(claim))
        best, best_doc = 0.0, None
        for d in evidence:
            d_tok = set(tokenize(d['text']))
            if not c_tok:
                continue
            ov = len(c_tok & d_tok) / float(len(c_tok))
            if ov > best:
                best, best_doc = ov, d
        if best_doc is None:
            return 'UNVERIFIED', '无检索证据'
        # ---- 1. 夹具对账（条目级） ----
        # 真实后端下这一步由 LLM 的语义判定完成（"这句话是不是那条已知误区的
        # 另一种说法"），离线夹具用表述相似度代做。**因此离线消融测的是
        # 流水线能否抓住并修正，不是判定器的识别能力**（见 README「已知局限」）。
        fx_best, fx_kind, fx_doc = 0.0, None, None
        for d in evidence:
            fx = FIXTURE.get(d['doc_id'])
            if not fx:
                continue
            ow = _jaccard(claim, fx['wrong'])
            oc = _jaccard(claim, fx['corrected'])
            if ow >= FIXTURE_FLOOR and ow >= oc and ow > fx_best:
                fx_best, fx_kind, fx_doc = ow, 'wrong', d
            if oc >= FIXTURE_FLOOR and oc > ow and oc > fx_best:
                fx_best, fx_kind, fx_doc = oc, 'corrected', d
        if fx_kind == 'wrong':
            return ('CONTRADICTED',
                    '与知识库条目「%s」记录的领域误区表述一致'
                    '（条目级对账 %.2f）' % (fx_doc['title'], fx_best))
        if fx_kind == 'corrected':
            return ('SUPPORTED',
                    '与知识库条目「%s」的修正表述一致（条目级对账 %.2f）'
                    % (fx_doc['title'], fx_best))

        # ---- 2. 与证据实质一致（必须放在绝对化判定之前） ----
        # 知识库原文本身可能写着"而不是默认越大越好"这类限定性表述：
        # 先按重叠判为一致，才不会把证据原文误判成幻觉。
        #
        # 但这一步的方向性重叠（|A∩B|/|A|）**只对足够长的声明成立**：
        # 对"γ 越大越好。"这种极短句，|A| 只有一两个词元，
        # 只要证据文档里恰好出现过"越大越好"，比值就是 1.00，
        # 于是一条典型的折扣因子误区会因为消息压缩的条目原文里
        # 写着"不是越大越好"而被判成 SUPPORTED（真实踩到过，见下方测试）。
        # 短语料不足以支撑"实质性一致"这个结论 ⇒ 交给第 3 步的绝对化词表。
        if best >= 0.62 and len(c_tok) >= MIN_OVERLAP_TOKENS:
            return ('SUPPORTED', '与证据「%s」表述一致（重叠 %.2f）'
                    % (best_doc['title'], best))

        # ---- 3. 绝对化/无条件化表述 ----
        # 只对"既非夹具误区、也不与证据一致"的声明生效 ——
        # 这正是幻觉最典型的形态：无来源的绝对断言。
        hit = [w for w in ABS_WORDS if w in claim]
        if hit:
            return ('CONTRADICTED',
                    '声明含绝对化表述 %s，与证据「%s」的限定性表述冲突'
                    % ('/'.join(hit), best_doc['title']))

        return ('UNVERIFIED', '重叠仅 %.2f，证据不足以判定' % best)


# ---------------------------------------------------------------- 真实后端
class OpenAICompatLLM(LLMBackend):
    """OpenAI 兼容接口后端（urllib 实现，无需第三方 SDK）。

    环境变量：PHMARL_LLM_API_KEY / PHMARL_LLM_BASE_URL / PHMARL_LLM_MODEL
    或命令行 --api-key / --base-url / --model。
    """
    name = 'openai'

    SYS = ('你是垂直领域（人工智能/具身智能/多智能体系统）的资深教学资源生成专家。'
           '严格基于给定的【知识库证据】撰写，不得引入证据之外的断言；'
           '无法从证据中确认的内容必须显式写"证据不足"。')

    def __init__(self, api_key=None, base_url=None, model=None, timeout=60):
        self.api_key = api_key or os.environ.get('PHMARL_LLM_API_KEY', '')
        self.base_url = (base_url or os.environ.get(
            'PHMARL_LLM_BASE_URL', 'https://api.openai.com/v1')).rstrip('/')
        self.model = model or os.environ.get('PHMARL_LLM_MODEL', 'gpt-4o-mini')
        self.timeout = timeout

    def available(self):
        return bool(self.api_key)

    def _chat(self, user_prompt, temperature=0.3):
        if not self.api_key:
            raise RuntimeError(
                '未提供 API key：请设置环境变量 PHMARL_LLM_API_KEY，或用 --backend mock 离线运行。')
        body = json.dumps({
            'model': self.model, 'temperature': temperature,
            'messages': [{'role': 'system', 'content': self.SYS},
                         {'role': 'user', 'content': user_prompt}],
        }).encode('utf-8')
        req = urllib.request.Request(
            self.base_url + '/chat/completions', data=body,
            headers={'Content-Type': 'application/json',
                     'Authorization': 'Bearer ' + self.api_key})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return data['choices'][0]['message']['content']

    def _evidence_block(self, evidence):
        return '\n'.join('[%s] %s §%s：%s' % (d['doc_id'], d['title'],
                                             d.get('section', '-'), d['text'])
                         for d in evidence)

    def generate_resource(self, kind, kp, difficulty, evidence, profile):
        name = {'lecture': '定制讲义', 'guide': '实操指南', 'quiz': '分阶测试题'}[kind]
        prompt = (
            '【学习者画像】%s\n【知识点】%s\n【目标难度】%s\n'
            '【知识库证据】\n%s\n\n'
            '请生成一份「%s」。要求：分小节、结构清晰、面向该学习者的基础水平；'
            '每条关键结论后用 [doc_id] 标注其证据出处；'
            '证据无法支持的内容写"证据不足"，不要臆测。'
            % (json.dumps(profile, ensure_ascii=False), kp, difficulty,
               self._evidence_block(evidence), name))
        return self._chat(prompt)

    def revise(self, claim, evidence, corrected):
        prompt = (
            '【待修正声明】%s\n【知识库证据】\n%s\n\n'
            '请对该声明做**最小改动**的改写，使其与证据完全一致：'
            '保留原有的表述风格与信息意图，只修改与证据冲突的部分；'
            '只输出改写后的这一句话。' % (claim, self._evidence_block(evidence)))
        return self._chat(prompt, temperature=0.1).strip()

    def judge(self, claim, evidence):
        prompt = (
            '【待判定声明】%s\n【知识库证据】\n%s\n\n'
            '请判定该声明相对证据属于以下哪一类，只输出一行：\n'
            'SUPPORTED —— 证据支持该声明\n'
            'CONTRADICTED —— 证据与该声明冲突\n'
            'UNVERIFIED —— 证据不足以判定\n'
            '格式：<判定>|<一句话理由>'
            % (claim, self._evidence_block(evidence)))
        out = self._chat(prompt, temperature=0.0).strip()
        m = re.match(r'\s*(SUPPORTED|CONTRADICTED|UNVERIFIED)\s*\|?\s*(.*)',
                     out, re.S)
        if not m:
            return 'UNVERIFIED', '后端返回不可解析：%s' % out[:60]
        return m.group(1), (m.group(2).strip() or '大模型判定')


def get_backend(name='mock', **kw):
    if name == 'mock':
        return MockLLM(**kw)
    if name in ('openai', 'openai-compatible'):
        return OpenAICompatLLM(**kw)
    raise ValueError('未知后端: %s（可选 mock / openai）' % name)


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from .kb import load
    kb = load()
    llm = get_backend('mock')
    ev = [d for _, d in kb.retrieve('折扣因子取值', k=1)]
    prof = {'profile_id': 'L001', 'education': '本科大三',
            'theory_score': '85/100', 'hands_on': '有 Python，无 RL 经验'}
    for k in ('lecture', 'guide', 'quiz'):
        print('=' * 70)
        print(llm.generate_resource(k, '折扣因子的作用与取值', '进阶', ev, prof))
    print('=' * 70)
    v, r = llm.judge('软更新系数 τ 建议取 0.5 左右，这样目标网络跟踪最快、训练最稳定。',
                     [d for _, d in kb.retrieve('软更新系数取值', k=1)])
    print('判定:', v, '|', r)
