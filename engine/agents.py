# -*- coding: utf-8 -*-
"""四个职责明确的智能体。

A1 学情诊断 Agent   —— 画像 → 知识点掌握度估计 + 知识盲区 + 能力表征 z
A2 领域知识生成 Agent —— 诊断结论 + 知识库证据 → 三类个性化资源（带引用）
A3 审核纠偏 Agent   —— 声明级取证判定 + 双视角交叉验证与辩论 + 知识约束投影
A4 协同调度 Agent   —— 门控通信判决 + 教学目标/难度/动作决策

每个 Agent 只暴露一个语义清晰的方法，便于单独单元测试与在 UI 上逐步可视化。
"""
import re

from . import claims as CLAIMS
from . import projection as PROJ
from .gating import impact_of

DIFF_ORDER = ['基础', '进阶', '挑战']
DIFF_REQUIRED = {'基础': 0.30, '进阶': 0.55, '挑战': 0.75}   # 掌握度门槛
HANDS_LEVEL = {
    '无': 0.0, '无编程实操': 0.0, '初级': 0.25, '有 Python，无 RL 经验': 0.25,
    '有 Python，无 RL': 0.25, '中级': 0.5, '做过单智能体 RL': 0.5,
    '高级': 0.75, '做过多智能体项目': 0.75,
}
BLIND_THRESHOLD = 0.60


def _sigmoid(x):
    import math
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _num(v, default=0.0):
    """宽容解析画像里的数值字段：支持 62、'62'、'62/100'、62.0、'62%'。"""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace('%', '')
    if '/' in s:
        a, b = s.split('/', 1)
        try:
            return float(a) / float(b)
        except (ValueError, ZeroDivisionError):
            return default
    try:
        return float(s)
    except ValueError:
        return default


def _ratio(v, default):
    """把画像里的比率字段统一归一化到 [0,1]：0.62 / '62%' / '62/100' / 62 都认。"""
    x = _num(v, None)
    if x is None:
        return default
    if x > 1.5:
        x = x / 100.0
    return min(1.0, max(0.0, x))


# ══════════════════════════════════════════════════════════════════ A1
class DiagnoserAgent(object):
    name = 'A1 学情诊断'

    def __init__(self, kb, llm=None):
        self.kb = kb
        self.llm = llm

    def diagnose(self, profile):
        """画像 → {mastery, blind_spots, z, ...}

        掌握度估计口径（**规则基线**，可替换为 BKT/IRT 等知识追踪模型）：
            skill   = 0.55·理论分 + 0.30·实操经历 + 0.15·历史正确率
            p̂_k     = σ( 6·(skill − 该知识点的难度门槛) )
        其中历史正确率缺省时按 0.5 处理，并在输出中标注为"未知"。
        """
        theory = _ratio(profile.get('theory_score'), 0.5)
        hands = HANDS_LEVEL.get(str(profile.get('hands_on', '')).strip(), 0.25)
        hist = profile.get('history_accuracy')
        hist_known = hist is not None
        hist = 0.5 if hist is None else _ratio(hist, 0.5)
        skill = 0.55 * theory + 0.30 * hands + 0.15 * hist

        mastery = {}
        for d in self.kb.docs:
            req = DIFF_REQUIRED.get(d.get('difficulty', '进阶'), 0.55)
            mastery[d['title']] = round(_sigmoid(6.0 * (skill - req)), 4)

        blind = sorted([(kp, p) for kp, p in mastery.items()
                        if p < BLIND_THRESHOLD], key=lambda x: x[1])
        z = [round(theory, 3), round(hands, 3),
             round(hist, 3) if hist_known else 0.5,
             round(sum(mastery.values()) / max(len(mastery), 1), 3)]
        return {
            'agent': self.name,
            'profile_id': profile.get('profile_id'),
            'skill_level': round(skill, 4),
            'history_accuracy': hist if hist_known else None,
            'mastery': mastery,
            'blind_spots': [{'knowledge_point': kp, 'mastery': p} for kp, p in blind],
            'representation_z': z,
            '说明': ('掌握度由画像特征的规则基线估计得到；'
                     '历史正确率缺失时按 0.5 处理并标注为未知。'),
        }


# ══════════════════════════════════════════════════════════════════ A2
class GeneratorAgent(object):
    name = 'A2 领域知识生成'

    RESOURCE_KINDS = [('lecture', '定制讲义'), ('guide', '实操指南'), ('quiz', '分阶测试题')]

    def __init__(self, kb, llm):
        self.kb = kb
        self.llm = llm

    def retrieve(self, kp, topk=2):
        """按知识点检索证据。返回 [(score, doc)]。"""
        return self.kb.retrieve(kp, k=topk)

    def generate(self, kp, difficulty, profile, topk=2):
        """生成一个知识点的三类资源。返回 dict（含证据出处，供审核与溯源）。"""
        cand = self.retrieve(kp, topk=topk)
        evidence = [d for _, d in cand]
        resources = []
        for kind, label in self.RESOURCE_KINDS:
            text = self.llm.generate_resource(kind, kp, difficulty, evidence, profile)
            resources.append({
                'kind': kind, 'label': label, 'knowledge_point': kp,
                'difficulty': difficulty, 'text_raw': text,
                'evidence': [{'doc_id': d['doc_id'], 'citation': self.kb.citation(d),
                              'score': round(s, 2)} for s, d in cand],
            })
        return {'knowledge_point': kp, 'difficulty': difficulty,
                'agent': self.name, 'resources': resources}


# ══════════════════════════════════════════════════════════════════ A3
class AuditorAgent(object):
    name = 'A3 审核纠偏'

    def __init__(self, kb, llm, eps=0.05):
        self.kb = kb
        self.llm = llm
        self.eps = eps

    # ---- 两个独立视角：检索策略不同 ⇒ 判定偏差不同源 ----
    # 两个方法都返回 (判定, 理由, 证据列表)，**理由必须原样带出**：
    # 三态判定若没有理由，交叉验证的分歧就无法解释，裁决就成了黑箱。
    def _view_a(self, claim_text, topk=3):
        """视角 A：BM25 全文检索取证（不限定模块，召回优先）。"""
        cand = self.kb.retrieve(claim_text, k=topk)
        ev = [d for _, d in cand]
        v, r = self.llm.judge(claim_text, ev)
        return v, r, ev

    def _view_b(self, claim_text, topk=3):
        """视角 B：按知识点标签精确命中，再在命中条目**所属模块内**取证。

        与视角 A 的关键差别：A 走全文检索（可能跨模块召回），
        B 先做标签匹配把候选**锁在知识点所在的模块**，再在该模块内检索。
        两条路线的取证范围不同 ⇒ 判定偏差不同源，才构成真正的交叉验证。
        """
        hits = []
        for d in self.kb.docs:
            if any(k in claim_text for k in d.get('knowledge_points', [])):
                hits.append(d)
        if hits:
            mod = hits[0].get('module')
            cand = self.kb.retrieve(claim_text, k=topk, module=mod)
        else:
            cand = self.kb.retrieve(claim_text, k=topk)
        ev = [d for _, d in cand]
        v, r = self.llm.judge(claim_text, ev)
        return v, r, ev

    def audit(self, text, kb_doc_hint=None):
        """声明级审核：视角 A 判定为主，视角 B 交叉验证，分歧走第三视角裁决。

        返回 (claims, disputes, non_domain)：
        non_domain 是被判为「操作性/流程性表述」而不计入幻觉率分母的句子，
        逐条带理由返回，由调用方原样写进审核明细（不做隐式丢弃）。
        """
        claim_list, non_domain = CLAIMS.audit(text, self.kb, self.llm, topk=3,
                                              extra_doc_ids=None)
        disputes = []
        for c in claim_list:
            vA, rA = c.verdict, c.reason
            vB, rB, evB = self._view_b(c.text)
            if vA != vB:
                # 第三视角裁决：合并两方证据后重判（并记录裁决理由）
                ev_ids = [e[0] for e in c.evidence]
                evB_ids = [d['doc_id'] for d in evB]
                union = sorted(set(ev_ids) | set(evB_ids))
                union_ev = [self.kb.get(i) for i in union if self.kb.get(i)]
                vC, reasonC = self.llm.judge(c.text, union_ev)
                disputes.append({
                    'cid': c.cid, '声明': c.text, '视角A': vA, '视角A理由': rA,
                    '视角B': vB, '视角B理由': rB, '裁决': vC, '裁决理由': reasonC,
                    '裁决证据': union,
                })
                c.verdict = vC
                c.reason = '（三视角裁决：A=%s / B=%s）%s' % (vA, vB, reasonC)
        return claim_list, disputes, non_domain

    def project(self, claim_list):
        return PROJ.project(claim_list, self.kb, self.llm, eps=self.eps)

    def review(self, text):
        """一轮完整审核：取证判定 → 交叉验证/辩论 → 约束投影 → 交付文本。

        顺序很关键：**投影前的三态统计必须在投影之前快照**，
        否则投影会就地改写 claim.verdict，投影前的幻觉率会被覆盖成投影后的值，
        整条"投影把幻觉率压到多少"的因果链就断了。
        """
        claim_list, disputes, non_domain = self.audit(text)
        pre_summary = CLAIMS.summarize(claim_list, non_domain)   # ← 投影前快照
        pre_claims = [c.to_dict() for c in claim_list]
        records, stats, feasible = self.project(claim_list)
        delivered = PROJ.apply_to_text(text, claim_list)
        return {
            'agent': self.name,
            'pre_audit': pre_summary,                        # 投影前
            'post_audit': CLAIMS.summarize_delivered(claim_list),   # 投影后（仅交付声明）
            'non_domain': non_domain,
            'claims_pre': pre_claims,
            'claims': [c.to_dict() for c in claim_list],
            'claims_obj': claim_list,
            'disputes': disputes,
            'projection_records': records,
            'projection_stats': stats,
            'feasible': feasible,
            'text_delivered': delivered,
        }


# ══════════════════════════════════════════════════════════════════ A4
class SchedulerAgent(object):
    name = 'A4 协同调度'

    # 「资源支撑系数」与编排器的学习者响应模型**共用同一张表**：
    # 调度方用它**预测**作答正确率，据此选难度档。真实系统里这张表由历史
    # 作答数据标定（而不是写死），本版本用同一组常数以便离线复现。
    FACTOR = {'基础': 1.45, '进阶': 1.10, '挑战': 0.80}
    BAND = (0.60, 0.80)          # 心流区间
    # 可及范围下沿 = 心流区间下沿。这两条线必须重合：低于区间的知识点即使投放到
    # 最基础档也进不了心流区间，投出去只会制造挫败（这正是"最近发展区"的本意）。
    # 早期版本取 0.55（低于区间下沿 0.60），于是系统会稳定投出**必然不适配**的项，
    # 适配率被自己的策略压低 —— 见 eval/oracle_ceiling.py。
    REACH_FLOOR = 0.60           # 预测正确率低于此值 ⇒ 该知识点超出当前可及范围

    def __init__(self, gate, budget_kp=2, kb=None):
        self.gate = gate
        self.budget_kp = budget_kp
        self.kb = kb
        self.kp_diff = {}
        if kb is not None:
            for d in kb.docs:
                self.kp_diff[d['title']] = d.get('difficulty', '进阶')

    # ---- 难度决策：直接选"预测正确率最接近心流区间中心"的那一档 ----
    @classmethod
    def pick_difficulty(cls, p):
        """返回 (难度档, 预测正确率)。"""
        best, best_gap = '基础', 1e9
        for d in DIFF_ORDER:
            rate = min(0.97, max(0.02, p * cls.FACTOR[d]))
            gap = abs(rate - 0.70)              # 以区间中心 0.70 为目标
            if gap < best_gap:
                best, best_gap = d, gap
        return best, min(0.97, max(0.02, p * cls.FACTOR[best]))

    @classmethod
    def shift(cls, difficulty, delta):
        """难度档升降一档，顶/底封口。"""
        i = DIFF_ORDER.index(difficulty) if difficulty in DIFF_ORDER else 1
        return DIFF_ORDER[max(0, min(len(DIFF_ORDER) - 1, i + delta))]

    # ---- 教学动作决策（基于答题反馈） ----
    @classmethod
    def next_action(cls, correct_rate, attempts):
        if correct_rate < 0.5:
            return '降维解释' if attempts > 1 else '补充前置知识'
        if correct_rate < cls.BAND[0]:
            return '降维解释'
        if correct_rate <= cls.BAND[1]:
            return '维持当前难度'
        return '进阶挑战'

    def _difficulty_of(self, kp):
        return self.kp_diff.get(kp, '进阶')

    def plan_round(self, diagnosis, round_idx=1, prev_feedback=None):
        """选定本轮目标知识点与难度，并用门控判决是否把诊断结论发给生成方。

        **选点规则（最近发展区优先，ZPD-first）**，两步：
          1. 过滤：按掌握度估计预测该知识点在心流区间内的可及难度，
             若即使给到「基础」档、预测正确率仍低于 REACH_FLOOR，
             说明该知识点**超出当前可及范围**（缺的是前置知识）⇒ 本轮不推它，
             改推同一模块内更基础、已进入可及范围的知识点；
          2. 排序：在可及的知识点里，取**掌握度最低**的那个优先。
        这样既保证"先补最大的洞"，又保证"补的洞是够得着的"。

        第五步「反馈」在这里**真正生效**：若上一轮某知识点的实测正确率
        低于心流下沿（0.60），本轮该知识点降一档难度（降维解释）；
        高于上沿（0.80），升一档（进阶挑战）——这正是榜题要求的
        "依据反馈做降维解释或进阶挑战"，而不是只把动作写进日志。
        """
        blind = diagnosis['blind_spots']
        prev = {}
        for x in (prev_feedback or {}).get('per_kp', []) or []:
            prev[x['knowledge_point']] = x

        # 候选集：优先盲区；盲区不足时用全库掌握度最低的知识点补齐。
        # 这一步是必需的：对高水平学习者（盲区为 0）系统不能"无事可做"，
        # 它应当转为**能力拓展**——推送其尚未精通的最难知识点。
        cand, seen = [], set()
        for b in blind:
            cand.append({'knowledge_point': b['knowledge_point'],
                         'mastery': b['mastery'], '目标类型': '盲区补强'})
            seen.add(b['knowledge_point'])
        if len(cand) < max(self.budget_kp, 1):
            rest = sorted(((k, v) for k, v in diagnosis['mastery'].items()
                           if k not in seen), key=lambda x: x[1])
            for k, v in rest:
                cand.append({'knowledge_point': k, 'mastery': v,
                             '目标类型': '能力拓展（无盲区，推送尚未精通的最难知识点）'})
                if len(cand) >= max(self.budget_kp, 1) * 3:
                    break

        reachable, deferred = [], []
        for b in cand:
            kp, p = b['knowledge_point'], b['mastery']
            diff, rate = self.pick_difficulty(p)
            row = {'knowledge_point': kp, 'mastery': p,
                   'difficulty_base': diff, '预测正确率': round(rate, 4),
                   '知识点台阶': self._difficulty_of(kp),
                   '目标类型': b['目标类型']}
            if rate < self.REACH_FLOOR:
                row['暂缓理由'] = ('掌握度 %.3f，即使投放到「基础」档预测正确率仍仅 '
                                '%.2f < %.2f ⇒ 超出可及范围，需先补前置知识'
                                % (p, rate, self.REACH_FLOOR))
                deferred.append(row)
            else:
                reachable.append(row)
        # 可及范围内：掌握度最低者优先；同掌握度时更基础的知识点优先
        reachable.sort(key=lambda r: (r['mastery'],
                                     DIFF_ORDER.index(r['知识点台阶'])
                                     if r['知识点台阶'] in DIFF_ORDER else 1))

        targets, level_changing, adjusts = [], 0, []
        for row in reachable[:self.budget_kp]:
            kp, base = row['knowledge_point'], row['difficulty_base']
            diff, why = base, ('由掌握度估计 %.3f 选定「%s」档，'
                             '预测正确率 %.2f（心流区间目标中心 0.70）'
                             % (row['mastery'], base, row['预测正确率']))
            pv = prev.get(kp)
            if pv and pv.get('total'):
                rate = pv['correct'] / float(pv['total'])
                if rate < self.BAND[0]:
                    diff = self.shift(base, -1)
                    why = ('上一轮实测正确率 %.0f%% 低于心流下沿 60%% ⇒ 降维解释，'
                           '%s → %s' % (rate * 100, base, diff))
                elif rate > self.BAND[1]:
                    diff = self.shift(base, +1)
                    why = ('上一轮实测正确率 %.0f%% 高于心流上沿 80%% ⇒ 进阶挑战，'
                           '%s → %s' % (rate * 100, base, diff))
                else:
                    why = ('上一轮实测正确率 %.0f%% 落在心流区间 ⇒ 维持 %s'
                           % (rate * 100, diff))
            if diff != '进阶':
                level_changing += 1
            if diff != base:
                adjusts.append({'知识点': kp, '原难度': base, '新难度': diff,
                                '依据': why})
            row = dict(row)
            row.update({'difficulty': diff, 'difficulty_依据': why})
            targets.append(row)
        payload = {'targets': targets, 'round': round_idx,
                   'prev_feedback': prev_feedback}
        msg = self.gate.consider(
            'diagnosis', 'A1 学情诊断', 'A2 领域知识生成', payload,
            ctx={'n_targets': len(targets), 'n_level_changing': level_changing})
        # 门控抑制时：生成方按默认档位工作（这正是"省通信"的代价，需在指标中体现）
        if not msg.sent:
            for t in targets:
                t['difficulty_gated_default'] = True
        return {'round': round_idx, 'targets': targets,
                'difficulty_adjustments': adjusts,
                '超出可及范围_暂缓': deferred[:20],
                '暂缓数': len(deferred),
                'gate_message': msg.to_dict(),
                'action': (prev_feedback or {}).get('action', '首轮诊断')}

    def after_feedback(self, claims_summary, disputes, feedback, n_targets=1):
        """把审核结论与学习反馈汇总，门控判决是否回报给生成方。"""
        payload = {'disputes': len(disputes),
                   'contradicted': claims_summary.get('CONTRADICTED', 0),
                   'feedback': feedback}
        msg = self.gate.consider(
            'dispute', 'A3 审核纠偏', 'A4 协同调度', payload,
            ctx={'n_targets': n_targets,
                 'n_disputes': len(disputes) + claims_summary.get(
                     'CONTRADICTED', 0)})
        return {'gate_message': msg.to_dict(),
                'action': self.next_action(
                    feedback.get('correct_rate', 0.0),
                    feedback.get('attempts', 1))}
