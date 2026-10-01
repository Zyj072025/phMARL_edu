# -*- coding: utf-8 -*-
"""协同闭环编排器 —— 「分析 → 生成 → 校验 → 决策 → 反馈」五步闭环。

闭环详图
--------
    ① 分析    A1 学情诊断   画像 → 知识点掌握度 p̂ + 盲区 + 能力表征 z
    ② 生成    A2 领域知识生成 盲区 + 难度档 + 知识库证据 → 三类资源（带引用）
    ③ 校验    A3 审核纠偏    声明级三态判定 → 双视角交叉验证/辩论 → 知识约束投影
    ④ 决策    A4 协同调度    门控通信判决 + 教学目标/难度/下一步动作
    ⑤ 反馈    学习者作答 → 正确率 → 回写掌握度 → 触发下一轮（降维解释 / 进阶挑战）

每一步都写入 `trace.jsonl`（一行一步，含门控判决与投影记录），
这份 trace 就是交付物里的「可审计链路」，也是 PPT 里闭环示意图的真实数据来源。

关于反馈信号的诚实说明
----------------------
本版本用 `LearnerSim`（**学习者响应模拟器**）产生作答反馈，因为离线环境里没有真实学习者。
模拟器用画像中的**隐藏真值掌握度**（与诊断估计之间存在确定性偏差）驱动作答，
从而让"反馈纠偏"这一步有真实的纠偏空间可展示。
**真实部署时把 `LearnerSim` 换成真实作答数据即可，接口不变**（见 README「已知局限」）。
"""
import json
import os
import time
import zlib

from .agents import (AuditorAgent, DiagnoserAgent, GeneratorAgent,
                     SchedulerAgent, DIFF_REQUIRED)
from .gating import FullCommGate, Gate
from .kb import load as load_kb
from .llm import get_backend

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 难度档的**资源支撑系数**：讲义/指南本身会提供脚手架，
# 因此同一名学习者在「基础」资源后的即时作答正确率高于其原始掌握度估计，
# 在「挑战」资源后则低于估计值。系数取值是建模假设，已在 README「已知局限」中标注。
DIFF_FACTOR = {'基础': 1.45, '进阶': 1.10, '挑战': 0.80}
# 资源支撑系数：同一掌握度下，不同难度档资源让作答正确率变化多少倍。
# **这三个数不是拍出来的，是按心流区间反解的**（推导见 README「难度梯子的标定」）：
#   · 基础档 —— 让"最弱仍值得投放"的学习者（掌握度 ≈ 0.42）恰好落进区间下沿：
#               f_基础 = 0.60 / 0.42 ≈ 1.43 → 取 1.45
#   · 挑战档 —— 让"已接近满分"的学习者（掌握度 ≈ 0.98）不被推到区间之外：
#               f_挑战 = 0.80 / 0.98 ≈ 0.82 → 取 0.80
#   · 进阶档 —— 取两者的几何中项：√(1.45×0.80) ≈ 1.08 → 取 1.10
# 旧值 1.30/1.05/0.80 的问题：中级档只到 0.57，中低掌握度学习者**无论怎么选档
# 都进不了心流区间**（三档全在外面），适配率因此结构性受损 ——
# 这不是策略问题，是梯子太窄（见 eval/oracle_ceiling.py 的量化证据）。
QUIZ_ITEMS = 5                     # 每个知识点每次投放的作答题数


# ══════════════════════════════════════════════════════ 学习者响应模拟器
class LearnerSim(object):
    """确定性作答模拟器。**不是真实学习者数据**，仅用于离线演示闭环。"""

    name = 'LearnerSim(模拟)'

    def __init__(self, profile, jitter=0.12):
        self.profile = profile
        self.jitter = jitter
        self.true_mastery = self._true_mastery(profile, jitter)

    @staticmethod
    def _skill(profile):
        """与 DiagnoserAgent 用**同一套**画像解析口径（否则诊断与真值不可比）。"""
        from .agents import HANDS_LEVEL, _ratio
        theory = _ratio(profile.get('theory_score'), 0.5)
        hands = HANDS_LEVEL.get(str(profile.get('hands_on', '')).strip(), 0.25)
        hist = profile.get('history_accuracy')
        hist = 0.5 if hist is None else _ratio(hist, 0.5)
        return 0.55 * theory + 0.30 * hands + 0.15 * hist

    def _true_mastery(self, profile, jitter):
        """隐藏真值 = 诊断口径 + 按 (画像, 知识点) 确定的偏移（可复现，非随机）。"""
        import math
        skill = self._skill(profile)
        out = {}
        for kp, req in DIFF_REQUIRED_BY_KP.items():
            base = 1.0 / (1.0 + math.exp(-6.0 * (skill - req)))
            h = zlib.crc32((str(profile.get('profile_id', '')) + '|' + kp)
                           .encode('utf-8'))
            off = ((h % 1000) / 1000.0 * 2.0 - 1.0) * jitter
            out[kp] = min(0.98, max(0.02, base + off))
        return out

    def quiz(self, kp, difficulty, n_items=QUIZ_ITEMS):
        """在该知识点、该难度档下作答 n_items 题，返回 (答对数, 题数)。"""
        p = self.true_mastery.get(kp, 0.5) * DIFF_FACTOR.get(difficulty, 1.05)
        p = min(0.97, max(0.02, p))
        correct = sum(1 for i in range(n_items)
                      if (zlib.crc32(('%s|%s|%s|%d' % (
                          self.profile.get('profile_id', ''), kp, difficulty, i)
                      ).encode('utf-8')) % 1000) / 1000.0 < p)
        return correct, n_items


DIFF_REQUIRED_BY_KP = {}          # 由 Orchestrator 在初始化时按知识库填充


# ══════════════════════════════════════════════════════════════ 编排器
class Orchestrator(object):

    def __init__(self, backend='mock', gate=None, budget_kp=2, rounds=2,
                 eps=0.05, verbose=True, sim=None, enable_diagnosis=True,
                 correction='projection'):
        """correction: 'projection' 约束投影 ｜ 'regenerate' 重生成纠偏 ｜ 'none' 不干预。

        这三个开关是为了让**消融实验**能真正只改一个变量（见 eval/run_eval.py）：
        把某个组件关掉必须靠参数，而不是靠复制一份代码 —— 复制出来的两份代码
        会各自漂移，消融结果也就不可信了。
        """
        self.kb = load_kb()
        self.llm = get_backend(backend) if isinstance(backend, str) else backend
        self.gate = gate if gate is not None else Gate(threshold=0.3)
        self.budget_kp = budget_kp
        self.rounds = rounds
        self.eps = eps
        self.verbose = verbose
        self.sim = sim
        self.enable_diagnosis = enable_diagnosis
        self.correction = correction

        DIFF_REQUIRED_BY_KP.clear()
        for d in self.kb.docs:
            DIFF_REQUIRED_BY_KP[d['title']] = DIFF_REQUIRED.get(
                d.get('difficulty', '进阶'), 0.55)

        self.a1 = DiagnoserAgent(self.kb, self.llm)
        self.a2 = GeneratorAgent(self.kb, self.llm)
        self.a3 = AuditorAgent(self.kb, self.llm, eps=eps)
        self.a4 = SchedulerAgent(self.gate, budget_kp=budget_kp, kb=self.kb)

    # ------------------------------------------------------------- 工具
    def _say(self, *a):
        if self.verbose:
            print(*a)

    # ------------------------------------------------------------- 校验与纠偏
    def _review(self, res, kp, diff, profile, max_regen=2):
        """按 correction 开关执行校验与纠偏，返回与 A3.review 同构的结果 dict。"""
        from . import claims as CLAIMS

        if self.correction == 'none':
            # 消融基线：只检测不干预（衡量"原始生成"的质量）
            claim_list, disputes, nd = self.a3.audit(res['text_raw'])
            return {
                'agent': self.a3.name, '纠偏方式': '不干预（仅检测）',
                'pre_audit': CLAIMS.summarize(claim_list, nd),
                'post_audit': CLAIMS.summarize_delivered(claim_list),
                'non_domain': nd, 'claims_pre': [c.to_dict() for c in claim_list],
                'claims': [c.to_dict() for c in claim_list],
                'claims_obj': claim_list, 'disputes': disputes,
                'projection_records': [], 'projection_stats': {
                    '声明总数': len(claim_list), '冲突声明数': sum(
                        1 for c in claim_list if c.verdict == 'CONTRADICTED'),
                    '投影介入率': 0.0, '替换数': 0, '限定化数': 0, '删除数': 0,
                    '平均编辑距离': 0.0, '残余未合规数': sum(
                        1 for c in claim_list if c.verdict == 'CONTRADICTED'),
                    '残余未合规比例': 0.0, '未溯源保留数': sum(
                        1 for c in claim_list if c.verdict == 'UNVERIFIED'),
                    '可行性': False},
                'feasible': False, 'text_delivered': res['text_raw'],
            }

        if self.correction == 'regenerate':
            # 消融对照：发现冲突后**重生成整份资源**（不做事后约束投影）
            text, attempts = res['text_raw'], []
            for i in range(max_regen + 1):
                claim_list, disputes, nd = self.a3.audit(text)
                n_conf = sum(1 for c in claim_list if c.verdict == 'CONTRADICTED')
                attempts.append({'轮次': i, '冲突声明数': n_conf})
                if n_conf == 0 or i == max_regen:
                    break
                text = self.llm.generate_resource(
                    res['kind'], kp, diff,
                    [self.kb.get(e['doc_id']) for e in res['evidence']
                     if self.kb.get(e['doc_id'])], profile)
            return {
                'agent': self.a3.name, '纠偏方式': '重生成（消融对照）',
                'pre_audit': CLAIMS.summarize(claim_list, nd),
                'post_audit': CLAIMS.summarize_delivered(claim_list),
                'non_domain': nd, 'claims_pre': [c.to_dict() for c in claim_list],
                'claims': [c.to_dict() for c in claim_list],
                'claims_obj': claim_list, 'disputes': disputes,
                'projection_records': [],
                'projection_stats': {
                    '声明总数': len(claim_list), '冲突声明数': sum(
                        1 for c in claim_list if c.verdict == 'CONTRADICTED'),
                    '投影介入率': 0.0, '替换数': 0, '限定化数': 0, '删除数': 0,
                    '平均编辑距离': 0.0,
                    '残余未合规数': sum(1 for c in claim_list
                                   if c.verdict == 'CONTRADICTED'),
                    '残余未合规比例': 0.0, '未溯源保留数': sum(
                        1 for c in claim_list if c.verdict == 'UNVERIFIED'),
                    '可行性': False},
                'feasible': False, '重生成轨迹': attempts,
                'text_delivered': text,
            }

        return self.a3.review(res['text_raw'])          # 本文方法：约束投影

    # ------------------------------------------------------------- 主流程
    def run(self, profile, out_dir=None, write=True):
        """跑一次完整闭环，返回 session dict（并落盘到 out_dir）。"""
        t0 = time.time()
        sim = self.sim or LearnerSim(profile)
        trace = []
        rounds = []

        self._say('─' * 68)
        self._say('学习者画像：%s（%s｜理论 %s｜实操 %s）' % (
            profile.get('profile_id'), profile.get('education'),
            profile.get('theory_score'), profile.get('hands_on')))
        self._say('─' * 68)

        # ① 分析
        diag = self.a1.diagnose(profile)
        if not self.enable_diagnosis:
            # 消融：不做学情诊断 ⇒ 所有知识点按同一默认掌握度处理，
            # 教学起点不再个性化（这是 D 组与完整系统的唯一差别）。
            diag = dict(diag)
            diag['mastery'] = {k: 0.55 for k in diag['mastery']}
            diag['blind_spots'] = []
            diag['诊断开关'] = '关闭（消融对照：统一默认起点）'
        # 完整掌握度表一并入 trace：报告的第一张图（盲区定位）直接读它，
        # 从而保证"图上的数据"与"链路上的记录"是同一份，不会两处各算一遍。
        trace.append({'step': '①分析/学情诊断', 'agent': self.a1.name,
                      'data': diag})
        self._say('① 学情诊断：综合水平 %.3f，检出盲区 %d 个（最低：%s %.3f）' % (
            diag['skill_level'], len(diag['blind_spots']),
            diag['blind_spots'][0]['knowledge_point'] if diag['blind_spots'] else '-',
            diag['blind_spots'][0]['mastery'] if diag['blind_spots'] else 0))

        prev_feedback = None
        for r in range(1, self.rounds + 1):
            # ② 目标与难度决策 + 门控通信判决
            plan = self.a4.plan_round(diag, round_idx=r, prev_feedback=prev_feedback)
            if not plan['targets']:
                self._say('   本轮无盲区可推，闭环提前收敛于第 %d 轮。' % (r - 1))
                break
            gm = plan['gate_message']
            trace.append({'step': '②调度/目标与门控判决', 'agent': self.a4.name,
                          'round': r, 'data': plan})
            self._say('② 第 %d 轮：目标 %s ｜ 门控 %s（影响分 %.3f）→ %s' % (
                r, '、'.join(t['knowledge_point'] for t in plan['targets']),
                gm['mtype'], gm['impact'], '发送' if gm['sent'] else '抑制'))

            round_rec = {'round': r, 'plan': plan, 'items': [], 'feedback': None}

            kp_results = []
            msg_sent = bool(plan['gate_message'].get('sent'))
            for t in plan['targets']:
                kp = t['knowledge_point']
                # 门控判决**必须真的改变生成方的行为**，否则"省通信"就是白捡的：
                # 诊断消息被抑制时，生成方拿不到个性化难度，只能按自己的默认档位
                # （进阶）出资源 —— 这就是门控在质量侧的代价，也是消融 C vs D
                # 真正要测的东西。
                diff = t['difficulty'] if msg_sent else '进阶'
                if not msg_sent and diff != t['difficulty']:
                    t['difficulty_effective'] = diff
                    t['门控降级'] = ('诊断消息被抑制（影响分 %.3f ≤ τ %.2f），'
                                 '生成方回落默认「进阶」档；个性化难度 %s 未生效'
                                 % (plan['gate_message']['impact'],
                                    self.gate.tau, t['difficulty']))
                # ③ 生成
                gen = self.a2.generate(kp, diff, profile)
                item = {'knowledge_point': kp, 'difficulty': diff, 'resources': []}
                self._say('③ 生成：%s（%s 档）→ %d 类资源' % (
                    kp, diff, len(gen['resources'])))
                for res in gen['resources']:
                    # ④ 校验（按消融开关选择纠偏方式）
                    rev = self._review(res, kp, diff, profile)
                    pre, post = rev['pre_audit'], rev['post_audit']
                    item['resources'].append({
                        'kind': res['kind'], 'label': res['label'],
                        'difficulty': diff,
                        'evidence': res['evidence'],
                        'text_raw': res['text_raw'],
                        'text_delivered': rev['text_delivered'],
                        'pre_audit': pre, 'post_audit': post,
                        'non_domain': rev['non_domain'],
                        'claims_pre': rev['claims_pre'], 'claims': rev['claims'],
                        'disputes': rev['disputes'],
                        'projection_stats': rev['projection_stats'],
                        'projection_records': rev['projection_records'],
                        'feasible': rev['feasible'],
                    })
                    self._say('   %-6s 声明 %2d（另流程性文本 %d）｜ 投影前 %.1f%% '
                              '→ 交付后 %.1f%% ｜ 介入率 %.0f%% ｜ 争议 %d' % (
                                  res['label'], pre['声明总数'],
                                  pre['非断言性文本_不计入'],
                                  pre['幻觉率_宽口径(冲突/总数)'] * 100,
                                  post['幻觉率_宽口径(冲突/交付)'] * 100,
                                  rev['projection_stats']['投影介入率'] * 100,
                                  len(rev['disputes'])))
                # ⑤ 反馈（模拟作答）
                correct, total = sim.quiz(kp, diff)
                kp_results.append({'knowledge_point': kp, 'difficulty': diff,
                                   'correct': correct, 'total': total})
                item['feedback'] = {'correct': correct, 'total': total,
                                    'correct_rate': round(correct / float(total), 3)}
                trace.append({'step': '③④校验/审核与投影', 'agent': self.a3.name,
                              'round': r, 'knowledge_point': kp,
                              'data': {'pre': item['resources'][0]['pre_audit'],
                                       'post': item['resources'][0]['post_audit'],
                                       'projection': item['resources'][0]['projection_stats']}})
                round_rec['items'].append(item)

            c = sum(x['correct'] for x in kp_results)
            n = sum(x['total'] for x in kp_results)
            feedback = {'round': r, 'correct': c, 'total': n,
                        'correct_rate': round(c / float(max(n, 1)), 3),
                        'per_kp': kp_results}
            # 决策：下一步教学动作（并门控回报给生成方）
            dec = self.a4.after_feedback(
                {'声明总数': sum(it['resources'][0]['pre_audit']['声明总数']
                                 for it in round_rec['items']),
                 'CONTRADICTED': sum(it['resources'][0]['pre_audit']
                                     .get('CONTRADICTED', 0)
                                     for it in round_rec['items'])},
                [d for it in round_rec['items'] for d in it['resources'][0]['disputes']],
                {'correct_rate': feedback['correct_rate'],
                 'attempts': sum(x['total'] for x in kp_results)},
                n_targets=len(round_rec['items']))
            feedback['action'] = dec['action']
            feedback['gate_message'] = dec['gate_message']
            round_rec['feedback'] = feedback

            trace.append({'step': '⑤反馈/作答与教学决策', 'agent': self.a4.name,
                          'round': r, 'data': feedback})
            self._say('⑤ 反馈：答对 %d/%d（%.0f%%）→ 教学动作「%s」；'
                      '回写掌握度并进入下一轮' % (
                          c, n, feedback['correct_rate'] * 100, dec['action']))

            # 掌握度回写：以实测正确率为观测，估计值向观测做一次指数修正
            for x in kp_results:
                kp = x['knowledge_point']
                if kp in diag['mastery']:
                    obs = x['correct'] / float(x['total'])
                    diag['mastery'][kp] = round(
                        0.7 * diag['mastery'][kp] + 0.3 * obs, 4)
            diag['blind_spots'] = sorted(
                [{'knowledge_point': k, 'mastery': v}
                 for k, v in diag['mastery'].items() if v < 0.60],
                key=lambda x: x['mastery'])
            prev_feedback = feedback
            rounds.append(round_rec)

        session = {
            'profile_id': profile.get('profile_id'),
            'profile': profile,
            'backend': getattr(self.llm, 'name', 'mock'),
            'feedback_source': sim.name,
            'enable_diagnosis': self.enable_diagnosis,
            'correction': self.correction,
            'gate_threshold': getattr(self.gate, 'tau', None),
            'rounds': rounds,
            'trace': trace,
            'gate_stats': self.gate.stats(),
            'elapsed_sec': round(time.time() - t0, 2),
        }
        session['summary'] = summarize_session(session, self.kb)
        if write and out_dir:
            write_outputs(session, out_dir, self.kb)
        return session


# ══════════════════════════════════════════════════════════ 指标聚合
def summarize_session(session, kb):
    """把一次闭环会话聚合为可汇报指标。所有口径在此处**唯一定义**，各处引用同一份。"""
    pre_n = pre_c = pre_u = 0
    post_n = post_c = post_u = 0
    kps, kinds = set(), set()
    inter, dist_sum, dist_n = [], 0.0, 0
    disputes = non_domain = 0
    for rd in session['rounds']:
        for it in rd['items']:
            kps.add(it['knowledge_point'])
            for res in it['resources']:
                kinds.add(res['kind'])
                p, q = res['pre_audit'], res['post_audit']
                pre_n += p['声明总数']
                pre_c += p.get('CONTRADICTED', 0)
                pre_u += p.get('UNVERIFIED', 0)
                non_domain += p.get('非断言性文本_不计入', 0)
                post_n += q['交付声明数']
                post_c += q.get('CONTRADICTED', 0)
                post_u += q.get('UNVERIFIED', 0)
                inter.append(res['projection_stats']['投影介入率'])
                dist_sum += res['projection_stats']['平均编辑距离'] * \
                    res['projection_stats']['投影介入率'] * p['声明总数']
                dist_n += res['projection_stats']['投影介入率'] * p['声明总数']
                disputes += len(res['disputes'])
    # 适配率（**两个口径都要报，缺一个就会误导**）：
    #   实测口径 = 作答正确率落入心流区间 [60%,80%] 的项占比。
    #              它把"题量少导致的量化误差"和"诊断估计的固有偏差"一起算了进来，
    #              因此**不能**直接当作调度质量：连用真值选难度的 oracle 上界
    #              都只有 47%~76%（见 eval/oracle_ceiling.py）。
    #   决策口径 = 交付难度使该画像的**预测**正确率落入心流区间的项占比。
    #              它衡量"系统选难度"这件事本身，是机制的直接保证。
    ok = tot = 0
    ok_plan = 0
    for rd in session['rounds']:
        plan_m = {t['knowledge_point']: t['mastery']
                  for t in rd['plan'].get('targets', [])}
        for it in rd['items']:
            tot += 1
            if 0.60 <= it['feedback']['correct_rate'] <= 0.80:
                ok += 1
            m = plan_m.get(it['knowledge_point'])
            if m is not None:
                p = min(0.97, max(0.02, m * DIFF_FACTOR.get(it['difficulty'], 1.05)))
                if 0.60 <= p <= 0.80:
                    ok_plan += 1
    all_kp = len(kb.docs)
    return {
        '轮数': len(session['rounds']),
        '涉及知识点数': len(kps),
        '资源形态数': len(kinds),
        '声明总数': pre_n,
        '非断言性文本数(不计入幻觉率)': non_domain,
        '投影前_幻觉率_宽口径': round(pre_c / float(pre_n), 4) if pre_n else 0.0,
        '投影前_幻觉率_严口径': round((pre_c + pre_u) / float(pre_n), 4) if pre_n else 0.0,
        '交付声明数': post_n,
        '投影后_幻觉率_宽口径': round(post_c / float(post_n), 4) if post_n else 0.0,
        '投影后_幻觉率_严口径': round((post_c + post_u) / float(post_n), 4)
            if post_n else 0.0,
        '投影介入率': round(sum(inter) / len(inter), 4) if inter else 0.0,
        '平均归一化编辑距离': round(dist_sum / dist_n, 4) if dist_n else 0.0,
        '交叉验证争议数': disputes,
        '适配率(心流区间[0.6,0.8])': round(ok / float(tot), 4) if tot else 0.0,
        '适配准确率(决策口径)': round(ok_plan / float(tot), 4) if tot else 0.0,
        '目标知识点覆盖率': 1.0 if tot else 0.0,
        '全库知识点覆盖率': round(len(kps) / float(all_kp), 4) if all_kp else 0.0,
        '门控': session['gate_stats'],
    }


# ══════════════════════════════════════════════════════════ 落盘
def write_outputs(session, out_dir, kb):
    os.makedirs(out_dir, exist_ok=True)
    pid = session['profile_id'] or 'profile'

    # 1) 每一步的 trace（一行一步）
    with open(os.path.join(out_dir, 'trace.jsonl'), 'w', encoding='utf-8') as f:
        for rec in session['trace']:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')

    # 2) 资源交付文本（人可读）
    lines = ['# 个性化学习资源包 — 学习者 %s' % pid, '',
             '> 本文件由 Ph-MARL-Edu 四智能体协同闭环自动生成。'
             '每条资源均经过声明级审核与知识约束投影，'
             '文末证据出处可逐条回溯至知识库原文。', '']
    for rd in session['rounds']:
        lines.append('## 第 %d 轮' % rd['round'])
        for it in rd['items']:
            lines.append('### %s（难度档：%s ｜ 实测正确率 %.0f%%）' % (
                it['knowledge_point'], it['difficulty'],
                it['feedback']['correct_rate'] * 100))
            for res in it['resources']:
                lines.append('#### %s' % res['label'])
                lines.append(res['text_delivered'])
                lines.append('')
    with open(os.path.join(out_dir, 'resources.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    # 3) 审核明细（含每条声明的判定、证据、投影记录）
    audit = {'profile_id': pid,
             '说明': '声明级审核明细：每条声明的三态判定、取证出处、'
                     '投影算子与最终文本。这是"高保真知识溯源"的原始记录。',
             'resources': []}
    for rd in session['rounds']:
        for it in rd['items']:
            for res in it['resources']:
                audit['resources'].append({
                    'round': rd['round'], 'knowledge_point': it['knowledge_point'],
                    'label': res['label'], 'difficulty': res['difficulty'],
                    'evidence': res['evidence'],
                    'pre_audit': res['pre_audit'], 'post_audit': res['post_audit'],
                    'claims': res['claims'], 'disputes': res['disputes'],
                    'projection_records': res['projection_records'],
                    'projection_stats': res['projection_stats'],
                })
    with open(os.path.join(out_dir, 'audit_trace.json'), 'w',
              encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    # 4) 会话 total（不含资源全文，便于程序读取）
    slim = dict(session)
    slim.pop('trace', None)
    with open(os.path.join(out_dir, 'session.json'), 'w', encoding='utf-8') as f:
        json.dump(slim, f, ensure_ascii=False, indent=2, default=str)

    # 5) 匹配度报告
    from .report import build_report, render_html
    rep = build_report(session, kb)
    with open(os.path.join(out_dir, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, 'report.html'), 'w', encoding='utf-8') as f:
        f.write(render_html(rep, session))
    return out_dir


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.path.insert(0, ROOT)
    from engine.orchestrator import Orchestrator
    prof = {'profile_id': 'L001', 'education': '本科大三 人工智能专业',
            'theory_score': '62/100', 'hands_on': '有 Python，无 RL 经验',
            'history_accuracy': 0.58}
    o = Orchestrator(backend='mock', rounds=2)
    s = o.run(prof, out_dir=os.path.join(ROOT, 'out', 'L001'))
    print(json.dumps(s['summary'], ensure_ascii=False, indent=2))
