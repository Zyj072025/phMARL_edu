# -*- coding: utf-8 -*-
"""门控通信（Gated Communication）—— 本作品的核心创新点二。

问题
----
多 Agent 协同最省事的做法是"每轮把一切都传给下一棒"（全通信）。代价是：
调用次数与 token 随角色数线性甚至平方增长，且冗余消息会淹没关键信号——
在审核场景里，把无关材料一股脑塞给审核 Agent，反而会降低判定的准确率。

机制（与前期科研中的 I2C 交互感知通信一致）
------------------------------------------
每一条候选消息在发送前，由**通信必要性判决器**给出"影响分"（impact）：
即"这条消息会不会改变接收方的决策"。只有 impact > τ 才真正发送，否则保持沉默。

* 生产形态（I2C 原实现）：影响分由 **PriorNet** 训练得到，
  监督标签是 Critic 对"接收方动作分布变化"的估计（因果效应）。
* 本教育系统形态：影响分由**可观测信号的代理量**计算（见 `impact_of`），
  接口与 PriorNet 完全一致，可无缝替换 —— 因为在 LLM Agent 流水线里，
  没有 Critic 可供反事实查询，代理量是当前可达的最佳估计。
  这一取舍在文档中如实标注（见 README「已知局限」）。

可测指标
--------
通信稀疏度 = 实际发送次数 / 候选消息总数（越高越省）；
token 节省率 = 1 − 发送消息字数 / 全部候选消息字数；
两者都必须与任务质量（幻觉率、适配率）**一并**报告 —— 只省成本不保质量不算收益。
"""
import json

# 优先级：数值越大越重要
PRIORITY = {'dispute': 3, 'diagnosis': 2, 'evidence': 1, 'ack': 0}


class Message(object):
    __slots__ = ('mtype', 'sender', 'receiver', 'payload', 'impact', 'sent',
                 'reason', 'chars')

    def __init__(self, mtype, sender, receiver, payload, impact=0.0,
                 reason=''):
        self.mtype = mtype
        self.sender = sender
        self.receiver = receiver
        self.payload = payload
        self.impact = float(impact)
        self.reason = reason
        self.sent = False
        self.chars = len(json.dumps(payload, ensure_ascii=False))

    def to_dict(self):
        return {'mtype': self.mtype, 'sender': self.sender,
                'receiver': self.receiver, 'impact': round(self.impact, 4),
                'sent': self.sent, 'reason': self.reason, 'chars': self.chars,
                'payload_brief': _brief(self.payload)}


def _brief(payload, n=120):
    s = json.dumps(payload, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + '…'


# ── 影响分（代理量） ─────────────────────────────────────────────────────
def impact_of(mtype, payload, ctx=None):
    """估计该消息对接收方决策的影响分 ∈ [0,1]。

    代理量按消息类型的语义定义，全部来自**发送方可观测**的量
    （真实系统里这些量就是 PriorNet 的输入特征）：

    dispute（审核→调度）  : 本轮每 1 个目标知识点摊到几条问题声明（争议+冲突）。
                            没有争议 ⇒ 调度方的教学动作不会变 ⇒ 不必发。
    diagnosis（诊断→生成）: 本轮目标里"与生成方默认档位（进阶）不同"的比例。
                            若全部目标都是进阶档，这条消息不改变生成方的产出 ⇒
                            不必发；若有目标被调成基础/挑战，则必须发。
    evidence（生成→审核） : 证据包对判定的边际影响（与接收方已有证据的重叠越低，
                            越可能改变判定）。
    ack（任意→任意）      : 固定低分，纯确认类消息按规则基本不发。

    归一化口径很关键：分母取**本轮目标数**而不是"盲区总数"。
    早期版本按盲区总数归一，结果盲区越多、影响分反而越小（26 个盲区 ⇒
    两个目标的影响力被摊薄到 0.15），门控几乎全关 —— 这是把
    "信息量"错算成了"占比"。按目标数归一后，影响分的语义回到它该有的样子：
    这条消息会不会改变接收方的决策。
    """
    ctx = ctx or {}
    if mtype == 'dispute':
        n = max(int(ctx.get('n_targets', 1)), 1)
        return min(1.0, float(ctx.get('n_disputes', 0)) / n)
    if mtype == 'diagnosis':
        n = max(int(ctx.get('n_targets', 1)), 1)
        return min(1.0, float(ctx.get('n_level_changing', 0)) / n)
    if mtype == 'evidence':
        return max(0.0, 1.0 - float(ctx.get('evidence_overlap', 1.0)))
    return 0.1


class Gate(object):
    """通信必要性判决器。threshold = τ（越大越省通信）。"""

    def __init__(self, threshold=0.3, verbose=False):
        self.tau = float(threshold)
        self.verbose = verbose
        self.candidates = 0
        self.sent = 0
        self.chars_all = 0
        self.chars_sent = 0
        self.log = []

    def consider(self, mtype, sender, receiver, payload, ctx=None):
        """登记一条候选消息并按 τ 判决是否发送。返回 Message（含 sent 标志）。"""
        m = Message(mtype, sender, receiver, payload,
                    impact=impact_of(mtype, payload, ctx))
        m.sent = m.impact > self.tau
        m.reason = ('影响分 %.3f %s τ=%.2f' %
                    (m.impact, '>' if m.sent else '≤', self.tau))
        self.candidates += 1
        self.chars_all += m.chars
        if m.sent:
            self.sent += 1
            self.chars_sent += m.chars
        self.log.append(m)
        if self.verbose:
            print('  [门控] %-9s %s→%s 影响分 %.3f  %s'
                  % (mtype, sender, receiver, m.impact,
                     '发送' if m.sent else '抑制'))
        return m

    def stats(self):
        c = max(self.candidates, 1)
        ca = max(self.chars_all, 1)
        return {
            '候选消息数': self.candidates,
            '实际发送数': self.sent,
            '通信稀疏度(发送/候选)': round(self.sent / float(c), 4),
            'token节省率(按字符数)': round(1.0 - self.chars_sent / float(ca), 4),
            '发送字符数': self.chars_sent, '全部字符数': self.chars_all,
        }

    def stats_line(self):
        s = self.stats()
        return ('候选 %d 条、实际发送 %d 条 ⇒ 通信稀疏度 %.1f%%，'
                'token 节省率 %.1f%%' % (s['候选消息数'], s['实际发送数'],
                                        s['通信稀疏度(发送/候选)'] * 100,
                                        s['token节省率(按字符数)'] * 100))

    def reset(self):
        self.candidates = self.sent = self.chars_all = self.chars_sent = 0
        self.log = []


class FullCommGate(Gate):
    """对照组：全通信（τ = −1，什么都发）。用于消融 B vs C。"""

    def __init__(self, verbose=False):
        Gate.__init__(self, threshold=-1.0, verbose=verbose)
