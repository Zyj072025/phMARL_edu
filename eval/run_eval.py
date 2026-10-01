# -*- coding: utf-8 -*-
"""消融评测 —— 五组对照 + 门控阈值扫描，全部指标由同一套口径计算。

五组消融（**每组只改一个变量**，靠参数开关而不是复制代码）
----------------------------------------------------------
  A 基线        诊断+生成+**审核**，但不做纠偏（冲突原样交付）；全通信
  B 协同+重生成  同 A，冲突后「重生成整份资源」，全通信
  C 协同+投影    同 A，冲突后做「知识约束投影」，全通信
  D 本文方法     在 C 之上把通信改为「门控通信」
  E 消融：无诊断 在 D 之上去掉学情诊断（统一默认起点）

A 与 B/C/D 共用同一份学情诊断与同一批目标知识点，**唯一差别是纠偏方式**，
因此「A 的冲突率 − C 的冲突率」可以直接读作约束投影的贡献。
单独增设 E，是因为「诊断」与「纠偏」是两件事，必须各自有一个只改它的对照组。

三组对照各自回答一个问题：
  A vs B/C —— 冲突不修（A）、重生成（B）、约束投影（C），哪个真的能修掉？
  C vs D   —— 门控省下的通信成本，是否以任务质量为代价？
  D vs E   —— 交付质量里有多少来自"千人千面的学情诊断"？

⚠️ 离线后端（mock）下的**必须注意**的一点
------------------------------------------
mock 是**确定性**后端：同样的输入必然得到同样的输出。因此 B 组的「重生成」
在离线口径下**不可能**改变任何内容，B 组必然等于 A 组。这不是把 B 组做弱了，
而是确定性后端下重生成策略的真实上限。要在真实后端上比较这两条路线，
必须用 `--backend openai`（温度 > 0，重生成才有意义）。

同时，mock 后端测出的幻觉率衡量的是**审核与投影流水线的有效性**
（夹具里注入了 10 条真实存在的领域误区作为 ground truth；
「取证判定」这一步在离线夹具下由条目级对账代做，真实后端下由 LLM 语义判定），
**不是**真实大模型的幻觉率，也**不含**判定器自身的识别能力。
真实数值请用 `--backend openai` 复测。

用法
----
    python eval/run_eval.py                        # 全部画像 × 五组 + τ 扫描
    python eval/run_eval.py --profiles L001,L003 --rounds 4 --budget-kp 3
    python eval/run_eval.py --backend openai --out out/eval_openai
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from engine.gating import FullCommGate, Gate          # noqa: E402
from engine.orchestrator import Orchestrator           # noqa: E402
from run import list_profiles, load_profile            # noqa: E402

# 五组消融的定义。**所有组共用同一批画像、同一轮数、同一预算**，
# 差别只在这三个开关上 —— 这是消融实验可信的前提。
ARMS = [
    {'id': 'A', '名称': 'A 基线（审核但不纠偏）',
     '说明': '同一份诊断、同一批目标知识点；审核照做但冲突原样交付，不做任何纠偏；全通信',
     'enable_diagnosis': True, 'correction': 'none', 'gated': False},
    {'id': 'B', '名称': 'B 协同 + 重生成纠偏',
     '说明': '与 A 共用同一份诊断与目标；冲突后重生成整份资源，而不是不做纠偏；全通信',
     'enable_diagnosis': True, 'correction': 'regenerate', 'gated': False},
    {'id': 'C', '名称': 'C 协同 + 知识约束投影',
     '说明': '诊断+生成+审核；冲突后做知识约束投影；全通信。与 B 的差别只有纠偏方式',
     'enable_diagnosis': True, 'correction': 'projection', 'gated': False},
    {'id': 'D', '名称': 'D 本文方法（门控通信 + 约束投影）',
     '说明': '在 C 的基础上把通信改为门控通信。与 C 的差别只有门控',
     'enable_diagnosis': True, 'correction': 'projection', 'gated': True},
    {'id': 'E', '名称': 'E 消融：关闭学情诊断',
     '说明': '在 D 的基础上去掉学情诊断（统一默认起点）。与 D 的差别只有诊断',
     'enable_diagnosis': False, 'correction': 'projection', 'gated': True},
]



# ══════════════════════════════════════════════════════════════ 跑一组
def run_arm(arm, profiles, args, tau=0.3):
    rows = []
    for prof in profiles:
        gate = Gate(threshold=tau, verbose=False) if arm['gated'] else FullCommGate()
        orc = Orchestrator(backend=args.backend, gate=gate,
                           budget_kp=args.budget_kp, rounds=args.rounds,
                           eps=args.eps, verbose=False,
                           enable_diagnosis=arm['enable_diagnosis'],
                           correction=arm['correction'])
        sess = orc.run(prof, out_dir=None, write=False)
        s = sess['summary']
        rows.append({
            '画像': sess['profile_id'],
            '投影前幻觉率': s['投影前_幻觉率_宽口径'],
            '投影后幻觉率': s['投影后_幻觉率_宽口径'],
            '严口径幻觉率': s['投影前_幻觉率_严口径'],
            '适配率': s['适配率(心流区间[0.6,0.8])'],
            '适配准确率': s['适配准确率(决策口径)'],
            '知识点数': s['涉及知识点数'],
            '全库覆盖率': s['全库知识点覆盖率'],
            '投影介入率': s['投影介入率'],
            '平均编辑距离': s['平均归一化编辑距离'],
            '声明总数': s['声明总数'],
            '通信稀疏度': s['门控']['通信稀疏度(发送/候选)'],
            'token节省率': s['门控']['token节省率(按字符数)'],
            '候选消息数': s['门控']['候选消息数'],
            '发送消息数': s['门控']['实际发送数'],
            'cross_check争议数': s['交叉验证争议数'],
            '轮数': s['轮数'],
        })
    return rows


def mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return sum(vals) / float(len(vals)) if vals else 0.0


def aggregate(rows):
    keys = ['投影前幻觉率', '投影后幻觉率', '严口径幻觉率', '适配率', '适配准确率',
            '全库覆盖率', '投影介入率', '平均编辑距离', '通信稀疏度',
            'token节省率', '声明总数', '知识点数', 'cross_check争议数']
    return {k: round(mean(rows, k), 4) for k in keys} | {'画像数': len(rows)}


# ══════════════════════════════════════════════════════════════ τ 扫描
def sweep_tau(profiles, args, taus):
    """门控阈值扫描：把「省了多少通信」与「质量掉了多少」画在同一条线上。"""
    out = []
    for tau in taus:
        gate = Gate(threshold=tau, verbose=False)
        rows = []
        for prof in profiles:
            orc = Orchestrator(backend=args.backend, gate=gate, budget_kp=args.budget_kp,
                               rounds=args.rounds, eps=args.eps, verbose=False,
                               enable_diagnosis=True, correction='projection')
            gate.reset()
            s = orc.run(prof, out_dir=None, write=False)['summary']
            rows.append({'稀疏度': s['门控']['通信稀疏度(发送/候选)'],
                         'token节省率': s['门控']['token节省率(按字符数)'],
                         '投影后幻觉率': s['投影后_幻觉率_宽口径'],
                         '适配率': s['适配率(心流区间[0.6,0.8])'],
            '适配准确率': s['适配准确率(决策口径)'],
                         '知识点数': s['涉及知识点数']})
        out.append({'tau': tau,
                    '通信稀疏度': round(mean(rows, '稀疏度'), 4),
                    'token节省率': round(mean(rows, 'token节省率'), 4),
                    '投影后幻觉率': round(mean(rows, '投影后幻觉率'), 4),
                    '适配率': round(mean(rows, '适配率'), 4),
                    '涉及知识点数': round(mean(rows, '知识点数'), 2)})
    return out


# ══════════════════════════════════════════════════════════════ 输出
def render_md(result, args):
    L = ['# Ph-MARL-Edu 消融评测报告', '',
         '生成时间：%s' % time.strftime('%Y-%m-%d %H:%M:%S'),
         '后端：`%s`　画像：%s　轮数：%d　每轮知识点预算：%d'
         % (result['backend'], '、'.join(result['profiles']),
            args.rounds, args.budget_kp), '',
         '## 一、五组消融对照', '',
         '| 组 | 投影前幻觉率 | **投影后幻觉率** | 严口径幻觉率 | '
         '**适配准确率(决策口径)** | 适配率(实测口径) | 通信稀疏度 | token 节省率 | '
         '投影介入率 | 平均编辑距离 |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for a in result['arms']:
        g = a['汇总']
        L.append('| %s | %.2f%% | **%.2f%%** | %.2f%% | **%.1f%%** | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.4f |'
                 % (a['名称'], g['投影前幻觉率'] * 100, g['投影后幻觉率'] * 100,
                    g['严口径幻觉率'] * 100, g['适配准确率'] * 100, g['适配率'] * 100,
                    g['通信稀疏度'] * 100, g['token节省率'] * 100,
                    g['投影介入率'] * 100, g['平均编辑距离']))
    L += ['', '### 各组定义', '']
    for a in result['arms']:
        L.append('- **%s**：%s' % (a['名称'], a['说明']))

    L += ['', '## 二、门控阈值 τ 扫描', '',
          '| τ | 通信稀疏度 | token 节省率 | 投影后幻觉率 | 适配率 |',
          '|---|---|---|---|---|']
    for r in result['tau_sweep']:
        L.append('| %.1f | %.1f%% | %.1f%% | %.2f%% | %.1f%% |'
                 % (r['tau'], r['通信稀疏度'] * 100, r['token节省率'] * 100,
                    r['投影后幻觉率'] * 100, r['适配率'] * 100))

    L += ['', '## 三、逐画像明细', '',
          '| 组 | 画像 | 声明总数 | 投影前幻觉率 | 投影后幻觉率 | 适配率 | '
          '通信稀疏度 | token 节省率 | 争议数 |', '|---|---|---|---|---|---|---|---|---|']
    for a in result['arms']:
        for r in a['逐画像']:
            L.append('| %s | %s | %d | %.2f%% | %.2f%% | %.1f%% | %.1f%% | %.1f%% | %d |'
                     % (a['id'], r['画像'], r['声明总数'],
                        r['投影前幻觉率'] * 100, r['投影后幻觉率'] * 100,
                        r['适配率'] * 100, r['通信稀疏度'] * 100,
                        r['token节省率'] * 100, r['cross_check争议数']))

    L += ['', '## 四、口径与已知局限（必须与上面的数字一起读）', '',
          '1. **离线后端（mock）的幻觉率衡量的是机制有效性，不是真实大模型的幻觉率。**',
          '   夹具中注入了 10 条真实存在的领域误区作为 ground truth，'
          '用于验证「声明切分→取证判定→约束投影」这条链路能否抓住并修正错误。'
          '其中「取证判定」这一步在离线夹具下由**条目级对账**代做'
          '（`engine/llm.py::MockLLM.judge` 第 1 条判据），真实后端下'
          '由 LLM 的语义判定完成。因此本表衡量的是**流水线**能否抓住并修正冲突，'
          '**不含**判定器自身的识别能力；真实数值请用 `--backend openai` 复测。',
          '2. **B 组（重生成）在离线后端下必然等于 A 组**：mock 是确定性后端，'
          '同样的输入必得同样的输出，重生成不可能改变任何内容。'
          '要在真实后端上比较「重生成 vs 约束投影」，必须用 `--backend openai`。',
          '3. **适配率是模拟口径**：离线演示没有真实学习者，作答由确定性模拟器'
          '（`LearnerSim`）产生，且模拟器与调度方**共用同一张资源支撑系数表**，'
          '因此该数值高于真实学习者群体能达到的水平。真实适配率需要真实作答数据。',
          '4. **幻觉率只统计「领域事实断言」**：操作性与流程性表述、疑问句题干'
          '（题干提出的是问题，不作出断言）不计入分母，其条数在审核明细中'
          '逐条列出并标明归类理由。',
          '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description='Ph-MARL-Edu 消融评测')
    ap.add_argument('--profiles', default='', help='逗号分隔的画像 ID，默认全部')
    ap.add_argument('--backend', default='mock', choices=['mock', 'openai'])
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--budget-kp', type=int, default=3)
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--taus', default='0.0,0.2,0.3,0.5,0.8,1.0')
    ap.add_argument('--out', default=os.path.join(ROOT, 'out', 'eval'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pids = [p.strip() for p in args.profiles.split(',') if p.strip()] \
        or list_profiles()
    profiles = [load_profile(p) for p in pids]

    print('=' * 74)
    print('Ph-MARL-Edu 消融评测 ｜ 后端 %s ｜ 画像 %s' % (args.backend, '、'.join(pids)))
    print('轮数 %d ｜ 每轮预算 %d 个知识点 ｜ 五组消融 + τ 扫描'
          % (args.rounds, args.budget_kp))
    print('=' * 74)

    arms_out = []
    for a in ARMS:
        rows = run_arm(a, profiles, args)
        agg = aggregate(rows)
        arms_out.append({'id': a['id'], '名称': a['名称'], '说明': a['说明'],
                         '逐画像': rows, '汇总': agg})
        print('%-34s 投影后幻觉率 %6.2f%% ｜ 适配准确率 %5.1f%% ｜ 实测适配率 %5.1f%% ｜ '
              '稀疏度 %5.1f%% ｜ token 节省 %5.1f%%' % (
                  a['名称'], agg['投影后幻觉率'] * 100, agg['适配准确率'] * 100,
                  agg['适配率'] * 100, agg['通信稀疏度'] * 100,
                  agg['token节省率'] * 100))

    taus = [float(t) for t in args.taus.split(',') if t.strip()]
    print('\nτ 扫描：')
    sweep = sweep_tau(profiles, args, taus)
    for r in sweep:
        print('  τ=%.1f  稀疏度 %5.1f%%  token 节省 %5.1f%%  投影后幻觉率 %5.2f%%  '
              '适配率 %5.1f%%' % (r['tau'], r['通信稀疏度'] * 100,
                                r['token节省率'] * 100, r['投影后幻觉率'] * 100,
                                r['适配率'] * 100))

    result = {'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
              'backend': args.backend, 'profiles': pids,
              'rounds': args.rounds, 'budget_kp': args.budget_kp,
              'arms': arms_out, 'tau_sweep': sweep}
    with open(os.path.join(args.out, 'eval_results.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    md = render_md(result, args)
    with open(os.path.join(args.out, '消融评测.md'), 'w', encoding='utf-8') as f:
        f.write(md)
    print('\n产物：')
    print('  %s' % os.path.join(args.out, '消融评测.md'))
    print('  %s' % os.path.join(args.out, 'eval_results.json'))


if __name__ == '__main__':
    main()
