# -*- coding: utf-8 -*-
"""适配率上界检验（oracle ceiling）—— 回答一个必须先回答的问题：

    「适配率没到 85%，到底是**系统选错了难度**，还是**这个口径本身就测不出来**？」

做法：给调度方一个**作弊的 oracle** —— 直接把学习者模拟器的**隐藏真值掌握度**
交给它来选难度（现实系统不可能拿到），再比较三种口径下的心流区间命中率：

    ① 系统实际交付          —— 用诊断估计选难度，作答实测口径
    ② oracle 选难度（真值）  —— 上限：即使诊断完全准确，实测口径下能到多少
    ③ oracle 选难度（真值）  —— 但按**预测**正确率判定（决策口径的上限）

结论的读法
----------
若 ② 仍显著低于 85%，那么「实测口径适配率」的不足**与调度策略无关**，
它由两个不可消除的测量误差决定：
    · 每次投放只答 5 题 ⇒ 正确率被量化到 0/0.2/0.4/0.6/0.8/1.0，二项标准差 ≈ 0.2；
    · 模拟器真值与诊断估计之间按 (画像,知识点) 存在 ±0.12 的确定性偏差。
因此**离线环境不具备验证「画像-难度适配准确率 ≥ 85%」的能力**，
该项必须用真实学习者作答数据（或真实后端 + 真人作答）复测。
本脚本把这个结论**量化**成一张表，而不是写在局限里的一句话。

用法：python eval/oracle_ceiling.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from engine.agents import DiagnoserAgent, SchedulerAgent   # noqa: E402
from engine.kb import load                                 # noqa: E402
from engine.llm import get_backend                         # noqa: E402
from engine.orchestrator import DIFF_FACTOR, LearnerSim, Orchestrator  # noqa: E402
from run import list_profiles, load_profile                # noqa: E402

LEVELS = ('基础', '进阶', '挑战')


def in_band(x):
    return 0.60 <= x <= 0.80


def ceiling(profile, sim, mastery, n_items):
    """按给定掌握度选难度（就近心流区间中心），返回 (命中数, 交付数)。"""
    hit = tot = 0
    for kp, m in mastery.items():
        # 与 SchedulerAgent.pick_difficulty 同一条准则：预测正确率最靠近 0.70 的档位
        best = min(LEVELS, key=lambda d: abs(m * DIFF_FACTOR[d] - 0.70))
        tot += 1
        c, n = sim.quiz(kp, best, n_items)
        hit += in_band(c / float(n))
    return hit, tot


def main():
    kb, llm = load(), get_backend('mock')
    a1 = DiagnoserAgent(kb, llm)
    print('=' * 78)
    print('适配率上界检验：给调度方一个「知道真值」的 oracle，看实测口径能到多少')
    print('=' * 78)
    print('%-7s %-6s %-26s %-12s %-12s %s'
          % ('画像', '题量', '口径', '命中/交付', '适配率', '说明'))
    rows = []
    for pid in list_profiles():
        prof = load_profile(pid)
        diag = a1.diagnose(prof)
        # 顺序有讲究：DIFF_REQUIRED_BY_KP 由 Orchestrator 初始化时按知识库填充，
        # 而 LearnerSim 在构造时就要用它算真值表 —— 所以先建编排器再建模拟器。
        orc = Orchestrator(backend='mock', rounds=3, budget_kp=3, verbose=False)
        sim = LearnerSim(prof)
        sess = orc.run(prof, write=False)
        s = sess['summary']
        rows.append({'画像': pid,
                     '题量': sess['rounds'][0]['items'][0]['feedback']['total'],
                     '口径': '① 系统实际交付（诊断估计）',
                     '命中': round(s['适配率(心流区间[0.6,0.8])'] *
                                 s['涉及知识点数'], 1),
                     '交付': s['涉及知识点数'],
                     '适配率': s['适配率(心流区间[0.6,0.8])'],
                     '说明': '报告里的那个数'})
        print('%-7s %-6d %-26s %-12s %-12.3f %s'
              % (pid, sess['rounds'][0]['items'][0]['feedback']['total'],
                 '① 系统实际交付（诊断估计）',
                 '%d 项' % s['涉及知识点数'],
                 s['适配率(心流区间[0.6,0.8])'], '报告里的那个数'))

        for n in (5, 20):
            h, t = ceiling(prof, sim, sim.true_mastery, n)
            tag = '② oracle选难度(真值)'
            print('%-7s %-6d %-26s %-12s %-12.3f %s'
                  % (pid, n, tag, '%d/%d' % (h, t), h / float(t),
                     '上限：诊断完全准确时的实测口径'))
            rows.append({'画像': pid, '题量': n, '口径': tag,
                         '命中': h, '交付': t, '适配率': round(h / float(t), 4),
                         '说明': '上限：诊断完全准确时的实测口径'})
            h2, t2 = ceiling(prof, sim, diag['mastery'], n)
            print('%-7s %-6d %-26s %-12s %-12.3f %s'
                  % (pid, n, '③ 估计选难度（对照）', '%d/%d' % (h2, t2),
                     h2 / float(t2), '与②的差 = 诊断误差的代价'))
            rows.append({'画像': pid, '题量': n, '口径': '③ 估计选难度（对照）',
                         '命中': h2, '交付': t2,
                         '适配率': round(h2 / float(t2), 4),
                         '说明': '与②的差 = 诊断误差的代价'})

    # 决策口径：不涉及学习者作答，只看"系统选的难度是否让预测正确率落在区间内"
    print('-' * 78)
    dec = []
    for pid in list_profiles():
        sess = Orchestrator(backend='mock', rounds=3, budget_kp=3,
                            verbose=False).run(load_profile(pid), write=False)
        dec.append(sess['summary']['适配准确率(决策口径)'])
    print('④ 决策口径（系统侧，不依赖作答）：%s ｜ 均值 %.3f'
          % ('、'.join('%s %.3f' % (p, v)
                       for p, v in zip(list_profiles(), dec)), sum(dec) / len(dec)))

    out = os.path.join(ROOT, 'out', 'eval')
    os.makedirs(out, exist_ok=True)
    import json
    with open(os.path.join(out, 'oracle_ceiling.json'), 'w', encoding='utf-8') as f:
        json.dump({'rows': rows, '决策口径': dict(zip(list_profiles(), dec))},
                  f, ensure_ascii=False, indent=2)
    print('\n产物：%s' % os.path.join(out, 'oracle_ceiling.json'))
    print('\n读法：若②（oracle 上界）明显低于 85%，说明实测口径的不足来自')
    print('      「5 题作答的量化误差」与「模拟器真值 vs 诊断估计的 ±0.12 偏差」，')
    print('      而不是调度策略 —— 该口径必须用真实学习者数据复测。')


if __name__ == '__main__':
    main()
