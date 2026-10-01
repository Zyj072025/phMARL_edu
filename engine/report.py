# -*- coding: utf-8 -*-
"""学情-资源匹配度报告（可视化）—— 交付物「可视化 学情-资源匹配度报告」。

三张图，全部由本次闭环的**真实运行数据**生成（不是示意图）：

1. 盲区定位图      各知识点掌握度估计 + 盲区阈值线 ⇒ 直接读出"先补哪里"
2. 难度匹配曲线    难度档 × 实测正确率 + 心流区间带 ⇒ 直接读出"难度配得准不准"
3. 路径规划图      轮次 × 难度档的目标路径 ⇒ 直接读出"下一步学什么"

输出为 **单文件 HTML + 内联 SVG**：零依赖、零 CDN，双击即开，任何机器上都能看。
图表配色与 PPT 一致（深蓝/青/珊瑚/金），图中不出现 C1..Cn 之类的代号。
"""
import json
import time

# ── 配色（与 PPT 同一套） ────────────────────────────────────────────────
NAVY, TEAL, CORAL, GOLD = '#1F3B57', '#2E8B8B', '#EE6A5F', '#C88A2E'
INK, INK2, MUTED, LIGHT, BAND = '#1B2733', '#48596B', '#8494A6', '#EEF3F6', '#DCEDEA'
DIFF_Y = {'基础': 40, '进阶': 96, '挑战': 152}          # 难度档在图上的纵向位置


# ══════════════════════════════════════════════════════════════ 报告数据
def build_report(session, kb):
    """把一次闭环会话整理成报告结构（纯数据，便于程序读取与再绘图）。"""
    # ---- 1 盲区定位 ----
    # 掌握度用的是**诊断阶段的快照**（trace 第①步的完整掌握度表），
    # 而不是被反馈回写后的更新值：盲区定位图刻画的是"诊断看到了什么"。
    diag_mastery = {}
    for rec in session['trace']:
        if rec.get('step', '').startswith('①'):
            diag_mastery = dict(rec['data'].get('mastery', {}))
            break
    blind = sorted([(k, v) for k, v in diag_mastery.items() if v < 0.60],
                   key=lambda x: x[1])
    strong = sorted([(k, v) for k, v in diag_mastery.items() if v >= 0.60],
                    key=lambda x: -x[1])
    blind_rows = [{'知识点': k, '掌握度': v, '模块': _module_of(kb, k)}
                  for k, v in blind]
    # 盲区定位图只画最需要关注的 8 个（若盲区不足 8 个，则补足掌握度最低的强项一并展示）
    show = blind_rows[:8]
    if len(show) < 8:
        show = show + [{'知识点': k, '掌握度': v, '模块': _module_of(kb, k),
                        '已达掌握': True} for k, v in strong[:8 - len(show)]]

    # ---- 2 难度匹配曲线 ----
    curve = []
    for rd in session['rounds']:
        for it in rd['items']:
            curve.append({
                '轮次': rd['round'], '知识点': it['knowledge_point'],
                '难度档': it['difficulty'],
                '掌握度估计': round(_plan_mastery(rd, it), 4),
                '实测正确率': it['feedback']['correct_rate'],
                '答题': '%d/%d' % (it['feedback']['correct'], it['feedback']['total']),
                '落入心流区间': 0.60 <= it['feedback']['correct_rate'] <= 0.80,
            })

    # ---- 3 路径规划 ----
    path_nodes, path_edges = [], []
    pos = {}
    for rd in session['rounds']:
        for it in rd['items']:
            nid = 'R%d-%s' % (rd['round'], it['knowledge_point'])
            pos[it['knowledge_point']] = pos.get(it['knowledge_point'], [])
            pos[it['knowledge_point']].append(nid)
            path_nodes.append({
                'id': nid, '轮次': rd['round'],
                '知识点': it['knowledge_point'], '难度档': it['difficulty'],
                '实测正确率': it['feedback']['correct_rate'],
                '教学动作': rd['feedback']['action'] if rd['feedback'] else '',
            })
    for kp, ids in pos.items():
        for a, b in zip(ids, ids[1:]):
            path_edges.append({'from': a, 'to': b, 'type': '同一知识点跨轮'})

    # ---- 指标 ----
    s = session['summary']
    metrics = [
        {'指标': '投影前幻觉率（宽口径）', '值': '%.2f%%' % (s['投影前_幻觉率_宽口径'] * 100),
         '说明': '声明与知识库冲突的比例，投影前'},
        {'指标': '投影后幻觉率（宽口径）', '值': '%.2f%%' % (s['投影后_幻觉率_宽口径'] * 100),
         '说明': '经知识约束投影后的残余冲突比例'},
        {'指标': '投影介入率', '值': '%.1f%%' % (s['投影介入率'] * 100),
         '说明': '被替换/限定/删除的声明占比'},
        {'指标': '平均归一化编辑距离', '值': '%.4f' % s['平均归一化编辑距离'],
         '说明': '介入声明的平均改动幅度，越小越接近"最小改动"'},
        {'指标': '适配率（心流区间 60%~80%）', '值': '%.1f%%' % (s['适配率(心流区间[0.6,0.8])'] * 100),
         '说明': '实测正确率落进心流区间的（学习者,知识点）对占比'},
        {'指标': '资源形态数', '值': '%d 类' % s['资源形态数'], '说明': '讲义/指南/测试题'},
        {'指标': '涉及知识点数', '值': '%d 个' % s['涉及知识点数'],
         '说明': '本轮闭环实际产出的知识点数'},
        {'指标': '全库知识点覆盖率', '值': '%.1f%%' % (s['全库知识点覆盖率'] * 100),
         '说明': '占知识库全部 %d 个知识点的比例' % len(kb.docs)},
        {'指标': '交叉验证争议数', '值': '%d 条' % s['交叉验证争议数'],
         '说明': '两视角判定不一致、经第三视角裁决的声明数'},
        {'指标': '通信稀疏度', '值': '%.1f%%' % (s['门控']['通信稀疏度(发送/候选)'] * 100),
         '说明': '门控实际发送 / 候选消息数'},
        {'指标': 'token 节省率', '值': '%.1f%%' % (s['门控']['token节省率(按字符数)'] * 100),
         '说明': '门控相对全通信省下的消息字符占比'},
    ]

    return {
        'profile_id': session['profile_id'],
        '生成的时刻': time.strftime('%Y-%m-%d %H:%M:%S'),
        '后端': session['backend'],
        '反馈来源': session['feedback_source'],
        '门控阈值': session['gate_threshold'],
        '盲区定位': {'阈值': 0.60, '盲区数': len(blind_rows),
                     '全部掌握度': [{'知识点': k, '掌握度': v}
                                for k, v in sorted(diag_mastery.items(),
                                                   key=lambda x: x[1])],
                     '展示行': show},
        '难度匹配曲线': curve,
        '路径规划': {'节点': path_nodes, '边': path_edges},
        '指标': metrics,
        '原始指标': s,
    }


def _module_of(kb, kp):
    for d in kb.docs:
        if d['title'] == kp:
            return d.get('module', '-')
    return '-'


def _plan_mastery(rd, it):
    for t in rd['plan']['targets']:
        if t['knowledge_point'] == it['knowledge_point']:
            return t['mastery']
    return 0.0


# ══════════════════════════════════════════════════════════════ 绘图
def _svg_blind(rows, w=880):
    """盲区定位图：横向条形 + 0.6 阈值线。"""
    h = 60 + len(rows) * 34
    x0, x1 = 300, w - 70
    y = 46
    p = ['<svg viewBox="0 0 %d %d" width="100%%" role="img" '
         'aria-label="盲区定位图">' % (w, h)]
    p.append('<text x="0" y="18" font-size="13" fill="%s">知识点掌握度估计'
             '（条越短越需要优先补；虚线为盲区阈值 0.60）</text>' % INK2)
    # 阈值线
    tx = x0 + (x1 - x0) * 0.60
    p.append('<line x1="%.1f" y1="34" x2="%.1f" y2="%d" stroke="%s" '
             'stroke-dasharray="4 3" stroke-width="1.5"/>' % (tx, tx, h - 14, GOLD))
    p.append('<text x="%.1f" y="%d" font-size="11" fill="%s">0.60</text>'
             % (tx + 4, h - 4, GOLD))
    for i, r in enumerate(rows):
        v = r['掌握度']
        yy = y + i * 34
        bl = r['知识点']
        label = bl if len(bl) <= 16 else bl[:15] + '…'
        p.append('<text x="0" y="%d" font-size="12" fill="%s">%s</text>'
                 % (yy + 4, INK, _esc(label)))
        bw = max(2.0, (x1 - x0) * v)
        col = CORAL if v < 0.60 else TEAL
        p.append('<rect x="%d" y="%d" width="%.1f" height="14" rx="4" fill="%s"/>'
                 % (x0, yy - 8, bw, col))
        p.append('<text x="%.1f" y="%d" font-size="11" fill="%s">%.3f</text>'
                 % (x0 + bw + 6, yy + 4, INK2, v))
    p.append('</svg>')
    return '\n'.join(p)


def _svg_curve(curve, w=880):
    """难度匹配曲线：x = 难度档，y = 实测正确率，阴影带 = 心流区间 60%~80%。"""
    h = 300
    padl, padr, padt, padb = 62, 92, 46, 46
    xs = {'基础': 0, '进阶': 1, '挑战': 2}
    x0, x1 = padl, w - padr
    y0, y1 = padt, h - padb

    def X(d):
        return x0 + (x1 - x0) * (xs.get(d, 1) / 2.0)

    def Y(v):
        return y1 - (y1 - y0) * min(max(v, 0.0), 1.0)

    p = ['<svg viewBox="0 0 %d %d" width="100%%" role="img" '
         'aria-label="难度匹配曲线">' % (w, h)]
    # 心流区间带
    p.append('<rect x="%d" y="%.1f" width="%d" height="%.1f" fill="%s"/>'
             % (x0, Y(0.80), x1 - x0, Y(0.60) - Y(0.80), BAND))
    p.append('<text x="%d" y="%.1f" font-size="11" fill="%s">心流区间 60%%~80%%'
             '（此区间内的难度匹配为"适配"）</text>' % (x1 - 300, Y(0.80) - 6, TEAL))
    p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" '
             'stroke-width="1"/>' % (x0, Y(0.60), x1, Y(0.60), TEAL))
    p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" '
             'stroke-width="1"/>' % (x0, Y(0.80), x1, Y(0.80), TEAL))
    # 轴
    p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s"/>'
             % (x0, y1, x1, y1, MUTED))
    p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s"/>'
             % (x0, y0, x0, y1, MUTED))
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        p.append('<text x="%d" y="%.1f" font-size="11" fill="%s" '
                 'text-anchor="end">%.0f%%</text>' % (padl - 8, Y(v) + 4, MUTED, v * 100))
    for d in ('基础', '进阶', '挑战'):
        p.append('<text x="%.1f" y="%d" font-size="12" fill="%s" '
                 'text-anchor="middle">%s</text>' % (X(d), h - 24, INK2, d))
    p.append('<text x="%d" y="%d" font-size="12" fill="%s">难度档</text>'
             % ((x0 + x1) // 2 - 18, h - 6, INK2))
    p.append('<text x="14" y="%d" font-size="12" fill="%s" '
             'transform="rotate(-90 14 %d)">实测正确率</text>'
             % ((y0 + y1) // 2, INK2, (y0 + y1) // 2))
    # 按轮次连成折线（同一轮的点用同一根线；颜色区分轮次）
    rounds = sorted(set(c['轮次'] for c in curve))
    cols = {1: NAVY, 2: CORAL, 3: GOLD}
    for r in rounds:
        pts = sorted([c for c in curve if c['轮次'] == r], key=lambda c: xs[c['难度档']])
        if len(pts) >= 2:
            d = ' '.join('%.1f,%.1f' % (X(c['难度档']), Y(c['实测正确率'])) for c in pts)
            p.append('<polyline points="%s" fill="none" stroke="%s" '
                     'stroke-width="2"/>' % (d, cols.get(r, NAVY)))
    # 数据点（含 2px 底色描边，避免与阴影带糊在一起）
    for c in curve:
        cx, cy = X(c['难度档']), Y(c['实测正确率'])
        col = TEAL if c['落入心流区间'] else CORAL
        p.append('<circle cx="%.1f" cy="%.1f" r="5" fill="%s" stroke="#FFFFFF" '
                 'stroke-width="2"><title>%s｜%s档｜正确率 %s</title></circle>'
                 % (cx, cy, col, _esc(c['知识点']), c['难度档'], c['答题']))
        if abs(c['实测正确率'] - 0.6) < 0.25 or not c['落入心流区间']:
            p.append('<text x="%.1f" y="%.1f" font-size="10" fill="%s">%s</text>'
                     % (cx + 8, cy + 3, INK2, _esc(c['答题'])))
    # 图例（单序列 + 轮次）
    lx = x1 - 78
    for i, r in enumerate(rounds):
        p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="2"/>'
                 % (lx, 20 + i * 16, lx + 16, 20 + i * 16, cols.get(r, NAVY)))
        p.append('<text x="%d" y="%d" font-size="11" fill="%s">第 %d 轮</text>'
                 % (lx + 20, 24 + i * 16, INK2, r))
    p.append('<text x="%d" y="%d" font-size="11" fill="%s">'
             '点色：青=适配　珊瑚=未适配</text>' % (lx - 40, 20 + len(rounds) * 16 + 4, MUTED))
    p.append('</svg>')
    return '\n'.join(p)


def _svg_path(nodes, edges, w=880):
    """路径规划图：横轴 = 轮次，纵轴 = 难度档；同一知识点跨轮的推进用连线表示。"""
    h = 260
    padl, padr, padt, padb = 90, 120, 34, 40
    x0, x1 = padl, w - padr
    y0, y1 = padt, h - padb
    rounds = sorted(set(n['轮次'] for n in nodes)) or [1]

    def X(r):
        return x0 if len(rounds) == 1 else x0 + (x1 - x0) * (
            rounds.index(r) / float(len(rounds) - 1))

    def Y(d):
        return y0 + (y1 - y0) * (1.0 - (DIFF_Y.get(d, 96) - 20) / 160.0)

    p = ['<svg viewBox="0 0 %d %d" width="100%%" role="img" '
         'aria-label="路径规划图">' % (w, h)]
    p.append('<rect x="%d" y="%d" width="%d" height="%d" fill="%s" rx="6"/>'
             % (x0 - 40, y0 - 14, x1 - x0 + 80, y1 - y0 + 28, LIGHT))
    for d in ('基础', '进阶', '挑战'):
        yy = Y(d)
        p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#FFFFFF" '
                 'stroke-width="1"/>' % (x0 - 40, yy, x1 + 40, yy))
        p.append('<text x="%d" y="%.1f" font-size="12" fill="%s" '
                 'text-anchor="end">%s</text>' % (x0 - 50, yy + 4, INK2, d))
    for r in rounds:
        p.append('<text x="%.1f" y="%d" font-size="12" fill="%s" '
                 'text-anchor="middle">第 %d 轮</text>' % (X(r), h - 14, INK2, r))
    idx = {n['id']: n for n in nodes}
    for e in edges:
        a, b = idx.get(e['from']), idx.get(e['to'])
        if not a or not b:
            continue
        p.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
                 'stroke-width="1.5" stroke-dasharray="3 3"/>'
                 % (X(a['轮次']), Y(a['难度档']), X(b['轮次']), Y(b['难度档']), MUTED))
    for n in nodes:
        cx, cy = X(n['轮次']), Y(n['难度档'])
        col = TEAL if 0.60 <= n['实测正确率'] <= 0.80 else CORAL
        p.append('<circle cx="%.1f" cy="%.1f" r="7" fill="%s" stroke="#FFFFFF" '
                 'stroke-width="2"><title>%s｜%s｜正确率 %.0f%%｜%s</title></circle>'
                 % (cx, cy, col, _esc(n['知识点']), n['难度档'],
                    n['实测正确率'] * 100, _esc(n['教学动作'])))
        lbl = n['知识点'] if len(n['知识点']) <= 12 else n['知识点'][:11] + '…'
        p.append('<text x="%.1f" y="%.1f" font-size="11" fill="%s" '
                 'text-anchor="middle">%s</text>'
                 % (cx, cy - 12, INK, _esc(lbl)))
    p.append('<text x="%d" y="%d" font-size="11" fill="%s">'
             '实心点=该轮目标知识点；虚线=同一知识点跨轮推进</text>'
             % (x0 - 40, y0 - 20, MUTED))
    p.append('</svg>')
    return '\n'.join(p)


def _esc(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


# ══════════════════════════════════════════════════════════════ HTML
def render_html(rep, session):
    blind = rep['盲区定位']
    rows = [{'知识点': r['知识点'], '掌握度': r['掌握度']} for r in blind['展示行']]
    mrows = ''.join('<tr><td>%s</td><td class="v">%s</td><td>%s</td></tr>'
                    % (_esc(m['指标']), _esc(m['值']), _esc(m['说明']))
                    for m in rep['指标'])
    crows = ''.join(
        '<tr><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'
        % (c['轮次'], _esc(c['知识点']), _esc(c['难度档']),
           '%.3f' % c['掌握度估计'], c['答题'],
           '是' if c['落入心流区间'] else '否')
        for c in rep['难度匹配曲线'])
    return """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>学情-资源匹配度报告 — %(pid)s</title>
<style>
 :root{--navy:%(navy)s;--teal:%(teal)s;--coral:%(coral)s;--ink:%(ink)s;--ink2:%(ink2)s;--light:%(light)s}
 *{box-sizing:border-box}
 body{margin:0;background:#F7F9FA;color:var(--ink);
      font-family:"Microsoft YaHei","PingFang SC","Source Han Sans SC",sans-serif;line-height:1.7}
 .wrap{max-width:1000px;margin:0 auto;padding:0 0 60px}
 header{background:var(--navy);color:#fff;padding:26px 34px}
 header h1{margin:0;font-size:21px;font-weight:600}
 header .sub{opacity:.82;font-size:13px;margin-top:6px}
 .card{background:#fff;margin:18px 24px;padding:22px 26px;border-radius:10px;
       box-shadow:0 1px 3px rgba(31,59,87,.10)}
 .card h2{margin:0 0 4px;font-size:16px;color:var(--navy)}
 .card .cap{font-size:12.5px;color:var(--ink2);margin:0 0 16px}
 table{border-collapse:collapse;width:100%%;font-size:13px}
 th,td{border-bottom:1px solid #E6ECF1;padding:7px 10px;text-align:left}
 th{background:var(--light);color:var(--navy);font-weight:600}
 td.v{font-variant-numeric:tabular-nums;color:var(--navy);font-weight:600}
 .kv{display:flex;flex-wrap:wrap;gap:10px 26px;font-size:13px;color:var(--ink2)}
 .kv b{color:var(--navy)}
 .note{background:#FFF8EC;border-left:3px solid %(gold)s;padding:10px 14px;
       font-size:12.5px;color:var(--ink2);margin-top:14px;border-radius:0 6px 6px 0}
 footer{text-align:center;font-size:12px;color:#8494A6;margin-top:26px}
</style></head><body><div class="wrap">
<header>
  <h1>学情-资源匹配度报告</h1>
  <div class="sub">学习者 %(pid)s ｜ 生成于 %(ts)s ｜ 后端 %(backend)s ｜ 反馈来源 %(sim)s</div>
</header>

<div class="card">
  <h2>一、盲区定位图</h2>
  <p class="cap">掌握度低于 0.60 的知识点即为盲区，按掌握度升序排列——越靠上越应先补。
     共检出盲区 <b>%(nblind)d</b> 个。</p>
  %(svg1)s
</div>

<div class="card">
  <h2>二、难度匹配曲线</h2>
  <p class="cap">横轴为资源难度档，纵轴为该难度下的实测答题正确率。
     落在浅色带内（60%%~80%%）表示难度与能力匹配（心流区间）；
     高于上沿说明偏易、低于下沿说明偏难——这正是下一轮调整难度档的依据。</p>
  %(svg2)s
  <table><thead><tr><th>轮次</th><th>知识点</th><th>难度档</th>
    <th>掌握度估计</th><th>答题</th><th>落入心流区间</th></tr></thead>
    <tbody>%(crows)s</tbody></table>
</div>

<div class="card">
  <h2>三、路径规划图</h2>
  <p class="cap">横轴为闭环轮次，纵轴为难度档。每个点是一次"针对某知识点的资源投放"，
     绿色表示该次投放的难度已匹配，红色表示需要下一轮调整。</p>
  %(svg3)s
</div>

<div class="card">
  <h2>四、关键指标</h2>
  <p class="cap">全部指标来自本次闭环的真实运行记录，口径在 <code>engine/orchestrator.py
     ::summarize_session</code> 中唯一定义。</p>
  <table><thead><tr><th>指标</th><th>数值</th><th>口径说明</th></tr></thead>
  <tbody>%(mrows)s</tbody></table>
  <div class="note">反馈信号来源：<b>%(sim)s</b>。离线演示环境没有真实学习者，
     因此作答由确定性模拟器产生；真实部署时替换为真实作答数据，接口不变。
     此外，若使用离线后端（mock），幻觉率衡量的是<b>审核与投影机制的有效性</b>，
     而非真实大模型的幻觉率——真实数值请用 <code>--backend openai</code> 测量。</div>
</div>

<footer>Ph-MARL-Edu ｜ 四智能体协同闭环 ｜ 本报告为单文件 HTML，零外部依赖</footer>
</div></body></html>""" % {
        'pid': _esc(rep['profile_id']), 'ts': rep['生成的时刻'],
        'backend': _esc(rep['后端']), 'sim': _esc(rep['反馈来源']),
        'navy': NAVY, 'teal': TEAL, 'coral': CORAL, 'gold': GOLD, 'ink': INK,
        'ink2': INK2, 'light': LIGHT,
        'nblind': blind['盲区数'],
        'svg1': _svg_blind(rows), 'svg2': _svg_curve(rep['难度匹配曲线']),
        'svg3': _svg_path(rep['路径规划']['节点'], rep['路径规划']['边']),
        'crows': crows, 'mrows': mrows,
    }


if __name__ == '__main__':
    import sys
    import os
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine.orchestrator import Orchestrator
    prof = {'profile_id': 'L001', 'education': '本科大三 人工智能专业',
            'theory_score': '62/100', 'hands_on': '有 Python，无 RL 经验',
            'history_accuracy': 0.58}
    o = Orchestrator(backend='mock', rounds=2, verbose=False)
    s = o.run(prof)
    rep = build_report(s, o.kb)
    print(json.dumps(rep['指标'], ensure_ascii=False, indent=2))
    print('盲区数:', rep['盲区定位']['盲区数'])
    print('曲线点数:', len(rep['难度匹配曲线']))
