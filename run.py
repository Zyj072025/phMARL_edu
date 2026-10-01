# -*- coding: utf-8 -*-
"""Ph-MARL-Edu 统一入口 —— 领域知识个性化生成与多智能体协同决策系统。

零第三方依赖（纯 Python 标准库），一条命令跑通「分析→生成→校验→决策→反馈」闭环：

    python run.py --profile L001
    python run.py --all-profiles
    python run.py --ui                    # 浏览器界面（内置 http.server）
    python run.py --backend openai        # 接真实大模型（需 PHMARL_LLM_API_KEY）

产物落在 out/<画像ID>/：资源包、审核明细、可审计链路、匹配度报告（HTML）。
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from engine.gating import FullCommGate, Gate          # noqa: E402
from engine.orchestrator import Orchestrator           # noqa: E402

PROFILE_DIR = os.path.join(ROOT, 'data', 'profiles')


# ─────────────────────────────────────────────────────────── 画像加载
def list_profiles():
    if not os.path.isdir(PROFILE_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(PROFILE_DIR) if f.endswith('.json'))


def load_profile(spec):
    """支持三种写法：画像 ID（L001）、文件名、完整路径。"""
    cands = [spec, spec + '.json', os.path.join(PROFILE_DIR, spec),
             os.path.join(PROFILE_DIR, spec + '.json')]
    for c in cands:
        if os.path.isfile(c):
            with open(c, encoding='utf-8') as f:
                return json.load(f)
    raise SystemExit('找不到画像：%s\n可用画像：%s'
                     % (spec, '、'.join(list_profiles())))


# ─────────────────────────────────────────────────────────── 单次运行
def run_one(profile, args, out_root):
    gate = FullCommGate() if args.full_comm else Gate(
        threshold=args.gate_threshold, verbose=args.verbose)
    orc = Orchestrator(backend=args.backend, gate=gate,
                       budget_kp=(None if args.budget_kp <= 0 else args.budget_kp),
                       rounds=args.rounds, eps=args.eps, verbose=args.verbose)
    if orc.budget_kp is None:
        orc.budget_kp = 10 ** 9                  # 全量：所有盲区都出资源
        orc.a4.budget_kp = 10 ** 9
    out_dir = os.path.join(out_root, profile.get('profile_id', 'profile'))
    sess = orc.run(profile, out_dir=out_dir)
    return sess, out_dir, orc


def print_summary(sess, out_dir):
    s = sess['summary']
    g = s['门控']
    print('')
    print('═' * 68)
    print('画像 %s ｜ 后端 %s ｜ 反馈来源 %s ｜ 用时 %.1fs'
          % (sess['profile_id'], sess['backend'], sess['feedback_source'],
             sess['elapsed_sec']))
    print('─' * 68)
    print('  闭环轮数            %d' % s['轮数'])
    print('  涉及知识点          %d 个（全库覆盖率 %.1f%%）'
          % (s['涉及知识点数'], s['全库知识点覆盖率'] * 100))
    print('  资源形态            %d 类' % s['资源形态数'])
    print('  声明总数            %d 条' % s['声明总数'])
    print('  幻觉率（宽口径）    %.2f%% → %.2f%%   投影前 → 投影后'
          % (s['投影前_幻觉率_宽口径'] * 100, s['投影后_幻觉率_宽口径'] * 100))
    print('  幻觉率（严口径）    %.2f%%' % (s['投影前_幻觉率_严口径'] * 100))
    print('  投影介入率          %.1f%%  平均编辑距离 %.4f'
          % (s['投影介入率'] * 100, s['平均归一化编辑距离']))
    print('  交叉验证争议        %d 条（经第三视角裁决）' % s['交叉验证争议数'])
    print('  适配率（心流区间）  %.1f%%' % (s['适配率(心流区间[0.6,0.8])'] * 100))
    print('  通信稀疏度          %.1f%%   token 节省率 %.1f%%'
          % (g['通信稀疏度(发送/候选)'] * 100, g['token节省率(按字符数)'] * 100))
    print('─' * 68)
    print('  产物目录            %s' % out_dir)
    print('  · resources.md      个性化资源包（交付文本）')
    print('  · audit_trace.json  声明级审核明细（含证据与投影记录）')
    print('  · trace.jsonl       闭环每一步的可审计链路')
    print('  · report.html       学情-资源匹配度报告（双击打开）')
    print('═' * 68)


# ─────────────────────────────────────────────────────────── 浏览器界面
def serve_ui(args, out_root):
    import http.server
    import socketserver
    import urllib.parse

    os.makedirs(out_root, exist_ok=True)
    state = {'backend': args.backend, 'rounds': args.rounds,
             'gate_threshold': args.gate_threshold, 'eps': args.eps}

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            http.server.SimpleHTTPRequestHandler.__init__(
                self, *a, directory=out_root, **kw)

        def log_message(self, fmt, *a):
            if args.verbose:
                sys.stderr.write('[UI] ' + fmt % a + '\n')

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            if u.path in ('/', '/index.html'):
                return self._index()
            if u.path == '/run':
                return self._run(q.get('p', [''])[0])
            return http.server.SimpleHTTPRequestHandler.do_GET(self)

        def _index(self):
            rows = []
            for pid in list_profiles():
                try:
                    prof = load_profile(pid)
                except SystemExit:
                    continue
                rows.append(
                    '<tr><td class="id">%s</td><td>%s</td><td>%s</td>'
                    '<td>%s</td><td>%s</td>'
                    '<td><a class="btn" href="/run?p=%s">运行闭环</a></td></tr>'
                    % (pid, prof.get('education', '-'),
                       prof.get('theory_score', '-'), prof.get('hands_on', '-'),
                       prof.get('history_accuracy', '-'), pid))
            body = '''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Ph-MARL-Edu 控制台</title><style>
body{margin:0;background:#F7F9FA;color:#1B2733;font-family:"Microsoft YaHei",sans-serif}
header{background:#1F3B57;color:#fff;padding:24px 34px}
header h1{margin:0;font-size:20px}header .s{opacity:.8;font-size:13px;margin-top:6px}
.card{background:#fff;margin:20px 24px;padding:22px 26px;border-radius:10px;
box-shadow:0 1px 3px rgba(31,59,87,.1)}
table{border-collapse:collapse;width:100%%;font-size:13px}
th,td{border-bottom:1px solid #E6ECF1;padding:9px 10px;text-align:left}
th{background:#EEF3F6;color:#1F3B57}
.id{font-weight:600;color:#1F3B57}
a.btn{display:inline-block;background:#2E8B8B;color:#fff;text-decoration:none;
padding:5px 14px;border-radius:6px;font-size:12.5px}
a.btn:hover{background:#26706F}
.note{font-size:12.5px;color:#48596B;margin-top:14px}
</style></head><body>
<header><h1>Ph-MARL-Edu 控制台</h1>
<div class="s">领域知识个性化生成与多智能体协同决策系统 ｜
后端 %s ｜ 门控阈值 %.2f ｜ 闭环轮数 %d</div></header>
<div class="card"><table><thead><tr><th>画像</th><th>学历/专业</th>
<th>理论分</th><th>实操经历</th><th>历史正确率</th><th>操作</th></tr></thead>
<tbody>%s</tbody></table>
<div class="note">点击「运行闭环」即现场跑一次「分析→生成→校验→决策→反馈」，
结束后自动跳到该画像的学情-资源匹配度报告。
报告与资源包也可在 out/&lt;画像ID&gt;/ 目录下直接打开。</div>
</div></body></html>''' % (state['backend'], state['gate_threshold'],
                          state['rounds'], ''.join(rows) or
                          '<tr><td colspan="6">未找到画像文件</td></tr>')
            data = body.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _run(self, pid):
            if not pid:
                return self._redirect('/')
            try:
                prof = load_profile(pid)
                gate = FullCommGate() if args.full_comm else Gate(
                    threshold=state['gate_threshold'], verbose=False)
                orc = Orchestrator(backend=state['backend'], gate=gate,
                                   budget_kp=(10 ** 9 if args.budget_kp <= 0
                                              else args.budget_kp),
                                   rounds=state['rounds'], eps=state['eps'],
                                   verbose=False)
                orc.run(prof, out_dir=os.path.join(out_root, pid))
            except Exception as e:                        # 出错也要让人看见
                msg = ('<meta charset="utf-8"><p>运行失败：%s</p>'
                       '<p><a href="/">返回</a></p>' % str(e).replace('<', '&lt;'))
                data = msg.encode('utf-8')
                self.send_response(500)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            return self._redirect('/%s/report.html' % pid)

        def _redirect(self, to):
            self.send_response(302)
            self.send_header('Location', to)
            self.send_header('Content-Length', '0')
            self.end_headers()

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    port = args.port
    for attempt in range(20):                             # 端口占用则顺延
        try:
            httpd = Server(('127.0.0.1', port), Handler)
            break
        except OSError:
            port += 1
    else:
        raise SystemExit('无可用端口（%d 起试了 20 个）' % args.port)

    url = 'http://127.0.0.1:%d/' % port
    print('')
    print('═' * 68)
    print('  Ph-MARL-Edu 控制台已启动： %s' % url)
    print('  在浏览器打开上面的地址，选一个画像点「运行闭环」。')
    print('  停止服务：在本窗口按 Ctrl+C')
    print('═' * 68)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n已停止。')
        httpd.server_close()


# ─────────────────────────────────────────────────────────── CLI
def main():
    ap = argparse.ArgumentParser(
        description='Ph-MARL-Edu：领域知识个性化生成与多智能体协同决策系统（零依赖）',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--profile', default='L001',
                    help='画像 ID / 文件名 / 路径（默认 L001）')
    ap.add_argument('--all-profiles', action='store_true', help='跑全部画像')
    ap.add_argument('--backend', default='mock', choices=['mock', 'openai'],
                    help='LLM 后端（mock 离线确定性；openai 走真实大模型）')
    ap.add_argument('--rounds', type=int, default=2, help='闭环轮数（默认 2）')
    ap.add_argument('--budget-kp', type=int, default=2,
                    help='每轮生成资源的知识点数；<=0 表示全部盲区（默认 2）')
    ap.add_argument('--gate-threshold', type=float, default=0.3,
                    help='门控阈值 τ（越大越省通信，默认 0.3）')
    ap.add_argument('--full-comm', action='store_true',
                    help='消融对照：关闭门控，全通信')
    ap.add_argument('--eps', type=float, default=0.05,
                    help='投影可行性阈值（残余未合规比例上限，默认 0.05）')
    ap.add_argument('--out', default=os.path.join(ROOT, 'out'),
                    help='产物根目录（默认 out/）')
    ap.add_argument('--ui', action='store_true', help='启动浏览器界面')
    ap.add_argument('--port', type=int, default=8017, help='界面端口（默认 8017）')
    ap.add_argument('--json', action='store_true', help='额外打印 summary JSON')
    ap.add_argument('-v', '--verbose', action='store_true', help='打印每一步细节')
    args = ap.parse_args()

    if args.ui:
        return serve_ui(args, args.out)

    specs = list_profiles() if args.all_profiles else [args.profile]
    if not specs:
        raise SystemExit('data/profiles/ 下没有画像文件。')
    os.makedirs(args.out, exist_ok=True)
    allsum = []
    for spec in specs:
        sess, out_dir, _ = run_one(load_profile(spec), args, args.out)
        print_summary(sess, out_dir)
        allsum.append(sess['summary'])
    if args.json:
        print(json.dumps(allsum if len(allsum) > 1 else allsum[0],
                         ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
