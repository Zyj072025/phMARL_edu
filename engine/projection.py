# -*- coding: utf-8 -*-
"""知识约束投影（Knowledge-Constrained Projection）—— 本作品的核心创新点一。

形式化（与安全投影 CBF-QP 同构）
--------------------------------
前期科研中的安全投影：
        min_u ‖u − u_raw‖²   s.t.  A·u ≤ b   （即 h(x) ≥ 0）
即在**满足约束的可行集**内，找与期望动作**最接近**的动作。

转译到知识生成：
        min_{x′} EditDistance(x′, x_raw)   s.t.  每条声明 ∈ {SUPPORTED}
即在**每条声明均可溯源**的许可集内，找与原始生成**编辑距离最小**的改写。

工程实现（三步，每步都可审计）
------------------------------
1. **候选生成**：对每条被判 CONTRADICTED / UNVERIFIED 的声明，产生候选改写：
   - REPLACE  最小改写（由后端给出修正表述）
   - LIMIT    限定化（补适用范围/前提条件；真实后端下由 prompt 产出）
   - DELETE   删除并显式标注"证据不足"
2. **可行性判定（关键）**：候选必须**再审核一次**（re-audit），
   只有复审不再被 CONTRADICTED 的候选才算可行 —— 修正是**被验证过**的，
   不是"再生成一次赌它对"。
3. **最小改动选择**：在可行候选中按编辑距离取最小者。
   删除的编辑距离 = 该声明的长度，因此删除天然是"最后手段"，
   只有改写无法使其合规时才会被选中 —— 这与"最小改动"的目标一致。

补充：每条声明单独投影（逐条贪心），是全局最优的可计算近似；
声明之间若存在相互依赖的联合约束，需扩展为联合投影（本版本未实现，已在局限中注明）。
"""
import re

# ── 编辑距离（纯 Python，字符串短，性能足够） ────────────────────────────
def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# 距离相同时的算子偏好：改写 > 限定化 > 删除（尽量保留内容与个性化表达）
_OP_RANK = {'REPLACE': 0, 'LIMIT': 1, 'DELETE': 2}


def normalize_distance(orig, new):
    """归一化编辑距离 ∈ [0,1]；删除记为该声明被整体移除（距离 = 1.0）。

    定义：levenshtein(orig, new) / max(len(orig), len(new))，
    即"改动字符占较长一方的比例"。这样定义有两个必需的性质：
      · 天然落在 [0,1]（levenshtein ≤ max(len)），不需要额外截断；
      · "改写"的代价永远不会超过"整条删除"（=1.0），
        否则最小改动准则会退化成"一律删除"——把错误声明改写为一条更长的正确表述时，
        字面编辑距离虽可能超过原声明长度，但语义上不该比删掉它更贵。
    """
    if new is None:
        return 1.0
    return levenshtein(orig, new) / float(max(len(orig), len(new), 1))


# ── 候选生成 ────────────────────────────────────────────────────────────
def subject_overlap(claim, new_text):
    """「主体连续性」：改写后与改写前共享多少内容词元（∈[0,1]）。

    这是防止投影**引入新内容**的安全带。若不检查，最小改动准则会退化成
    "拿知识库里某条正确表述把原句整句换掉"——编辑距离看起来很小，
    但换上去的可能是与原文主题无关的另一条知识（例如把"先确认 Python 版本"
    换成"折扣因子的取值权衡"）。那不是改错，那是**替换主题**，
    比原来的错误更糟：它让资源在语义上不自洽。
    """
    from .kb import tokenize
    a = set(tokenize(claim))
    b = set(tokenize(new_text))
    if not a:
        return 0.0
    return len(a & b) / float(len(a))


REPLACE_SUBJECT_FLOOR = 0.25          # 低于此值不视为"最小改写"


def _corrective(claim, evidence, llm):
    """取该声明的修正表述。**必须同时满足两个条件**，缺一不可：

      1. 该证据条目在知识库里**带有已知误区**（即这条声明可能正是那类错误）；
      2. 该声明**确实与那条误区的表述一致**（相似度 ≥ FIXTURE_FLOOR）。

    历史教训（两次修正）：第一版允许在**全部**知识条目里做全局词元重叠搜索，
    结果给"确认 Python 版本"这类操作性表述匹配到了"折扣因子取值"的修正表述。
    第二版改成"只看该声明自己的证据"，但仍然只要证据里**碰巧**有一条带误区的
    文档就返回它的修正 —— 单元测试抓到：一条"γ 越大越好"的声明，因为检索到的
    证据里含「消息压缩」条目（那条目原文里恰好有"越大越好"），被改写成
    "消息维度不是越大越好…"。这不是改错，是**换了题目**，比原错误更糟。
    因此现在要求声明与误区**本身**对齐；对不上就不产生 REPLACE 候选，
    由 LIMIT（限定化）或 DELETE（显式标注证据不足）兜底。
    """
    from .llm import FIXTURE, FIXTURE_FLOOR, _jaccard
    claim_text = getattr(claim, 'text', claim)
    best = None
    for d in evidence:
        fx = FIXTURE.get(d['doc_id'])
        if not fx:
            continue
        ov = _jaccard(claim_text, fx['wrong'])
        if ov >= FIXTURE_FLOOR and (best is None or ov > best[0]):
            best = (ov, fx['corrected'], d)
    if best:
        return best[1], best[2]
    if getattr(llm, 'name', 'mock') != 'mock':
        ev = evidence[0] if evidence else None
        return (re.split(r'(?<=[。；])', ev['text'])[0].strip(), ev) if ev \
            else (None, None)
    return None, None


_KB_CACHE = {}


def _kb_doc(doc_id):
    """惰性加载知识库里的某一篇（供 _corrective 全局定位用）。"""
    if not _KB_CACHE:
        from .kb import load
        _KB_CACHE['kb'] = load()
    return _KB_CACHE['kb'].get(doc_id)


def candidates(claim, evidence, llm, enable_limit=True):
    """返回 [(operator, new_text_or_None), ...]。"""
    out = []

    # REPLACE：最小改写（须通过主体连续性检查，见 subject_overlap）
    corrected, src = _corrective(claim.text, evidence, llm)
    if corrected:
        try:
            new = llm.revise(claim.text, evidence, corrected)
            if new and new.strip():
                ov = subject_overlap(claim.text, new.strip())
                if ov >= REPLACE_SUBJECT_FLOOR:
                    out.append(('REPLACE', new.strip()))
                else:
                    out.append(('REPLACE_REJECTED',
                                '主体连续性 %.2f < %.2f，判定为"替换主题"而非'
                                '"最小改写"，拒绝该候选' % (ov, REPLACE_SUBJECT_FLOOR)))
        except Exception as e:                       # 后端失败 ⇒ 退回删除
            out.append(('REPLACE_ERROR', '%s' % e))

    # LIMIT：限定化（补适用范围）。真实后端下由 prompt 产出；
    # 离线夹具不具备"生成限定条件"的能力 ⇒ 该候选不可行，自动落到 REPLACE/DELETE。
    if enable_limit and getattr(llm, 'name', 'mock') != 'mock' and evidence:
        try:
            lim = llm.revise(claim.text, evidence, '限定化：保留原意，'
                             '补充适用范围与前提条件，使其与证据相容')
            if lim and lim.strip():
                out.append(('LIMIT', lim.strip()))
        except Exception:
            pass

    # DELETE：删除并显式标注（永远"可行"，但编辑距离最大 ⇒ 最后手段）
    out.append(('DELETE', None))
    keep = [x for x in out if x[0] not in ('REPLACE_ERROR', 'REPLACE_REJECTED')]
    rejects = [x for x in out if x[0] == 'REPLACE_REJECTED']
    return (keep or [('DELETE', None)]), rejects


# ── 投影 ────────────────────────────────────────────────────────────────
def project_claim(claim, kb, llm, topk=3, enable_limit=True, max_requery=1):
    """对单条声明做约束投影。就地更新 claim 并返回投影记录 dict。

    **投影只作用于 CONTRADICTED 声明**。这一条是刻意的边界：
      · CONTRADICTED 是错误（与知识库冲突），必须处置 —— 改写或移除；
      · UNVERIFIED 是"无法溯源"，不是错误。此时**没有**可依据的正确表述，
        任何"改写"都只能是凭空另写一句，反而引入新内容。
        正确处置是：保留原文 + 显式标注未溯源（记入严口径幻觉率），
        而不是拿知识库里某条不相干的正确表述去替换它。
    """
    record = {'cid': claim.cid, 'section': claim.section,
              '原始声明': claim.text, '原判定': claim.verdict,
              '尝试': [], '核定过程的拒绝': [],
              '选定算子': 'KEEP', '最终文本': claim.text}

    if claim.verdict != 'CONTRADICTED':
        record['不介入理由'] = (
            '判定为 %s，非冲突 ⇒ 不做改写，仅标注溯源状态' % claim.verdict)
        return record

    evidence = [kb.get(e[0]) for e in claim.evidence if kb.get(e[0])]

    # 冲突声明先做一次补充检索：若能据此转为可溯源，则无需改写
    if max_requery > 0:
        more = [d for _, d in kb.retrieve(claim.text, k=topk + 3)]
        seen = set(e[0] for e in claim.evidence)
        for d in more:
            if d['doc_id'] not in seen:
                evidence.append(d)
        c2, r2 = llm.judge(claim.text, evidence)
        record['补充检索后判定'] = c2
        if c2 != 'CONTRADICTED':
            claim.verdict = c2
            claim.reason = r2
            record['原判定'] = c2
            record['不介入理由'] = '补充检索后不再判为冲突 ⇒ 转为标注'
            return record

    cands, rejects = candidates(claim, evidence, llm, enable_limit=enable_limit)
    record['核定过程的拒绝'] = [{'算子': o, '拒绝理由': t} for o, t in rejects]

    best = None
    for op, new_text in cands:
        # ---- 可行性判定：候选必须通过**再次审核** ----
        if op == 'DELETE':
            feasible, verdict = True, 'DELETED'
        else:
            verdict, reason = llm.judge(new_text, evidence)
            feasible = (verdict != 'CONTRADICTED')
        dist = normalize_distance(claim.text, new_text)
        record['尝试'].append({'算子': op, '可行性': feasible, '复审判定': verdict,
                               '归一化编辑距离': round(dist, 4),
                               '候选文本': new_text})
        key = (dist, _OP_RANK.get(op, 9))
        if feasible and (best is None or key < best[0]):
            best = (key, op, new_text, verdict)

    if best is None:                                  # 理论上不会发生（DELETE 兜底）
        best = ((1.0, _OP_RANK['DELETE']), 'DELETE', None, 'DELETED')

    _key, op, new_text, verdict = best
    claim.operator = op
    claim.edit_distance = round(_key[0], 4)
    if op == 'DELETE':
        claim.final_text = '【证据不足，已移除该声明】原声明：%s' % claim.text
        claim.verdict = 'REMOVED'
    else:
        claim.final_text = new_text
        claim.verdict = 'SUPPORTED' if verdict == 'SUPPORTED' else verdict
    record['选定算子'] = op
    record['最终文本'] = claim.final_text
    return record


def project(claims, kb, llm, eps=0.05, enable_limit=True, max_requery=1):
    """对一组声明整体投影。返回 (records, stats, feasible)。

    eps          可行性阈值：残余未合规声明比例 > eps ⇒ 该资源判定**不可交付**
    feasible     False 时调用方应触发"补充检索后重生成"或返回"证据不足"
    """
    records = [project_claim(c, kb, llm, enable_limit=enable_limit,
                             max_requery=max_requery)
               for c in claims]
    n = len(claims)
    # 残余未合规 = 仍是 KEEP、但判定为 CONTRADICTED 的声明
    #            （= 投影未能处置的冲突，这是真正需要报警的情况）。
    # 注意两点，都是刻意的：
    #   · 被 DELETE 的声明已不在交付文本中，属于"已处置"，不计入残余；
    #   · 未溯源的声明（UNVERIFIED）保留在文本里但**不算未合规** ——
    #     它单独作为「未溯源保留数」报告，并计入严口径幻觉率。
    bad = [c for c in claims
           if c.operator == 'KEEP' and c.verdict == 'CONTRADICTED']
    unver = [c for c in claims
             if c.operator == 'KEEP' and c.verdict == 'UNVERIFIED']
    replaced = [c for c in claims if c.operator == 'REPLACE']
    limited = [c for c in claims if c.operator == 'LIMIT']
    deleted = [c for c in claims if c.operator == 'DELETE']
    touched = [c for c in claims if c.operator != 'KEEP']
    dists = [c.edit_distance for c in touched]
    stats = {
        '声明总数': n,
        '冲突声明数': sum(1 for r in records
                      if r.get('原判定') == 'CONTRADICTED'),
        '投影介入率': round(len(touched) / float(n), 4) if n else 0.0,
        '替换数': len(replaced), '限定化数': len(limited), '删除数': len(deleted),
        '平均编辑距离': round(sum(dists) / len(dists), 4) if dists else 0.0,
        '残余未合规数': len(bad),
        '残余未合规比例': round(len(bad) / float(n), 4) if n else 0.0,
        '未溯源保留数': len(unver),
        '可行性': (len(bad) / float(n)) <= eps if n else True,
    }
    return records, stats, stats['可行性']


def apply_to_text(text, claims, deleted_marker=True, unverified_marker=True):
    """把投影结果写回资源文本：替换/移除命中的句子，并附审核说明。

    保留原始小节结构。被移除与未被溯源的声明都在文末附录里**逐条列出**：
    读者能看见"这一条被移除了、原因是与知识库冲突"以及"这一条没有找到证据"，
    这比悄悄改写可审计得多，也正是榜题要求的"高保真知识溯源"。
    """
    out = text
    removed, unver, fixes = [], [], []
    for c in claims:
        if c.operator == 'KEEP':
            if unverified_marker and c.verdict in ('UNVERIFIED',):
                unver.append(c)
            continue
        if c.operator == 'DELETE':
            # 显式标注而非静默删除：不静默抹掉任何一条断言。
            out = out.replace(c.origin, '【证据不足，已移除该声明】')
            removed.append(c)
        else:
            out = out.replace(c.origin, c.final_text)
            fixes.append(c)
    out = re.sub(r'\n{3,}', '\n\n', out)
    if deleted_marker and (removed or unver or fixes):
        L = ['\n\n## 附：审核说明']
        L.append('本资源的每一条结论都经过声明级取证审核，处理结果如下：')
        L.append('')
        if fixes:
            L.append('- **修正 %d 条**（与知识库冲突，已做最小改写，'
                     '平均归一化编辑距离 %.4f）：'
                     % (len(fixes), sum(c.edit_distance for c in fixes) / len(fixes)))
            for c in fixes:
                L.append('  - `%s` %s → %s' % (c.cid, c.origin, c.final_text))
        if removed:
            L.append('- **移除 %d 条**（知识库无支持证据且无可行改写，'
                     '按"宁缺毋错"处理，保留可见标记）：' % len(removed))
            for c in removed:
                L.append('  - `%s` %s' % (c.cid, c.origin))
        if unver:
            L.append('- **保留但标注未溯源 %d 条**（知识库既无支持也无反驳，'
                     '不做改写 —— 改写会引入证据之外的新内容）：' % len(unver))
            for c in unver:
                L.append('  - `%s` %s' % (c.cid, c.origin))
        L.append('')
        L.append('> 完整记录（含每条声明的检索证据、判定理由、候选改写与'
                 '可行性复审结果）见 `audit_trace.json`。')
        out += '\n'.join(L) + '\n'
    return out


def stats_line(stats):
    return ('投影介入率 %.1f%%｜替换 %d、限定化 %d、删除 %d｜平均归一化编辑距离 %.4f'
            '｜可行性 %s' % (stats['投影介入率'] * 100, stats['替换数'],
                            stats['限定化数'], stats['删除数'],
                            stats['平均编辑距离'],
                            '通过' if stats['可行性'] else '不通过'))
