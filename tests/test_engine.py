# -*- coding: utf-8 -*-
"""Ph-MARL-Edu 单元测试。

直接运行：
    python -m unittest discover -s tests -v
或：
    python tests/test_engine.py

测试按被测模块分层组织，每个 test_* 方法针对一个**具体断言**，
方法名即断言内容。测试总量与覆盖范围见文末 TestCoverage。
"""
import json
import os
import sys
import unittest
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import claims as CLAIMS                       # noqa: E402
from engine import projection as PROJ                      # noqa: E402
from engine.agents import (AuditorAgent, DiagnoserAgent,    # noqa: E402
                           GeneratorAgent, SchedulerAgent)
from engine.gating import FullCommGate, Gate, impact_of     # noqa: E402
from engine.kb import KnowledgeBase, load, tokenize         # noqa: E402
from engine.llm import (ABS_WORDS, FIXTURE, FIXTURE_FLOOR,   # noqa: E402
                        MISCONCEPTIONS, MockLLM, _jaccard,
                        _stable_hash, get_backend)
from engine.orchestrator import LearnerSim, Orchestrator    # noqa: E402
from engine.report import build_report, render_html         # noqa: E402

PROF = {'profile_id': 'L001', 'education': '本科大三 人工智能专业',
        'theory_score': '62/100', 'hands_on': '有 Python，无 RL 经验',
        'history_accuracy': 0.62}


def profile(**kw):
    p = dict(PROF)
    p.update(kw)
    return p


def _claim(text, section='正文', cid='C001'):
    """造一条 Claim（用于只测投影算子、不经过真实链路的场景）。"""
    return CLAIMS.Claim(cid, text, section)


# ═══════════════════════════════════════════════════════════ 1 知识库
class TestKnowledgeBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = load()

    def test_kb_loads_at_least_30_entries(self):
        self.assertGreaterEqual(len(self.kb.docs), 30)

    def test_kb_has_5_modules(self):
        self.assertGreaterEqual(len(set(d['module'] for d in self.kb.docs)), 5)

    def test_kb_knowledge_point_tags_at_least_60(self):
        tags = set(k for d in self.kb.docs for k in d.get('knowledge_points', []))
        self.assertGreaterEqual(len(tags), 60)

    def test_kb_has_at_least_10_real_sources(self):
        src = set(d.get('source', '') for d in self.kb.docs)
        self.assertGreaterEqual(len(src), 10)

    def test_every_doc_has_required_fields(self):
        for d in self.kb.docs:
            for f in ('doc_id', 'title', 'module', 'difficulty', 'text',
                      'knowledge_points', 'source', 'section'):
                self.assertIn(f, d, '%s 缺字段 %s' % (d.get('doc_id'), f))

    def test_difficulty_labels_are_from_ladder(self):
        for d in self.kb.docs:
            self.assertIn(d['difficulty'], ('基础', '进阶', '挑战'))

    def test_retrieve_discount_factor_returns_its_own_doc(self):
        r = self.kb.retrieve('折扣因子应该取多大', k=1)
        self.assertEqual(r[0][1]['doc_id'], 'KB-M1-08')

    def test_retrieve_gated_communication_returns_its_own_doc(self):
        r = self.kb.retrieve('门控通信的稀疏度怎么算', k=1)
        self.assertEqual(r[0][1]['doc_id'], 'KB-M3-03')

    def test_retrieve_scores_are_descending(self):
        r = self.kb.retrieve('安全滤波器的介入率', k=5)
        self.assertEqual([s for s, _ in r], sorted([s for s, _ in r], reverse=True))

    def test_retrieve_respects_doc_ids_filter(self):
        r = self.kb.retrieve('折扣因子', k=3, doc_ids=['KB-M1-08', 'KB-M1-04'])
        self.assertTrue(all(d['doc_id'] in ('KB-M1-08', 'KB-M1-04') for _, d in r))

    def test_retrieve_respects_module_filter(self):
        r = self.kb.retrieve('通信', k=3, module='M3 多智能体协同')
        self.assertTrue(all(d['module'].startswith('M3') for _, d in r))

    def test_get_returns_doc_by_id(self):
        self.assertEqual(self.kb.get('KB-M1-08')['title'], '折扣因子的作用与取值')

    def test_get_returns_none_for_unknown_id(self):
        self.assertIsNone(self.kb.get('KB-NOPE-99'))

    def test_citation_contains_title_source_and_id(self):
        c = self.kb.citation(self.kb.get('KB-M1-08'))
        self.assertIn('折扣因子的作用与取值', c)
        self.assertIn('KB-M1-08', c)

    def test_stats_reports_entry_count(self):
        self.assertEqual(self.kb.stats()['条目数'], len(self.kb.docs))

    def test_tokenize_splits_ascii_and_cjk(self):
        t = tokenize('QMIX 的单调性约束')
        self.assertIn('qmix', t)
        self.assertIn('单调', t)

    def test_knowledge_base_accepts_custom_path(self):
        kb2 = KnowledgeBase(self.kb.path)
        self.assertEqual(len(kb2.docs), len(self.kb.docs))


# ═══════════════════════════════════════════════════════════ 2 LLM 后端
class TestMockBackend(unittest.TestCase):
    def setUp(self):
        self.kb = load()
        self.llm = get_backend('mock')

    def test_backend_name_is_mock(self):
        self.assertEqual(self.llm.name, 'mock')

    def test_stable_hash_is_process_independent(self):
        self.assertEqual(_stable_hash('abc'), _stable_hash('abc'))
        self.assertNotEqual(_stable_hash('abc'), _stable_hash('abd'))

    def test_fixture_has_10_misconceptions(self):
        self.assertEqual(len(MISCONCEPTIONS), 10)
        self.assertEqual(len(FIXTURE), 10)

    def test_every_fixture_doc_id_exists_in_kb(self):
        for doc_id in FIXTURE:
            self.assertIsNotNone(self.kb.get(doc_id), '%s 不在知识库中' % doc_id)

    def test_fixture_wrong_statements_all_judged_contradicted(self):
        """核心机制测试：10 条真实误区必须全部被判为冲突。"""
        for doc_id, m in FIXTURE.items():
            ev = [self.kb.get(doc_id)]
            v, _ = self.llm.judge(m['wrong'], ev)
            self.assertEqual(v, 'CONTRADICTED', '%s 未被判为冲突' % doc_id)

    def test_fixture_corrected_statements_none_judged_contradicted(self):
        """修正后的表述不得再被判为冲突（否则投影永远无解）。"""
        for doc_id, m in FIXTURE.items():
            ev = [self.kb.get(doc_id)]
            v, _ = self.llm.judge(m['corrected'], ev)
            self.assertNotEqual(v, 'CONTRADICTED', '%s 的修正表述仍被判冲突' % doc_id)

    def test_judge_flags_absolute_wording(self):
        ev = [self.kb.get('KB-M1-08')]
        v, why = self.llm.judge('γ 越大越好。', ev)
        self.assertEqual(v, 'CONTRADICTED')
        self.assertIn('越大越好', why)

    def test_fixture_alignment_separates_wrong_from_corrected(self):
        """FIXTURE_FLOOR=0.70 的标定依据：这个数是量出来的，不是拍的。

        条目级对账要求"与它自己那条误区"远高于"与别的表述"，
        否则短句会去认领**其他条目**的修正答案（见下面的回归测试）。
        """
        for a, ma in FIXTURE.items():
            self.assertEqual(_jaccard(ma['wrong'], ma['wrong']), 1.0)
            for b, mb in FIXTURE.items():
                if a == b:
                    continue
                self.assertLess(_jaccard(ma['wrong'], mb['wrong']), FIXTURE_FLOOR)
                self.assertLess(_jaccard(ma['corrected'], mb['wrong']),
                                FIXTURE_FLOOR)
            # 同一对里的"错误 vs 修正"也必须分得开（否则支持/冲突会互换）
            self.assertLess(_jaccard(ma['corrected'], ma['wrong']), FIXTURE_FLOOR)

    def test_short_claim_does_not_claim_another_items_correction(self):
        """回归：短句不得"认领"别的知识条目的修正表述。

        "γ 越大越好。" 只有 {γ,越大,越好} 三个词元。早期的方向性重叠
        （|A∩B|/|A|）让它与**消息压缩**那条误区拿到 0.83，于是判定理由写成
        "与知识库条目「消息压缩与信息瓶颈」的修正表述一致" → SUPPORTED，
        一条折扣因子的典型错误就这么被放过了。
        这条测试同时钉住两件事：相似度必须对称；判定不能跨条目乱认。
        """
        for other in FIXTURE:
            self.assertLess(_jaccard('γ 越大越好。', FIXTURE[other]['wrong']),
                            FIXTURE_FLOOR, other)
        # 走真实检索路径（top-3 里恰好有消息压缩那条）也必须判冲突
        ev = [d for _, d in self.kb.retrieve('γ 越大越好。', k=3)]
        self.assertIn('KB-M3-04', [d['doc_id'] for d in ev],
                      '前提：检索结果里确实混进了别的条目')
        v, why = self.llm.judge('γ 越大越好。', ev)
        self.assertEqual(v, 'CONTRADICTED', why)


    def test_judge_supports_verbatim_evidence(self):
        doc = self.kb.get('KB-M1-08')
        v, _ = self.llm.judge(doc['text'], [doc])
        self.assertEqual(v, 'SUPPORTED')

    def test_judge_returns_unverified_for_offtopic_text(self):
        v, _ = self.llm.judge('今天天气不错，适合出门散步。',
                              [self.kb.get('KB-M1-08')])
        self.assertEqual(v, 'UNVERIFIED')

    def test_judge_reports_a_reason_always(self):
        for t in ('γ 越大越好。', '无关的一句话在此。'):
            _, why = self.llm.judge(t, [self.kb.get('KB-M1-08')])
            self.assertTrue(why)

    def test_abs_words_list_is_nonempty_and_shared(self):
        self.assertGreaterEqual(len(ABS_WORDS), 10)

    def test_lecture_is_deterministic(self):
        ev = [d for _, d in self.kb.retrieve('折扣因子的作用与取值', k=2)]
        a = self.llm.generate_resource('lecture', '折扣因子的作用与取值', '进阶', ev, PROF)
        b = self.llm.generate_resource('lecture', '折扣因子的作用与取值', '进阶', ev, PROF)
        self.assertEqual(a, b)

    def test_three_kinds_produce_different_outputs(self):
        ev = [d for _, d in self.kb.retrieve('折扣因子的作用与取值', k=2)]
        outs = [self.llm.generate_resource(k, '折扣因子的作用与取值', '进阶', ev, PROF)
                for k in ('lecture', 'guide', 'quiz')]
        self.assertEqual(len(set(outs)), 3)

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            self.llm.generate_resource('podcast', 'x', '进阶', [], PROF)

    def test_injection_is_deterministic_across_instances(self):
        ev = [d for _, d in self.kb.retrieve('折扣因子的作用与取值', k=2)]
        a = MockLLM().generate_resource('lecture', '折扣因子的作用与取值', '进阶', ev, PROF)
        b = MockLLM().generate_resource('lecture', '折扣因子的作用与取值', '进阶', ev, PROF)
        self.assertEqual(a, b)

    def test_quiz_pulls_from_question_bank(self):
        ev = [d for _, d in self.kb.retrieve('折扣因子的作用与取值', k=2)]
        q = self.llm.generate_resource('quiz', '折扣因子的作用与取值', '进阶', ev, PROF)
        self.assertIn('题干', q)

    def test_get_backend_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            get_backend('nope')


# ═══════════════════════════════════════════════════════════ 3 声明切分
class TestClaimSplitting(unittest.TestCase):
    def setUp(self):
        self.kb = load()
        self.llm = get_backend('mock')

    def test_headings_are_not_claims(self):
        secs = [s for s, _ in CLAIMS.split_claims('# 一级标题\n\n正文内容足够长的一句话在这里。')]
        self.assertEqual(len(secs), 1)

    def test_citation_lines_are_not_claims(self):
        t = '## 一、证据\n- 折扣因子的作用与取值 §3.3\n正文断言足够长的一句话在这里。'
        texts = [x for _, x in CLAIMS.split_claims(t)]
        self.assertFalse(any('§3.3' in x for x in texts))

    def test_meta_lines_are_not_claims(self):
        t = '适用对象：L001\n本资源面向学习者 L001（本科大三），目标难度档：进阶。\n真正的断言在这里写着呢。'
        texts = [x for _, x in CLAIMS.split_claims(t)]
        self.assertEqual(len(texts), 1)

    def test_short_fragments_are_dropped(self):
        self.assertEqual(CLAIMS.split_claims('短。\n'), [])

    def test_long_sentence_is_split_on_period(self):
        t = '这是第一句足够长的断言内容。这是第二句足够长的断言内容。'
        self.assertEqual(len(CLAIMS.split_claims(t)), 2)

    def test_bold_markers_are_stripped(self):
        texts = [x for _, x in CLAIMS.split_claims('- **折扣因子**：取值需要在长期回报与方差之间权衡。')]
        self.assertNotIn('**', texts[0])

    def test_appendix_section_is_skipped(self):
        t = '## 附：审核说明\n本资源中有 2 条声明因证据不足被移除。'
        self.assertEqual(CLAIMS.split_claims(t), [])

    def test_process_text_is_classified_non_domain(self):
        t = '确认 Python 版本与依赖，固定随机种子以保证可复现。'
        ev = [d for _, d in self.kb.retrieve(t, k=3)]
        ok, _, why = CLAIMS.is_domain_claim(t, ev, self.kb)
        self.assertFalse(ok)
        self.assertIn('流程', why)

    def test_domain_assertion_is_accepted(self):
        t = '折扣因子 γ 的取值需要在重视长期回报与控制价值估计方差之间权衡。'
        ev = [d for _, d in self.kb.retrieve(t, k=3)]
        ok, _, _ = CLAIMS.is_domain_claim(t, ev, self.kb)
        self.assertTrue(ok)

    def test_question_is_not_a_claim(self):
        t = '损失函数里折扣因子 γ 是不是越大越好？'
        ev = [d for _, d in self.kb.retrieve(t, k=3)]
        ok, _, why = CLAIMS.is_domain_claim(t, ev, self.kb)
        self.assertFalse(ok)
        self.assertIn('疑问句', why)

    def test_absolute_wording_forces_audit_even_without_overlap(self):
        """兜底：绝对化表述必须进入审核，不能被"非断言"豁免吞掉。"""
        t = '任何情况下该系数都是越大越好。'
        ok, _, why = CLAIMS.is_domain_claim(t, [], self.kb)
        self.assertTrue(ok)
        self.assertIn('绝对化', why)

    def test_operational_rule_outranks_kp_label_fallback(self):
        """回归：纯流程指令不能因为"提了某个知识点名词"就被拖进幻觉率分母。

        早期版本的次序是 重叠 → 绝对化 → 知识点标签，而标签兜底在**全库**的知识点
        标签里做子串匹配，于是实操指南的三个步骤句（含"随机种子""VDN"）全部被判为
        UNVERIFIED 声明，一份指南的「非断言性文本数」是 0、严口径幻觉率被推高到 30%+。
        修法是把"以操作性动词起句"提到标签兜底之前（用户可见的差别在 eval 报告里）。
        """
        for t in ('确认 Python 版本与依赖，固定随机种子以保证可复现。',
                  '按本知识点的最小示例搭好训练回路，先只跑 1 个短程配置。',
                  '记录与「值分解方法（VDN / QMIX）」直接相关的读数。'):
            ev = [d for _, d in self.kb.retrieve(t, k=3)]
            ok, _, why = CLAIMS.is_domain_claim(t, ev, self.kb)
            self.assertFalse(ok, t)
            self.assertIn('操作性动词', why)

    def test_imperative_with_degree_marker_is_not_exempt(self):
        """反向边界：以操作动词起句、却顺带断言领域属性的句子不许被豁免。"""
        for t in ('先检查学习率调大一些效果更好。',
                  '固定随机种子后收敛曲线就完全一致了。'):
            ev = [d for _, d in self.kb.retrieve(t, k=3)]
            ok, _, why = CLAIMS.is_domain_claim(t, ev, self.kb)
            self.assertTrue(ok, t)
            self.assertIn('取消豁免', why)

    def test_short_absolutist_sentence_is_not_dropped(self):
        """回归：短句下限不能吞掉绝对化表述 —— 那是幻觉率被系统性低估的来源。

        "γ 越大越好。" 只有 7 个字符，早期版本在切分阶段就按长度把它丢掉，
        于是它既不进分子也不进分母；而它恰恰是投影算子最该处置的一类句子。
        """
        got = CLAIMS.split_claims('γ 越大越好。')
        self.assertEqual([t for _, t in got], ['γ 越大越好。'])
        cl, nd = CLAIMS.audit('γ 越大越好。', self.kb, self.llm)
        self.assertEqual(len(cl), 1, '短句里的绝对化表述必须进入审核')
        self.assertEqual(cl[0].verdict, 'CONTRADICTED')

    def test_short_non_absolutist_fragment_is_still_dropped(self):
        """但普通短碎片（标题、字段名）仍不参与审核，豁免没有扩大到全部短句。"""
        self.assertEqual(CLAIMS.split_claims('见上文。'), [])

    def test_citation_field_line_is_not_a_claim(self):
        """出处字段不是断言 —— 它指向元数据，不指向领域事实。

        回归依据：分阶测试题的解析行都长这样，实测 21 条被当成"无法溯源的断言"
        计入严口径幻觉率。那衡量的是"我们没给自己的引用行找到证据"，
        而不是"资源里有不可溯源的知识"，指标含义被带偏。
        """
        for t in ('解析：出处：《值分解方法（VDN / QMIX）》§4（KB-M1-05）。',
                  '溯源：KB-M1-05'):
            self.assertEqual(CLAIMS.split_claims(t), [], t)
        # 必须按句判：生成器把解析与出处写在同一行，按行判会整行漏过
        raw = ('**解析**：出处：《值分解方法（VDN / QMIX）》§4（KB-M1-05）。'
               '答案逐字取自知识库原文，可直接核对。')
        self.assertEqual([t for _, t in CLAIMS.split_claims(raw)],
                         ['答案逐字取自知识库原文，可直接核对。'])
        # 但在引用之上**作出断言**的句子照审不误
        t = '证据 KB-M1-05 指出 QMIX 使用单调性约束（KB-M1-05）。'
        self.assertEqual(len(CLAIMS.split_claims(t)), 1)

    def test_domain_floor_calibration(self):
        """DOMAIN_OVERLAP_FLOOR 的标定依据 —— 这个数是量出来的，不是拍的。

        实测（本系统生成器实际产出的句子，证据 = 该句自身的 top-3 检索结果）：
            流程性表述   0.19 ~ 0.39
            领域事实断言 0.54 ~ 0.65
        0.50 落在空档里，两侧各留 ≥0.11 余量。若将来改检索或分词，
        这个测试会先亮，提醒重新标定，而不是让阈值悄悄失效。
        """
        floor = CLAIMS.DOMAIN_OVERLAP_FLOOR
        proc = ['本讲义的每条结论均可在文末证据出处的原文中核对。',
                '打开日志文件查看损失曲线。']
        dom = ['折扣因子 γ 应当尽可能取接近 1 的值。',
               '超参直接照搬其他环境而不做标定，是训练失败的最常见根因。']
        pmax = max(self._best_overlap(t) for t in proc)
        dmin = min(self._best_overlap(t) for t in dom)
        self.assertLess(pmax, floor, '流程句重叠 %.2f 冲到了阈值以上' % pmax)
        self.assertLessEqual(floor, dmin, '领域断言重叠 %.2f 掉到阈值以下' % dmin)

    def _best_overlap(self, t):
        ev = [d for _, d in self.kb.retrieve(t, k=3)]
        return CLAIMS.is_domain_claim(t, ev, self.kb)[1]

    def test_audit_returns_claims_and_non_domain(self):
        ev = [d for _, d in self.kb.retrieve('折扣因子', k=2)]
        txt = self.llm.generate_resource('guide', '折扣因子的作用与取值', '进阶', ev, PROF)
        cl, nd = CLAIMS.audit(txt, self.kb, self.llm)
        self.assertIsInstance(cl, list)
        self.assertIsInstance(nd, list)

    def test_summarize_returns_full_keys_for_empty_claim_list(self):
        s = CLAIMS.summarize([], [{'text': 'x'}])
        for k in ('声明总数', 'SUPPORTED', 'CONTRADICTED', 'UNVERIFIED',
                  '幻觉率_宽口径(冲突/总数)', '幻觉率_严口径((冲突+未溯源)/总数)',
                  '非断言性文本_不计入'):
            self.assertIn(k, s)

    def test_summarize_wide_and_strict_rates_differ(self):
        ev = [self.kb.get('KB-M1-08')]
        cl, nd = CLAIMS.audit('γ 越大越好。\n完全无关的另一句话在这里写着。',
                              self.kb, self.llm)
        s = CLAIMS.summarize(cl, nd)
        self.assertGreaterEqual(s['幻觉率_严口径((冲突+未溯源)/总数)'],
                                s['幻觉率_宽口径(冲突/总数)'])

    def test_summarize_delivered_excludes_deleted(self):
        cl, nd = CLAIMS.audit('γ 越大越好。', self.kb, self.llm)
        cl[0].operator = 'DELETE'
        s = CLAIMS.summarize_delivered(cl)
        self.assertEqual(s['交付声明数'], 0)
        self.assertEqual(s['投影处置数'], 1)


# ═══════════════════════════════════════════════════════════ 4 投影
class TestProjection(unittest.TestCase):
    def setUp(self):
        self.kb = load()
        self.llm = get_backend('mock')

    def _lecture(self, kp='折扣因子的作用与取值'):
        """取一段**真实生成**的讲义文本及其声明级审核结果。

        投影的输入必须走真实链路（生成→切分→取证），
        不能拿一句手搓的句子去测：手搓句子的证据是 BM25 临时检索的，
        与原场景（证据来自被生成的那个知识点）不同，测出来的问题不是真问题。
        """
        ev = [d for _, d in self.kb.retrieve(kp, k=2)]
        txt = self.llm.generate_resource('lecture', kp, '进阶', ev, PROF)
        return txt, CLAIMS.audit(txt, self.kb, self.llm)

    def test_levenshtein_identical_is_zero(self):
        self.assertEqual(PROJ.levenshtein('abc', 'abc'), 0)

    def test_levenshtein_empty_string(self):
        self.assertEqual(PROJ.levenshtein('', 'abc'), 3)
        self.assertEqual(PROJ.levenshtein('abc', ''), 3)

    def test_levenshtein_known_value(self):
        self.assertEqual(PROJ.levenshtein('kitten', 'sitting'), 3)

    def test_normalize_distance_identical_is_zero(self):
        self.assertEqual(PROJ.normalize_distance('abc', 'abc'), 0.0)

    def test_normalize_distance_delete_is_one(self):
        self.assertEqual(PROJ.normalize_distance('abc', None), 1.0)

    def test_normalize_distance_stays_in_unit_interval(self):
        pairs = [('短', '很长很长的一句话在这里'), ('abc', 'xyz'), ('a', 'a' * 50)]
        for a, b in pairs:
            d = PROJ.normalize_distance(a, b)
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, 1.0)

    def test_rewrite_never_costs_more_than_delete(self):
        """关键性质：改写为更长的正确表述，代价不得超过整条删除。"""
        orig = 'γ 越大越好。'
        new = ('折扣因子 γ 的取值需要在重视长期回报与控制价值估计方差'
               '与自举偏差之间权衡，工程实践常见取值为 0.95～0.99。')
        self.assertLess(PROJ.normalize_distance(orig, new), 1.0)

    def test_subject_overlap_identical_is_one(self):
        self.assertAlmostEqual(PROJ.subject_overlap('折扣因子取值', '折扣因子取值'), 1.0)

    def test_subject_overlap_unrelated_is_low(self):
        self.assertLess(PROJ.subject_overlap('确认 Python 版本与依赖',
                                             '折扣因子取值需要权衡'), 0.25)

    def test_corrective_only_uses_own_evidence(self):
        """修正表述只能来自该声明自己的证据，不允许全局乱匹配。"""
        ev = [self.kb.get('KB-M5-03')]          # 与折扣因子无关
        c, src = PROJ._corrective('确认 Python 版本与依赖，固定随机种子。',
                                  ev, self.llm)
        self.assertIsNone(c)

    def test_corrective_returns_fixture_correction_for_its_own_misconception(self):
        """声明与某条已知误区对齐时，才允许给出那条误区的修正表述。"""
        ev = [self.kb.get('KB-M1-08')]
        c, src = PROJ._corrective(FIXTURE['KB-M1-08']['wrong'], ev, self.llm)
        self.assertEqual(c, FIXTURE['KB-M1-08']['corrected'])

    def test_corrective_refuses_when_claim_is_not_that_misconception(self):
        """回归测试（单元测试抓到的真实缺陷）：一条与「消息压缩」毫不相干的声明，
        仅因为检索到的证据里含该条目（其原文恰有"越大越好"），就被改写成
        "消息维度不是越大越好…" —— 这不是改错，是换了题目。"""
        ev = [self.kb.get('KB-M3-04')]
        c, _ = PROJ._corrective('折扣因子 γ 越大越好。', ev, self.llm)
        self.assertIsNone(c)

    def test_candidate_correction_stays_on_the_same_subject(self):
        """端到端复核上一条：素材是真实生成文本时，修正后主体不得漂移。"""
        txt, (cl, _) = self._lecture()
        for c in cl:
            if c.verdict != 'CONTRADICTED':
                continue
            PROJ.project_claim(c, self.kb, self.llm)
            if c.operator == 'REPLACE':
                self.assertGreaterEqual(
                    PROJ.subject_overlap(c.origin, c.final_text),
                    PROJ.REPLACE_SUBJECT_FLOOR)

    def test_candidates_always_include_delete(self):
        ev = [self.kb.get('KB-M1-08')]
        cands, _ = PROJ.candidates(
            _claim(FIXTURE['KB-M1-08']['wrong']), ev, self.llm)
        self.assertIn('DELETE', [o for o, _ in cands])

    def test_candidates_reject_offtopic_replacement(self):
        """主体连续性检查：离题替换必须被拒绝，而不是被采用。"""
        ev = [self.kb.get('KB-M1-08')]
        cands, _ = PROJ.candidates(
            _claim('确认 Python 版本与依赖，固定随机种子。'), ev, self.llm)
        self.assertFalse(any(o == 'REPLACE' for o, _ in cands))

    def test_project_claim_keeps_supported_claim(self):
        _, (cl, _) = self._lecture()
        ok = [c for c in cl if c.verdict == 'SUPPORTED']
        self.assertTrue(ok, '生成文本里应当有可溯源声明')
        rec = PROJ.project_claim(ok[0], self.kb, self.llm)
        self.assertEqual(rec['选定算子'], 'KEEP')

    def test_project_claim_does_not_rewrite_unverified(self):
        """未溯源声明不做改写 —— 改写会引入证据之外的新内容。

        这里直接造 Claim 而不走 audit：修好"操作性文本被误判为断言"之后，
        真实链路的产物里已经几乎不再有 UNVERIFIED 声明
        （原先那些其实是实操指南的流程句，见 claims.is_domain_claim 的次序说明），
        而这条边界本身仍然必须被钉住 —— 它是"没有证据时宁可不改"的守门测试。
        """
        c = _claim('本团队在去年的实验结果中观察到这一现象，具体成因尚待确认。')
        c.verdict = 'UNVERIFIED'
        rec = PROJ.project_claim(c, self.kb, self.llm)
        self.assertEqual(rec['选定算子'], 'KEEP')
        self.assertEqual(c.final_text, c.origin, '未溯源声明不得被改写')
        self.assertIn('非冲突', rec['不介入理由'])

    def test_project_claim_fixes_injected_misconception(self):
        """核心机制测试：注入的误区必须被投影处置掉（改写或明确移除）。"""
        txt, (cl, _) = self._lecture()
        bad = [c for c in cl if c.verdict == 'CONTRADICTED']
        self.assertTrue(bad, '夹具注入的误区必须在审核中被判为冲突')
        rec = PROJ.project_claim(bad[0], self.kb, self.llm)
        self.assertIn(rec['选定算子'], ('REPLACE', 'DELETE'))
        self.assertNotEqual(bad[0].final_text, bad[0].origin)
        v, _ = self.llm.judge(bad[0].final_text,
                              [self.kb.get(e[0]) for e in bad[0].evidence
                               if self.kb.get(e[0])] or
                              [self.kb.get('KB-M1-08')])
        self.assertNotEqual(v, 'CONTRADICTED')

    def test_project_reports_feasible_when_all_conflicts_handled(self):
        _, (cl, _) = self._lecture()
        _, stats, feasible = PROJ.project(cl, self.kb, self.llm)
        self.assertTrue(feasible)
        self.assertEqual(stats['残余未合规数'], 0)

    def test_project_stats_include_all_declared_keys(self):
        _, (cl, _) = self._lecture()
        _, stats, _ = PROJ.project(cl, self.kb, self.llm)
        for k in ('声明总数', '冲突声明数', '投影介入率', '替换数', '限定化数',
                  '删除数', '平均编辑距离', '残余未合规数', '未溯源保留数', '可行性'):
            self.assertIn(k, stats)

    def test_apply_to_text_removes_the_injected_misconception(self):
        txt, (cl, _) = self._lecture()
        bad = [c for c in cl if c.verdict == 'CONTRADICTED']
        PROJ.project(cl, self.kb, self.llm)
        out = PROJ.apply_to_text(txt, cl)
        body, _, note = out.partition('## 附：审核说明')
        for c in bad:
            self.assertNotIn(c.origin, body, '被处置的声明不应原样留在正文里')
            # 但必须**逐条**出现在审核说明里：附录按设计会引用原文
            # （"- `C004` 原文 → 修正文本"），读者要能看见"改了哪一条、为什么"。
            # 因此断言必须限定在正文段，不能对整份输出断言"不含原文" ——
            # 早期版本正是这么写的，于是把"可审计"这个特性误判成了缺陷。
            self.assertIn(c.cid, note, '被处置的声明必须逐条出现在审核说明里')

    def test_apply_to_text_appends_audit_note(self):
        txt, (cl, _) = self._lecture()
        PROJ.project(cl, self.kb, self.llm)
        self.assertIn('审核说明', PROJ.apply_to_text(txt, cl))

    def test_apply_to_text_lists_every_removed_and_unverified_claim(self):
        """没有任何声明被静默丢弃：移除与未溯源都必须逐条列进附录。"""
        txt, (cl, _) = self._lecture()
        PROJ.project(cl, self.kb, self.llm)
        out = PROJ.apply_to_text(txt, cl)
        for c in cl:
            if c.operator == 'DELETE' or c.verdict == 'UNVERIFIED':
                self.assertIn(c.cid, out, '%s 未出现在审核说明里' % c.cid)


# ═══════════════════════════════════════════════════════════ 5 门控通信
class TestGating(unittest.TestCase):
    def test_full_comm_gate_always_sends(self):
        g = FullCommGate()
        m = g.consider('ack', 'a', 'b', {'x': 1})
        self.assertTrue(m.sent)

    def test_gate_suppresses_low_impact_message(self):
        g = Gate(threshold=0.5)
        m = g.consider('diagnosis', 'a', 'b', {}, ctx={'n_targets': 4,
                                                       'n_level_changing': 0})
        self.assertFalse(m.sent)

    def test_gate_sends_high_impact_message(self):
        g = Gate(threshold=0.5)
        m = g.consider('diagnosis', 'a', 'b', {}, ctx={'n_targets': 2,
                                                       'n_level_changing': 2})
        self.assertTrue(m.sent)

    def test_impact_is_monotonic_in_level_changing(self):
        a = impact_of('diagnosis', {}, {'n_targets': 4, 'n_level_changing': 1})
        b = impact_of('diagnosis', {}, {'n_targets': 4, 'n_level_changing': 3})
        self.assertLess(a, b)

    def test_impact_is_capped_at_one(self):
        self.assertLessEqual(impact_of('dispute', {},
                                       {'n_targets': 1, 'n_disputes': 99}), 1.0)

    def test_zero_disputes_gives_zero_impact(self):
        self.assertEqual(impact_of('dispute', {}, {'n_targets': 3, 'n_disputes': 0}), 0.0)

    def test_impact_normalized_by_targets_not_blind_total(self):
        """回归测试：早期版本按盲区总数归一，导致盲区越多影响分越小、门控全关。"""
        small = impact_of('diagnosis', {}, {'n_targets': 2, 'n_level_changing': 2})
        self.assertEqual(small, 1.0)

    def test_stats_counts_candidates_and_sends(self):
        g = Gate(threshold=0.5)
        g.consider('ack', 'a', 'b', {})
        g.consider('diagnosis', 'a', 'b', {}, ctx={'n_targets': 1,
                                                   'n_level_changing': 1})
        s = g.stats()
        self.assertEqual(s['候选消息数'], 2)
        self.assertEqual(s['实际发送数'], 1)

    def test_sparsity_between_zero_and_one(self):
        g = Gate(threshold=0.5)
        for _ in range(3):
            g.consider('ack', 'a', 'b', {})
        s = g.stats()
        self.assertGreaterEqual(s['通信稀疏度(发送/候选)'], 0.0)
        self.assertLessEqual(s['通信稀疏度(发送/候选)'], 1.0)

    def test_reset_clears_counters(self):
        g = Gate()
        g.consider('ack', 'a', 'b', {})
        g.reset()
        self.assertEqual(g.stats()['候选消息数'], 0)

    def test_message_records_reason(self):
        g = Gate(threshold=0.5)
        m = g.consider('ack', 'a', 'b', {})
        self.assertIn('影响分', m.reason)

    def test_message_to_dict_is_json_serializable(self):
        g = Gate()
        m = g.consider('diagnosis', 'a', 'b', {'k': '值'})
        json.dumps(m.to_dict(), ensure_ascii=False)


# ═══════════════════════════════════════════════════════════ 6 四个 Agent
class TestAgents(unittest.TestCase):
    def setUp(self):
        self.kb = load()
        self.llm = get_backend('mock')

    # ---- A1 学情诊断 ----
    def test_diagnosis_includes_full_mastery_table(self):
        d = DiagnoserAgent(self.kb, self.llm).diagnose(PROF)
        self.assertEqual(len(d['mastery']), len(self.kb.docs))

    def test_mastery_is_in_unit_interval(self):
        d = DiagnoserAgent(self.kb, self.llm).diagnose(PROF)
        for v in d['mastery'].values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_higher_theory_score_raises_mastery(self):
        a = DiagnoserAgent(self.kb, self.llm).diagnose(profile(theory_score='40/100'))
        b = DiagnoserAgent(self.kb, self.llm).diagnose(profile(theory_score='90/100'))
        kp = list(a['mastery'])[0]
        self.assertLess(a['mastery'][kp], b['mastery'][kp])

    def test_blind_spots_all_below_threshold(self):
        d = DiagnoserAgent(self.kb, self.llm).diagnose(PROF)
        self.assertTrue(all(b['mastery'] < 0.60 for b in d['blind_spots']))

    def test_blind_spots_sorted_ascending(self):
        d = DiagnoserAgent(self.kb, self.llm).diagnose(PROF)
        ms = [b['mastery'] for b in d['blind_spots']]
        self.assertEqual(ms, sorted(ms))

    def test_strong_profile_has_no_blind_spots(self):
        d = DiagnoserAgent(self.kb, self.llm).diagnose(
            profile(theory_score='95/100', hands_on='做过多智能体项目',
                    history_accuracy=0.95))
        self.assertEqual(len(d['blind_spots']), 0)

    def test_representation_z_has_4_dimensions(self):
        d = DiagnoserAgent(self.kb, self.llm).diagnose(PROF)
        self.assertEqual(len(d['representation_z']), 4)

    def test_missing_history_is_marked_unknown(self):
        d = DiagnoserAgent(self.kb, self.llm).diagnose(
            {'profile_id': 'X', 'theory_score': 50, 'hands_on': '初级'})
        self.assertIsNone(d['history_accuracy'])

    def test_theory_score_accepts_plain_number(self):
        d = DiagnoserAgent(self.kb, self.llm).diagnose(
            profile(theory_score=62))
        self.assertGreater(d['skill_level'], 0.0)

    # ---- A2 生成 ----
    def test_generate_produces_three_resource_kinds(self):
        g = GeneratorAgent(self.kb, self.llm).generate('折扣因子的作用与取值', '进阶', PROF)
        self.assertEqual(len(g['resources']), 3)
        self.assertEqual(set(r['kind'] for r in g['resources']),
                         {'lecture', 'guide', 'quiz'})

    def test_generated_resources_carry_evidence(self):
        g = GeneratorAgent(self.kb, self.llm).generate('折扣因子的作用与取值', '进阶', PROF)
        for r in g['resources']:
            self.assertTrue(r['evidence'])
            self.assertIn('doc_id', r['evidence'][0])

    def test_top_retrieved_doc_for_discount_factor_is_correct(self):
        g = GeneratorAgent(self.kb, self.llm).generate('折扣因子的作用与取值', '进阶', PROF)
        self.assertEqual(g['resources'][0]['evidence'][0]['doc_id'], 'KB-M1-08')

    # ---- A3 审核 ----
    def test_audit_returns_three_parts(self):
        a = AuditorAgent(self.kb, self.llm)
        out = a.audit('折扣因子 γ 的取值需要权衡长期回报与方差。')
        self.assertEqual(len(out), 3)

    def test_two_views_agree_on_clear_domain_claim(self):
        a = AuditorAgent(self.kb, self.llm)
        cl, disputes, _ = a.audit('折扣因子 γ 的取值需要在重视长期回报与'
                                  '控制价值估计方差之间权衡，常见取值 0.95～0.99。')
        self.assertEqual(len(disputes), 0)

    def test_second_view_locks_to_label_module(self):
        """视角 B 的取证范围必须被锁在知识点所在模块内（与 A 不同源）。"""
        a = AuditorAgent(self.kb, self.llm)
        v, why, ev = a._view_b('门控通信的稀疏度定义为实际发送数除以候选消息数。')
        self.assertTrue(ev)
        self.assertTrue(all(d['module'].startswith('M3') for d in ev))

    def test_dispute_record_contains_both_views_and_reasons(self):
        a = AuditorAgent(self.kb, self.llm)
        for doc_id, m in FIXTURE.items():
            cl, disputes, _ = a.audit(m['wrong'])
            for d in disputes:
                self.assertIn('视角A', d)
                self.assertIn('视角B', d)
                self.assertIn('裁决', d)
                self.assertTrue(d['裁决理由'])

    def test_review_produces_delivered_text_and_stats(self):
        a = AuditorAgent(self.kb, self.llm)
        ev = [d for _, d in self.kb.retrieve('折扣因子的作用与取值', k=2)]
        txt = self.llm.generate_resource('lecture', '折扣因子的作用与取值', '进阶', ev, PROF)
        r = a.review(txt)
        self.assertIn('text_delivered', r)
        self.assertIn('pre_audit', r)
        self.assertIn('post_audit', r)

    def test_review_pre_audit_is_pre_projection_snapshot(self):
        """回归测试：投影会就地改写 verdict，必须先快照投影前的统计。"""
        a = AuditorAgent(self.kb, self.llm)
        ev = [d for _, d in self.kb.retrieve('折扣因子的作用与取值', k=2)]
        txt = self.llm.generate_resource('lecture', '折扣因子的作用与取值', '进阶', ev, PROF)
        r = a.review(txt)
        self.assertGreaterEqual(r['pre_audit']['声明总数'],
                                r['post_audit']['交付声明数'])

    # ---- A4 调度 ----
    def test_pick_difficulty_targets_band_center(self):
        """选的档位必须是**三个档里预测正确率最靠近心流区间中心 0.70 的那个**。

        断言的是这条准则本身，不是某个掌握度对应哪个档 ——
        后者的期望值会随资源支撑系数（DIFF_FACTOR）一起变，
        写死档位名会让"标定档位"变成"改测试"。实测对照：
        掌握度 0.55 在 1.45/1.10/0.80 三档下预测为 0.798/0.605/0.440，
        最靠近 0.70 的是「进阶」（|0.605−0.70| = 0.095 < |0.798−0.70| = 0.098）。
        """
        for m in (0.05, 0.2, 0.4, 0.55, 0.7, 0.85, 0.95, 1.0):
            d, rate = SchedulerAgent.pick_difficulty(m)
            others = [abs(m * SchedulerAgent.FACTOR[x] - 0.70)
                      for x in SchedulerAgent.FACTOR if x != d]
            self.assertLessEqual(abs(rate - 0.70), min(others) + 1e-12,
                                 '掌握度 %.2f 选了 %s，但它不是最靠近 0.70 的档' % (m, d))

    def test_pick_difficulty_chooses_challenge_for_high_mastery(self):
        self.assertEqual(SchedulerAgent.pick_difficulty(0.95)[0], '挑战')

    def test_pick_difficulty_chooses_basic_for_low_mastery(self):
        self.assertEqual(SchedulerAgent.pick_difficulty(0.05)[0], '基础')

    def test_shift_clamps_at_bottom(self):
        self.assertEqual(SchedulerAgent.shift('基础', -1), '基础')

    def test_shift_clamps_at_top(self):
        self.assertEqual(SchedulerAgent.shift('挑战', +1), '挑战')

    def test_shift_moves_one_step(self):
        self.assertEqual(SchedulerAgent.shift('基础', +1), '进阶')

    def test_next_action_asks_for_scaffolding_on_low_rate(self):
        self.assertEqual(SchedulerAgent.next_action(0.4, 5), '降维解释')

    def test_next_action_keeps_level_inside_band(self):
        self.assertEqual(SchedulerAgent.next_action(0.7, 5), '维持当前难度')

    def test_next_action_escalates_above_band(self):
        self.assertEqual(SchedulerAgent.next_action(0.95, 5), '进阶挑战')

    def test_next_action_adds_prerequisite_on_first_failure(self):
        self.assertEqual(SchedulerAgent.next_action(0.2, 1), '补充前置知识')

    def test_plan_round_defers_unreachable_knowledge_points(self):
        d = DiagnoserAgent(self.kb, self.llm).diagnose(
            profile(theory_score='20/100', hands_on='无', history_accuracy=0.1))
        a4 = SchedulerAgent(Gate(threshold=0.3), budget_kp=3, kb=self.kb)
        plan = a4.plan_round(d)
        self.assertTrue(plan['暂缓数'] > 0)
        self.assertTrue(all('暂缓理由' in x for x in plan['超出可及范围_暂缓']))

    def test_plan_round_always_returns_targets_for_strong_learner(self):
        """回归测试：高水平学习者没有盲区时，系统必须转为能力拓展而不是无事可做。"""
        d = DiagnoserAgent(self.kb, self.llm).diagnose(
            profile(theory_score='95/100', hands_on='做过多智能体项目',
                    history_accuracy=0.95))
        a4 = SchedulerAgent(Gate(threshold=0.3), budget_kp=2, kb=self.kb)
        plan = a4.plan_round(d)
        self.assertTrue(plan['targets'])
        self.assertIn('能力拓展', plan['targets'][0]['目标类型'])

    def test_plan_round_records_gate_decision(self):
        d = DiagnoserAgent(self.kb, self.llm).diagnose(PROF)
        a4 = SchedulerAgent(Gate(threshold=0.3), budget_kp=2, kb=self.kb)
        plan = a4.plan_round(d)
        self.assertIn('sent', plan['gate_message'])

    def test_feedback_shift_down_after_low_score(self):
        d = DiagnoserAgent(self.kb, self.llm).diagnose(PROF)
        a4 = SchedulerAgent(Gate(threshold=0.3), budget_kp=1, kb=self.kb)
        p1 = a4.plan_round(d)
        kp = p1['targets'][0]['knowledge_point']
        p2 = a4.plan_round(d, round_idx=2, prev_feedback={
            'per_kp': [{'knowledge_point': kp, 'difficulty': '挑战',
                        'correct': 0, 'total': 5}]})
        row = [t for t in p2['targets'] if t['knowledge_point'] == kp]
        if row:
            self.assertIn('降维解释', row[0]['difficulty_依据'])

    def test_feedback_shift_up_after_high_score(self):
        d = DiagnoserAgent(self.kb, self.llm).diagnose(PROF)
        a4 = SchedulerAgent(Gate(threshold=0.3), budget_kp=1, kb=self.kb)
        p1 = a4.plan_round(d)
        kp = p1['targets'][0]['knowledge_point']
        base = p1['targets'][0]['difficulty']
        p2 = a4.plan_round(d, round_idx=2, prev_feedback={
            'per_kp': [{'knowledge_point': kp, 'difficulty': base,
                        'correct': 5, 'total': 5}]})
        row = [t for t in p2['targets'] if t['knowledge_point'] == kp]
        if row and base != '挑战':
            self.assertEqual(SchedulerAgent.shift(base, +1), row[0]['difficulty'])


# ═══════════════════════════════════════════════════════════ 7 闭环编排
class TestOrchestrator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orc = Orchestrator(backend='mock', rounds=2, budget_kp=2, verbose=False)
        cls.sess = cls.orc.run(PROF, write=False)

    def test_session_has_two_rounds(self):
        self.assertEqual(len(self.sess['rounds']), 2)

    def test_trace_contains_all_five_closed_loop_steps(self):
        steps = [r['step'] for r in self.sess['trace']]
        joined = ''.join(steps)
        for mark in ('①', '②', '③④', '⑤'):
            self.assertIn(mark, joined)

    def test_trace_records_gate_decision_per_round(self):
        rounds = [r for r in self.sess['trace'] if r['step'].startswith('②')]
        self.assertEqual(len(rounds), 2)
        for r in rounds:
            self.assertIn('gate_message', r['data'])

    def test_every_resource_has_raw_and_delivered_text(self):
        for rd in self.sess['rounds']:
            for it in rd['items']:
                for res in it['resources']:
                    self.assertTrue(res['text_raw'])
                    self.assertTrue(res['text_delivered'])

    def test_every_resource_records_evidence(self):
        for rd in self.sess['rounds']:
            for it in rd['items']:
                for res in it['resources']:
                    self.assertTrue(res['evidence'])

    def test_feedback_written_back_to_mastery(self):
        """反馈必须真的回写掌握度，否则"闭环"是假的。"""
        d1 = [r for r in self.sess['trace'] if r['step'].startswith('①')][0]
        m0 = d1['data']['mastery']
        it0 = self.sess['rounds'][0]['items'][0]
        kp = it0['knowledge_point']
        plan2 = self.sess['rounds'][1]['plan']
        tgt = [t for t in plan2['targets'] if t['knowledge_point'] == kp]
        if tgt and it0['feedback']['total']:
            self.assertNotEqual(tgt[0]['mastery'], m0.get(kp))

    def test_delivered_hallucination_rate_not_above_raw(self):
        s = self.sess['summary']
        self.assertLessEqual(s['投影后_幻觉率_宽口径'], s['投影前_幻觉率_宽口径'])

    def test_summary_has_all_reported_metrics(self):
        s = self.sess['summary']
        for k in ('轮数', '涉及知识点数', '资源形态数', '声明总数', '交付声明数',
                  '投影前_幻觉率_宽口径', '投影后_幻觉率_宽口径',
                  '投影介入率', '平均归一化编辑距离', '交叉验证争议数',
                  '适配率(心流区间[0.6,0.8])', '适配准确率(决策口径)',
                  '全库知识点覆盖率', '门控',
                  '非断言性文本数(不计入幻觉率)'):
            self.assertIn(k, s)

    def test_three_resource_forms_present_in_session(self):
        self.assertEqual(self.sess['summary']['资源形态数'], 3)

    def test_all_profiles_run_without_error(self):
        from run import list_profiles, load_profile
        for pid in list_profiles():
            s = Orchestrator(backend='mock', rounds=2, budget_kp=2,
                             verbose=False).run(load_profile(pid), write=False)
            self.assertGreater(s['summary']['声明总数'], 0, pid)

    def test_gate_full_comm_sends_more_than_gated(self):
        s_full = Orchestrator(backend='mock', rounds=2, budget_kp=2, verbose=False,
                              gate=FullCommGate()).run(PROF, write=False)
        self.assertGreaterEqual(s_full['gate_stats']['实际发送数'],
                                self.sess['gate_stats']['实际发送数'])

    def test_correction_none_leaves_conflicts_in_place(self):
        s = Orchestrator(backend='mock', rounds=2, budget_kp=2, verbose=False,
                         correction='none').run(PROF, write=False)
        self.assertGreater(s['summary']['投影后_幻觉率_宽口径'],
                           self.sess['summary']['投影后_幻觉率_宽口径'])

    def test_correction_regenerate_cannot_beat_deterministic_mock(self):
        """确定性后端下重生成必然无效 —— 这条断言把该结论钉死在测试里。"""
        s = Orchestrator(backend='mock', rounds=2, budget_kp=2, verbose=False,
                         correction='regenerate').run(PROF, write=False)
        self.assertGreater(s['summary']['投影后_幻觉率_宽口径'], 0.0)

    def test_diagnosis_off_uses_uniform_starting_point(self):
        s = Orchestrator(backend='mock', rounds=2, budget_kp=2, verbose=False,
                         enable_diagnosis=False).run(PROF, write=False)
        self.assertLess(s['summary']['适配率(心流区间[0.6,0.8])'],
                        self.sess['summary']['适配率(心流区间[0.6,0.8])'] + 1e-9)

    def test_learner_sim_true_mastery_differs_from_estimate(self):
        """模拟器的隐藏真值必须与诊断估计不同，否则反馈闭环没有纠偏空间。"""
        sim = LearnerSim(PROF)
        d = DiagnoserAgent(load(), None).diagnose(PROF)
        kp = list(sim.true_mastery)[0]
        self.assertNotEqual(round(sim.true_mastery[kp], 4),
                            round(d['mastery'][kp], 4))

    def test_learner_sim_quiz_is_deterministic(self):
        sim = LearnerSim(PROF)
        self.assertEqual(sim.quiz('折扣因子的作用与取值', '进阶'),
                         sim.quiz('折扣因子的作用与取值', '进阶'))

    def test_learner_sim_correct_count_within_range(self):
        for kp in list(LearnerSim(PROF).true_mastery)[:5]:
            c, n = LearnerSim(PROF).quiz(kp, '进阶')
            self.assertGreaterEqual(c, 0)
            self.assertLessEqual(c, n)


# ═══════════════════════════════════════════════════════════ 8 报告
class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = load()
        cls.orc = Orchestrator(backend='mock', rounds=2, budget_kp=2, verbose=False)
        cls.sess = cls.orc.run(PROF, write=False)
        cls.rep = build_report(cls.sess, cls.kb)

    def test_report_has_three_sections(self):
        for k in ('盲区定位', '难度匹配曲线', '路径规划', '指标'):
            self.assertIn(k, self.rep)

    def test_blind_rows_are_sorted_ascending(self):
        rows = [r['掌握度'] for r in self.rep['盲区定位']['展示行']]
        self.assertEqual(rows, sorted(rows))

    def test_blind_chart_rows_capped_at_8(self):
        self.assertLessEqual(len(self.rep['盲区定位']['展示行']), 8)

    def test_curve_has_one_point_per_delivered_target(self):
        n = sum(len(rd['items']) for rd in self.sess['rounds'])
        self.assertEqual(len(self.rep['难度匹配曲线']), n)

    def test_curve_records_flow_band_membership(self):
        for c in self.rep['难度匹配曲线']:
            self.assertEqual(c['落入心流区间'],
                             0.60 <= c['实测正确率'] <= 0.80)

    def test_path_nodes_match_curve_points(self):
        self.assertEqual(len(self.rep['路径规划']['节点']),
                         len(self.rep['难度匹配曲线']))

    def test_metrics_table_has_11_rows(self):
        self.assertGreaterEqual(len(self.rep['指标']), 11)

    def test_every_metric_has_value_and_explanation(self):
        for m in self.rep['指标']:
            self.assertTrue(m['值'])
            self.assertTrue(m['说明'])

    def test_svg_blind_is_wellformed_xml(self):
        html = render_html(self.rep, self.sess)
        start = html.index('<svg viewBox="0 0 880')
        end = html.index('</svg>', start) + 6
        ET.fromstring(html[start:end])

    def test_html_contains_three_charts(self):
        html = render_html(self.rep, self.sess)
        self.assertEqual(html.count('<svg viewBox'), 3)

    def test_html_is_self_contained_no_external_assets(self):
        html = render_html(self.rep, self.sess)
        self.assertNotIn('http://', html)
        self.assertNotIn('https://', html)
        self.assertNotIn('<script', html)

    def test_html_declares_utf8(self):
        self.assertIn('charset="utf-8"', render_html(self.rep, self.sess))

    def test_html_states_the_simulator_limitation(self):
        html = render_html(self.rep, self.sess)
        self.assertIn('模拟', html)

    def test_html_states_mock_backend_limitation(self):
        html = render_html(self.rep, self.sess)
        self.assertIn('mock', html)


# ═══════════════════════════════════════════════════════════ 9 落盘产物
class TestArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmp = tempfile.mkdtemp(prefix='phmarl_eval_')
        cls.orc = Orchestrator(backend='mock', rounds=2, budget_kp=2, verbose=False)
        cls.sess = cls.orc.run(PROF, out_dir=os.path.join(cls.tmp, 'L001'))

    def _path(self, name):
        return os.path.join(self.tmp, 'L001', name)

    def test_resources_file_written(self):
        self.assertTrue(os.path.isfile(self._path('resources.md')))

    def test_audit_trace_written_and_parsable(self):
        with open(self._path('audit_trace.json'), encoding='utf-8') as f:
            json.load(f)

    def test_trace_jsonl_lines_are_json(self):
        with open(self._path('trace.jsonl'), encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    json.loads(line)

    def test_report_json_written(self):
        with open(self._path('report.json'), encoding='utf-8') as f:
            self.assertIn('盲区定位', json.load(f))

    def test_report_html_written(self):
        self.assertTrue(os.path.isfile(self._path('report.html')))

    def test_session_json_written(self):
        with open(self._path('session.json'), encoding='utf-8') as f:
            self.assertIn('summary', json.load(f))

    def test_audit_trace_lists_claims_with_evidence(self):
        with open(self._path('audit_trace.json'), encoding='utf-8') as f:
            data = json.load(f)
        self.assertTrue(data['resources'])
        for r in data['resources']:
            for c in r['claims']:
                self.assertIn('evidence', c)
                self.assertIn('verdict', c)

    def test_delivered_resources_contain_evidence_citations(self):
        with open(self._path('resources.md'), encoding='utf-8') as f:
            txt = f.read()
        self.assertIn('证据出处', txt)


# ═══════════════════════════════════════════════════════════ 10 共享口径
class TestSharedConventions(unittest.TestCase):
    def test_question_bank_has_at_least_90_questions(self):
        p = os.path.join(ROOT, 'data', 'bank', 'questions.json')
        with open(p, encoding='utf-8') as f:
            bank = json.load(f)
        self.assertGreaterEqual(len(bank), 90)

    def test_every_question_has_required_fields(self):
        p = os.path.join(ROOT, 'data', 'bank', 'questions.json')
        with open(p, encoding='utf-8') as f:
            bank = json.load(f)
        for q in bank:
            for f in ('knowledge_point', 'difficulty', 'stem', 'answer',
                      'explain', 'source'):
                self.assertIn(f, q)

    def test_question_difficulties_are_from_ladder(self):
        p = os.path.join(ROOT, 'data', 'bank', 'questions.json')
        with open(p, encoding='utf-8') as f:
            bank = json.load(f)
        for q in bank:
            self.assertIn(q['difficulty'], ('基础', '进阶', '挑战'))

    def test_profile_files_are_at_least_two(self):
        from run import list_profiles
        self.assertGreaterEqual(len(list_profiles()), 2)

    def test_every_profile_declares_two_data_sources(self):
        from run import list_profiles, load_profile
        for pid in list_profiles():
            prof = load_profile(pid)
            self.assertGreaterEqual(len(prof.get('数据源', [])), 2, pid)

    def test_no_profile_contains_real_phone_or_id_numbers(self):
        """交付物里不得出现手机号/身份证号 —— 测试把它钉死。"""
        import re
        from run import list_profiles, load_profile
        phone = re.compile(r'1[3-9]\d{9}')
        for pid in list_profiles():
            raw = json.dumps(load_profile(pid), ensure_ascii=False)
            self.assertIsNone(phone.search(raw), '%s 里出现手机号' % pid)


class TestCoverage(unittest.TestCase):
    """元测试：确认测试规模达到交付要求（≥50 条）。"""

    def test_test_count_at_least_50(self):
        n = 0
        for name, obj in globals().items():
            if name.startswith('Test') and isinstance(obj, type):
                n += len([m for m in dir(obj) if m.startswith('test_')])
        self.assertGreaterEqual(n, 50, '当前用例数 %d < 50' % n)


if __name__ == '__main__':
    unittest.main(verbosity=2)
