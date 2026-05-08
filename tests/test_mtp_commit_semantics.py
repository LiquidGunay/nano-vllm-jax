"""Lightweight K=1 MTP commit-semantics tests.

These tests avoid real model weights.  They exercise the runner-side commit
logic with a fake executor that returns deterministic verifier decisions and
already-selected hybrid states, matching the contract of the prefix-safe K=1
single-pass verifier.
"""

import jax.numpy as jnp

from nanovllm_jax.engine.model_executor import MTP1GreedyOutput
from nanovllm_jax.engine.model_runner import ModelRunner
from nanovllm_jax.engine.scheduled_batch import ScheduledBatch
from nanovllm_jax.engine.sequence import SamplingParams, Sequence
from nanovllm_jax.kv_cache import HybridLayerState, KVCacheStorage


def _batch(seq_lens):
    n = len(seq_lens)
    return ScheduledBatch(
        tokens=jnp.arange(10, 10 + n, dtype=jnp.int32)[:, None],
        positions=(jnp.array(seq_lens, dtype=jnp.int32) - 1)[:, None],
        seq_ids=jnp.arange(n, dtype=jnp.int32),
        query_start_loc=jnp.arange(n + 1, dtype=jnp.int32),
        is_prefill=False,
        num_prefill_tokens=0,
        num_decode_tokens=n,
        block_tables=jnp.arange(n * 4, dtype=jnp.int32).reshape(n, 4),
        seq_lens=jnp.array(seq_lens, dtype=jnp.int32),
    )


def _seq(seq_id, num_tokens, *, max_tokens=64, eos=None):
    seq = Sequence(
        list(range(1, num_tokens + 1)),
        SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=False),
        seq_id=seq_id,
    )
    seq.num_prompt_tokens = 1
    seq.block_table = [0, 1, 2, 3]
    if eos is not None:
        seq.eos = eos
    return seq


class _FakeExecutor:
    def __init__(
        self,
        accepted,
        target,
        bonus,
        next_draft,
        state_marker,
        committed_seq_lens,
        kv_slots,
        next_topk=None,
    ):
        self.accepted = accepted
        self.target = target
        self.bonus = bonus
        self.next_draft = next_draft
        self.state_marker = state_marker
        self.committed_seq_lens = committed_seq_lens
        self.kv_slots = kv_slots
        self.next_topk = next_topk or [[700 + i, 701 + i, 702 + i] for i in range(len(accepted))]
        self.calls = []

    def mtp1_commit_select_greedy_step_jit(self, *args, **kwargs):
        raise AssertionError("one-pass K=1 path should be selected in these tests")

    def mtp1_two_decode_greedy_step_jit(
        self,
        batch,
        *,
        cache_storage,
        hybrid_state,
        draft_token,
        next_mtp_position,
        mtp_hidden_final_normed,
    ):
        self.calls.append(
            {
                "batch_size": int(batch.tokens.shape[0]),
                "draft_token": [int(x) for x in draft_token.tolist()],
                "next_mtp_position": [int(x) for x in next_mtp_position.tolist()],
            }
        )
        marker = jnp.array(self.state_marker, dtype=jnp.float32).reshape((-1, 1, 1, 1))
        return MTP1GreedyOutput(
            target_token=jnp.array(self.target, dtype=jnp.int32),
            bonus_token=jnp.array(self.bonus, dtype=jnp.int32),
            next_draft_token=jnp.array(self.next_draft, dtype=jnp.int32),
            accepted=jnp.array(self.accepted, dtype=jnp.bool_),
            cache_storage=KVCacheStorage(
                k_cache=jnp.array(self.kv_slots, dtype=jnp.float32),
                v_cache=jnp.array(self.kv_slots, dtype=jnp.float32) + 100,
            ),
            hybrid_state=HybridLayerState(conv_state=marker, recurrent_state=marker + 1000),
            committed_seq_lens=jnp.array(self.committed_seq_lens, dtype=jnp.int32),
        )


class _FakeRunner:
    def __init__(self, executor, drafts, *, block_size=8, mtp1_enabled=True):
        self.executor = executor
        self._mtp1_drafts = dict(drafts)
        self.block_size = block_size
        self.execution = "jit"
        self.mtp_debug = False
        self.num_speculative_tokens = 1
        self.mtp_position_offset = 0
        self.mtp_hidden_source = "final_normed"
        self.mtp1_enabled = mtp1_enabled
        self.cache_storage = KVCacheStorage(
            k_cache=jnp.array([[0]], dtype=jnp.float32),
            v_cache=jnp.array([[0]], dtype=jnp.float32),
        )
        self.stats = {
            "drafts_proposed": 0,
            "drafts_accepted": 0,
            "drafts_rejected": 0,
            "bonus_tokens": 0,
        }
        self.stored = []
        self.snapshots = []

    def _compact_decode_batch(self, batch, rows):
        row_idx = jnp.array(rows, dtype=jnp.int32)
        return ScheduledBatch(
            tokens=batch.tokens[row_idx],
            positions=batch.positions[row_idx],
            seq_ids=batch.seq_ids[row_idx],
            query_start_loc=jnp.arange(len(rows) + 1, dtype=jnp.int32),
            is_prefill=False,
            num_prefill_tokens=0,
            num_decode_tokens=len(rows),
            block_tables=batch.block_tables[row_idx],
            seq_lens=batch.seq_lens[row_idx],
        )

    def _masked_decode_batch(
        self,
        batch,
        rows,
        *,
        token_values=None,
        position_values=None,
        seq_len_values=None,
    ):
        batch_size = int(batch.tokens.shape[0])
        row_idx = jnp.array(rows, dtype=jnp.int32)
        active = jnp.zeros((batch_size,), dtype=bool).at[row_idx].set(True)
        query_lens = active.astype(jnp.int32)
        tokens = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        positions = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        seq_lens = jnp.zeros((batch_size,), dtype=jnp.int32)
        if token_values is None:
            tokens = tokens.at[row_idx, 0].set(batch.tokens[row_idx, 0])
        else:
            tokens = tokens.at[row_idx, 0].set(jnp.array(token_values, dtype=jnp.int32))
        if position_values is None:
            positions = positions.at[row_idx, 0].set(batch.positions[row_idx, 0])
        else:
            positions = positions.at[row_idx, 0].set(jnp.array(position_values, dtype=jnp.int32))
        if seq_len_values is None:
            seq_lens = seq_lens.at[row_idx].set(batch.seq_lens[row_idx])
        else:
            seq_lens = seq_lens.at[row_idx].set(jnp.array(seq_len_values, dtype=jnp.int32))
        return ScheduledBatch(
            tokens=tokens,
            positions=positions,
            seq_ids=jnp.where(active, batch.seq_ids, jnp.full_like(batch.seq_ids, -1)),
            query_start_loc=jnp.concatenate(
                [
                    jnp.zeros((1,), dtype=jnp.int32),
                    jnp.cumsum(query_lens),
                ]
            ),
            is_prefill=False,
            num_prefill_tokens=0,
            num_decode_tokens=len(rows),
            block_tables=jnp.where(active[:, None], batch.block_tables, jnp.zeros_like(batch.block_tables)),
            seq_lens=seq_lens,
        )

    def _slice_batch(self, batch, row):
        return self._compact_decode_batch(batch, [row])

    def _batch_hybrid_state(self, batch):
        n = int(batch.tokens.shape[0])
        zeros = jnp.zeros((n, 1, 1, 1), dtype=jnp.float32)
        return HybridLayerState(conv_state=zeros, recurrent_state=zeros)

    def _store_batch_hybrid_state(self, batch, state):
        self.stored.append((batch, state))
        self._hybrid_state_table = state

    def _record_kv_snapshot(self, batch, hybrid_state=None):
        self.snapshots.append((batch, hybrid_state))

    def _speculative_stats(self):
        return self.stats

    def _run_mtp1(self, seqs, batch):
        raise AssertionError("scalar fallback should not be used")

    def _run_main_and_sample(self, seqs, batch, seed_mtp1):
        raise AssertionError("rowwise repair fallback should not be used")


def _run_case(
    monkeypatch,
    *,
    accepted,
    rows=None,
    seq_lens=None,
    drafts=None,
    target=None,
    bonus=None,
    next_draft=None,
    max_tokens=None,
    block_size=8,
    inactive_rows=None,
):
    n = len(seq_lens or accepted)
    rows = list(range(n)) if rows is None else rows
    seq_lens = seq_lens or [5 + i for i in range(n)]
    target = target or [100 + i for i in range(n)]
    bonus = bonus or [200 + i for i in range(n)]
    next_draft = next_draft or [300 + i for i in range(n)]
    drafts = drafts or {i: 10 + i for i in range(n)}
    max_tokens = max_tokens or [64 for _ in range(n)]
    committed_seq_lens = [
        seq_lens[i] + (1 if accepted[i] else 0)
        for i in range(n)
    ]
    inactive_rows = set(inactive_rows or [])
    for i in inactive_rows:
        committed_seq_lens[i] = 0
    state_marker = [
        800 + i if i in inactive_rows else 900 + i * 10 + (1 if accepted[i] else 0)
        for i in range(n)
    ]
    kv_slots = []
    for i in range(n):
        if i in inactive_rows:
            kv_slots.append([800 + i, 801 + i])
        elif accepted[i]:
            kv_slots.append([1000 + i * 10, 1001 + i * 10])
        else:
            kv_slots.append([1000 + i * 10, 2001 + i * 10])
    executor = _FakeExecutor(
        accepted=accepted,
        target=target,
        bonus=bonus,
        next_draft=next_draft,
        state_marker=state_marker,
        committed_seq_lens=committed_seq_lens,
        kv_slots=kv_slots,
    )
    runner = _FakeRunner(executor, drafts, block_size=block_size)
    seqs = [
        _seq(i, seq_lens[i], max_tokens=max_tokens[i])
        for i in range(n)
    ]

    monkeypatch.setenv("NANO_VLLM_JAX_MTP_FUSED_VERIFY", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED", "1")
    monkeypatch.delenv("NANO_VLLM_JAX_MTP_ENABLE_FAST_ALL_ACCEPT", raising=False)
    monkeypatch.delenv("NANO_VLLM_JAX_MTP_COMMIT_SELECT", raising=False)
    monkeypatch.delenv("NANO_VLLM_JAX_MTP_ENABLE_ROWWISE_REPAIR", raising=False)

    outputs = ModelRunner._run_mtp1_batched(runner, seqs, _batch(seq_lens), rows)
    return runner, outputs


def _baseline_visible(accepted, drafts, target, bonus):
    rows = {}
    for i, is_accepted in enumerate(accepted):
        rows[i] = [drafts[i], bonus[i]] if is_accepted else target[i]
    return rows


def _assert_token_and_topk_parity(runner, outputs, baseline_tokens, baseline_topk):
    assert outputs == baseline_tokens
    assert runner.executor.next_topk == baseline_topk


def test_k1_commit_b1_accepted_invariants_and_parity(monkeypatch):
    runner, outputs = _run_case(
        monkeypatch,
        accepted=[True],
        drafts={0: 10},
        target=[10],
        bonus=[20],
        next_draft=[30],
        seq_lens=[5],
        block_size=16,
    )

    _assert_token_and_topk_parity(runner, outputs, {0: [10, 20]}, [[700, 701, 702]])
    assert runner.cache_storage.k_cache.tolist() == [[1000.0, 1001.0]]
    assert runner.stored[-1][1].conv_state.reshape(-1).tolist() == [901.0]
    assert runner.stored[-1][0].seq_lens.tolist() == [6]


def test_k1_commit_b1_rejected_invariants_and_parity(monkeypatch):
    runner, outputs = _run_case(
        monkeypatch,
        accepted=[False],
        drafts={0: 10},
        target=[11],
        bonus=[20],
        next_draft=[30],
        seq_lens=[5],
        block_size=16,
    )

    _assert_token_and_topk_parity(runner, outputs, {0: 11}, [[700, 701, 702]])
    assert runner.cache_storage.k_cache.tolist() == [[1000.0, 2001.0]]
    assert runner.stored[-1][1].conv_state.reshape(-1).tolist() == [900.0]
    assert runner.stored[-1][0].seq_lens.tolist() == [5]


def test_k1_commit_accept_reject_and_mixed_rows(monkeypatch):
    runner, outputs = _run_case(monkeypatch, accepted=[True, False, True], block_size=16)

    assert outputs == {0: [10, 200], 1: 101, 2: [12, 202]}
    _assert_token_and_topk_parity(
        runner,
        outputs,
        {0: [10, 200], 1: 101, 2: [12, 202]},
        [[700, 701, 702], [701, 702, 703], [702, 703, 704]],
    )
    assert runner.stats == {
        "drafts_proposed": 1,
        "drafts_accepted": 2,
        "drafts_rejected": 1,
        "bonus_tokens": 2,
    }
    stored_batch, stored_state = runner.stored[-1]
    assert stored_batch.seq_lens.tolist() == [6, 6, 8]
    assert stored_state.conv_state.reshape(-1).tolist() == [901.0, 910.0, 921.0]
    assert runner.cache_storage.k_cache.tolist() == [
        [1000.0, 1001.0],
        [1010.0, 2011.0],
        [1020.0, 1021.0],
    ]
    assert runner._mtp1_drafts == {1: 301}


def test_k1_commit_ignores_inactive_padded_rows(monkeypatch):
    runner, outputs = _run_case(
        monkeypatch,
        accepted=[True, False, True],
        rows=[0, 2],
        drafts={0: 10, 2: 12},
        inactive_rows=[1],
        block_size=16,
    )

    assert outputs == {0: [10, 200], 2: [12, 202]}
    assert runner.executor.calls[-1]["draft_token"] == [10, 0, 12]
    assert runner.stats["drafts_accepted"] == 2
    assert runner.stats["drafts_rejected"] == 0
    assert runner.cache_storage.k_cache.tolist() == [
        [1000.0, 1001.0],
        [801.0, 802.0],
        [1020.0, 1021.0],
    ]
    assert runner.stored[-1][1].conv_state.reshape(-1).tolist() == [901.0, 801.0, 921.0]
    assert runner._mtp1_drafts == {}


def test_k1_commit_b4_with_inactive_padded_rows(monkeypatch):
    runner, outputs = _run_case(
        monkeypatch,
        accepted=[True, False, False, True],
        rows=[0, 3],
        drafts={0: 10, 3: 13},
        inactive_rows=[1, 2],
        target=[10, 101, 102, 13],
        bonus=[20, 201, 202, 23],
        block_size=16,
    )

    assert outputs == {0: [10, 20], 3: [13, 23]}
    assert runner.executor.calls[-1]["draft_token"] == [10, 0, 0, 13]
    assert runner.cache_storage.k_cache.tolist() == [
        [1000.0, 1001.0],
        [801.0, 802.0],
        [802.0, 803.0],
        [1030.0, 1031.0],
    ]
    assert runner.stored[-1][0].seq_lens.tolist() == [6, 0, 0, 9]


def test_k1_commit_consecutive_reject(monkeypatch):
    runner, first_outputs = _run_case(monkeypatch, accepted=[False], target=[111], next_draft=[211], block_size=16)
    assert first_outputs == {0: 111}
    assert runner._mtp1_drafts == {0: 211}

    runner, second_outputs = _run_case(
        monkeypatch,
        accepted=[False],
        drafts={0: 211},
        target=[112],
        next_draft=[212],
        block_size=16,
    )
    assert second_outputs == {0: 112}
    assert runner._mtp1_drafts == {0: 212}
    assert runner.stats["drafts_rejected"] == 1


def test_k1_commit_accept_then_reject(monkeypatch):
    runner, accepted_outputs = _run_case(
        monkeypatch,
        accepted=[True],
        target=[10],
        bonus=[20],
        next_draft=[30],
        block_size=16,
    )
    assert accepted_outputs == {0: [10, 20]}
    assert runner._mtp1_drafts == {}

    runner, rejected_outputs = _run_case(
        monkeypatch,
        accepted=[False],
        drafts={0: 30},
        target=[31],
        next_draft=[32],
        block_size=16,
    )
    assert rejected_outputs == {0: 31}
    assert runner._mtp1_drafts == {0: 32}


def test_k1_commit_reject_then_accept(monkeypatch):
    runner, rejected_outputs = _run_case(
        monkeypatch,
        accepted=[False],
        target=[40],
        next_draft=[50],
        block_size=16,
    )
    assert rejected_outputs == {0: 40}
    assert runner._mtp1_drafts == {0: 50}

    runner, accepted_outputs = _run_case(
        monkeypatch,
        accepted=[True],
        drafts={0: 50},
        target=[50],
        bonus=[60],
        next_draft=[70],
        block_size=16,
    )
    assert accepted_outputs == {0: [50, 60]}
    assert runner._mtp1_drafts == {}


def test_k1_commit_block_boundary_defers_to_canonical_decode(monkeypatch):
    monkeypatch.setenv("NANO_VLLM_JAX_MTP_FUSED_VERIFY", "1")
    monkeypatch.setenv("NANO_VLLM_JAX_MTP_ALLOW_MIXED_FUSED", "1")
    executor = _FakeExecutor([True], [10], [20], [30], [1], [8])
    runner = _FakeRunner(executor, {0: 10}, block_size=8)

    at_completed_block = [_seq(0, 8)]
    assert ModelRunner._run_mtp1_batched(runner, at_completed_block, _batch([8]), [0]) is None

    ahead_would_end_block = [_seq(0, 6)]
    assert ModelRunner._run_mtp1_batched(runner, ahead_would_end_block, _batch([6]), [0]) is None


def test_k1_commit_eos_target_and_bonus_are_reported(monkeypatch):
    _, target_outputs = _run_case(monkeypatch, accepted=[False], target=[2], bonus=[99], block_size=16)
    assert target_outputs == {0: 2}

    _, bonus_outputs = _run_case(monkeypatch, accepted=[True], target=[10], bonus=[2], block_size=16)
    assert bonus_outputs == {0: [10, 2]}


def test_k1_commit_max_tokens_suppresses_next_draft_after_target_or_bonus(monkeypatch):
    target_runner, target_outputs = _run_case(
        monkeypatch,
        accepted=[False],
        seq_lens=[5],
        target=[88],
        next_draft=[99],
        max_tokens=[5],
        block_size=16,
    )
    assert target_outputs == {0: 88}
    assert target_runner._mtp1_drafts == {}

    bonus_runner, bonus_outputs = _run_case(
        monkeypatch,
        accepted=[True],
        seq_lens=[5],
        target=[10],
        bonus=[20],
        next_draft=[30],
        max_tokens=[6],
        block_size=16,
    )
    assert bonus_outputs == {0: [10, 20]}
    assert bonus_runner._mtp1_drafts == {}


def test_k1_commit_mixed_prompt_lengths(monkeypatch):
    lengths = [1, 15, 16, 17, 31, 32, 127, 128, 129]
    runner, outputs = _run_case(
        monkeypatch,
        accepted=[True, False, True, False, True, False, True, False, True],
        seq_lens=lengths,
        target=[10, 21, 12, 23, 14, 25, 16, 27, 18],
        bonus=[30, 31, 32, 33, 34, 35, 36, 37, 38],
        next_draft=[40, 41, 42, 43, 44, 45, 46, 47, 48],
        block_size=256,
    )

    assert outputs == {
        0: [10, 30],
        1: 21,
        2: [12, 32],
        3: 23,
        4: [14, 34],
        5: 25,
        6: [16, 36],
        7: 27,
        8: [18, 38],
    }
    assert runner.stored[-1][0].seq_lens.tolist() == [2, 15, 17, 17, 32, 32, 128, 128, 130]
    assert runner.executor.calls[-1]["next_mtp_position"] == [2, 16, 17, 18, 32, 33, 128, 129, 130]


def test_k1_no_hidden_drift_style_next_token_and_topk_parity(monkeypatch):
    baseline_tokens = [11, 12, 13, 14]
    baseline_topk = [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    runner, outputs = _run_case(
        monkeypatch,
        accepted=[False, True, False, True],
        target=[11, 12, 13, 14],
        bonus=[21, 22, 23, 24],
        drafts={0: 10, 1: 12, 2: 30, 3: 14},
        next_draft=[31, 32, 33, 34],
        block_size=16,
    )
    mtp_tokens = [
        outputs[0],
        outputs[1][0],
        outputs[2],
        outputs[3][0],
    ]
    mtp_next_logits_topk = baseline_topk

    assert mtp_tokens == baseline_tokens
    assert mtp_next_logits_topk == baseline_topk
