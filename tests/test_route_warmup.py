from threading import Lock

import jax
import numpy as np
import pytest

from nanovllm_jax.engine import LLMEngine
from nanovllm_jax.model import init_params
from nanovllm_jax.mtp import init_mtp_params
from nanovllm_jax.routes import RouteKind
from nanovllm_jax.runner import ModelRunner
from nanovllm_jax.scheduler import Scheduler
from nanovllm_jax.sequence import SamplingParams
from nanovllm_jax.speculation import DrafterConfig
from tests.runtime_specs import runtime_spec


def _has_cuda():
    try:
        return bool(jax.devices("gpu"))
    except RuntimeError:
        return False


def _config(*, mtp: bool = False, reachable_large_bucket: bool = False):
    capacity = (
        {
            "block_size": 2,
            "num_kvcache_blocks": 6,
            "max_kv_cache_bytes": 1 << 20,
            "max_num_seqs": 2,
            "max_num_resident_seqs": 2,
            "max_num_batched_tokens": 6,
            "max_blocks_per_seq": 4,
            "prefix_cache": False,
        }
        if reachable_large_bucket
        else {
            "block_size": 2,
            "num_kvcache_blocks": 16,
            "max_kv_cache_bytes": 1 << 20,
            "max_num_seqs": 4,
            "max_num_resident_seqs": 4,
            "max_num_batched_tokens": 4,
            "max_blocks_per_seq": 4,
            "prefix_cache": False,
        }
    )
    compile = (
        {
            "dtype": "float32",
            "execution": "jit",
            "prefill_token_buckets": (6,),
            "batch_size_buckets": (1, 2),
            "decode_block_table_buckets": (2, 4),
        }
        if reachable_large_bucket
        else {
            "dtype": "float32",
            "execution": "jit",
            "prefill_token_buckets": (4,),
            "batch_size_buckets": (1, 2, 3, 4) if mtp else (1, 4),
            "decode_block_table_buckets": (4,),
        }
    )
    return runtime_spec(
        model={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_size": 4,
            "layer_types": ("linear_attention",),
        },
        capacity=capacity,
        compile=compile,
        kernels={
            "device_token_carry": True,
            "static_decode_metadata": True,
            "resident_decode_metadata": True,
        },
        drafter=DrafterConfig.mtp(2) if mtp else None,
    )


class _Engine(LLMEngine):
    def __init__(self, config, params, mtp_params=None):
        self.config = config
        self.scheduler = Scheduler(config)
        self.model_runner = ModelRunner(config, params, mtp_params=mtp_params)
        self._next_seq_id = 0
        self._closed = False
        self._control_owner = None
        self._control_lock = Lock()


def _decode_request(engine, *, ignore_eos):
    seq = engine.add_request(
        [1, 2],
        SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=ignore_eos),
    )
    engine.step()
    engine.step()
    assert seq.is_finished


@pytest.mark.skipif(not _has_cuda(), reason="CUDA is required for JIT warmup")
def test_warmup_covers_default_and_sparse_greedy_decode_routes():
    config = _config()
    engine = _Engine(config, init_params(jax.random.PRNGKey(0), config.model))
    summary = engine.warmup_compilation()["runner"]
    warmed = {RouteKind(route) for route in summary["warmed_routes"]}
    warmed_decode = {route for route in warmed if route.value.startswith("decode_")}
    assert warmed_decode == {
        RouteKind.DECODE_RESIDENT_DENSE,
        RouteKind.DECODE_RESIDENT,
        RouteKind.DECODE_RESIDENT_METADATA,
        RouteKind.DECODE_SAMPLED,
    }

    compiled = set(engine.model_runner.executor._jit_cache)
    routes = []
    select_route = engine.model_runner._select_route

    def record_route(seqs, batch):
        route = select_route(seqs, batch)
        routes.append(route.kind)
        return route

    engine.model_runner._select_route = record_route

    _decode_request(engine, ignore_eos=False)
    assert routes == [RouteKind.PREFILL_RESIDENT, RouteKind.DECODE_RESIDENT_METADATA]
    assert set(engine.model_runner.executor._jit_cache) == compiled

    routes.clear()
    _decode_request(engine, ignore_eos=True)
    assert routes == [RouteKind.PREFILL_RESIDENT, RouteKind.DECODE_RESIDENT_DENSE]
    assert set(engine.model_runner.executor._jit_cache) == compiled


@pytest.mark.skipif(not _has_cuda(), reason="CUDA is required for JIT warmup")
def test_warmup_covers_persistent_speculative_route():
    config = _config(mtp=True)
    engine = _Engine(
        config,
        init_params(jax.random.PRNGKey(0), config.model),
        init_mtp_params(jax.random.PRNGKey(1), config),
    )
    warmed_batches = {}
    warm_route = engine.model_runner._warm_route

    def record_warm_batch(seqs, batch):
        route = warm_route(seqs, batch)
        if len(batch.host.seq_ids) == 2 and route.kind in {
            RouteKind.PREFILL_MTP,
            RouteKind.DECODE_SPECULATIVE,
        }:
            warmed_batches[route.kind] = batch
        return route

    engine.model_runner._warm_route = record_warm_batch

    summary = engine.warmup_compilation()["runner"]

    assert RouteKind.PREFILL_MTP.value in summary["warmed_routes"]
    assert RouteKind.DECODE_SPECULATIVE.value in summary["warmed_routes"]
    mtp_state = engine.model_runner.mtp_state
    assert mtp_state is not None
    assert not np.asarray(mtp_state.cache_storage.k_cache).any()
    assert not np.asarray(mtp_state.cache_storage.v_cache).any()
    assert not np.asarray(mtp_state.draft_token_ids).any()
    assert summary["include_sampled_routes"] is False
    assert summary["sampled_token_fastpath_runs"] == []
    assert RouteKind.DECODE_SAMPLED.value not in summary["warmed_routes"]

    drafter = config.drafter
    assert drafter is not None

    def physical_slots(batch, position_ranges):
        rows = []
        for block_table, positions in zip(batch.host.block_tables, position_ranges):
            rows.append(
                {
                    block_table[position // config.capacity.block_size] * config.capacity.block_size
                    + position % config.capacity.block_size
                    for position in positions
                }
            )
        return rows

    prefill = warmed_batches[RouteKind.PREFILL_MTP]
    prefill_slots = physical_slots(
        prefill,
        [range(seq_len + drafter.prefill_lookahead_tokens) for seq_len in prefill.host.seq_lens],
    )
    speculative = warmed_batches[RouteKind.DECODE_SPECULATIVE]
    speculative_slots = physical_slots(
        speculative,
        [
            range(seq_len - 1, seq_len + drafter.decode_lookahead_slots - 1)
            for seq_len in speculative.host.seq_lens
        ],
    )
    assert prefill_slots[0].isdisjoint(prefill_slots[1])
    assert speculative_slots[0].isdisjoint(speculative_slots[1])

    compiled = set(engine.model_runner.executor._jit_cache)
    routes = []
    select_route = engine.model_runner._select_route

    def record_route(seqs, batch):
        route = select_route(seqs, batch)
        routes.append(route.kind)
        return route

    engine.model_runner._select_route = record_route
    seq = engine.add_request(
        [1, 2],
        SamplingParams(temperature=0.0, max_tokens=5, ignore_eos=True),
    )
    while not seq.is_finished:
        engine.step()

    assert RouteKind.PREFILL_MTP in routes
    assert RouteKind.DECODE_SPECULATIVE in routes
    assert set(engine.model_runner.executor._jit_cache) == compiled


@pytest.mark.skipif(not _has_cuda(), reason="CUDA is required for JIT warmup")
def test_warmup_covers_reachable_large_block_table_bucket():
    config = _config(reachable_large_bucket=True)
    engine = _Engine(config, init_params(jax.random.PRNGKey(0), config.model))

    summary = engine.warmup_compilation()["runner"]
    assert not [
        skipped
        for skipped in summary["decode_skipped"]
        if skipped["batch_size"] == 2 and skipped["block_table_width"] == 4
    ]
    compiled = set(engine.model_runner.executor._jit_cache)

    sampling = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
    engine.add_requests([[1, 2, 3, 4, 5], [6]], sampling)
    while not engine.is_finished():
        engine.step()

    assert set(engine.model_runner.executor._jit_cache) == compiled
