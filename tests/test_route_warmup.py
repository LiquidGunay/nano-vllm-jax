import jax
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


def _config(*, mtp: bool = False):
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
        capacity={
            "block_size": 2,
            "num_kvcache_blocks": 16,
            "max_kv_cache_bytes": 1 << 20,
            "max_num_seqs": 4,
            "max_num_resident_seqs": 4,
            "max_num_batched_tokens": 4,
            "max_blocks_per_seq": 4,
            "prefix_cache": False,
        },
        compile={
            "dtype": "float32",
            "execution": "jit",
            "prefill_token_buckets": (4,),
            "batch_size_buckets": (1, 2, 3, 4) if mtp else (1, 4),
            "decode_block_table_buckets": (4,),
        },
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

    summary = engine.warmup_compilation()["runner"]

    assert RouteKind.PREFILL_MTP.value in summary["warmed_routes"]
    assert RouteKind.DECODE_SPECULATIVE.value in summary["warmed_routes"]
    assert summary["include_sampled_routes"] is False
    assert summary["sampled_token_fastpath_runs"] == []
    assert RouteKind.DECODE_SAMPLED.value not in summary["warmed_routes"]
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
