import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.summarize_profile_trace import render_markdown, summarize_trace
from benchmarks.benchmark_jax_server_trace import _profile_counters


def _write_trace(path, events):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"traceEvents": events}, handle)


def test_summarize_trace_filters_gpu_scope_and_patterns(tmp_path):
    trace = tmp_path / "trace.json.gz"
    _write_trace(
        trace,
        [
            {
                "ph": "M",
                "pid": 1,
                "name": "process_name",
                "args": {"name": "/device:GPU:0"},
            },
            {
                "ph": "M",
                "pid": 2,
                "name": "process_name",
                "args": {"name": "/host:CPU"},
            },
            {"ph": "X", "pid": 1, "name": "gemm_fusion_dot", "dur": 2000},
            {"ph": "X", "pid": 1, "name": "gemm_fusion_dot", "dur": 3000},
            {"ph": "X", "pid": 1, "name": "MemcpyD2D", "dur": 1000},
            {"ph": "X", "pid": 2, "name": "gemm_fusion_cpu_wrapper", "dur": 7000},
        ],
    )

    summary = summarize_trace(
        trace,
        scope="gpu",
        top_events=2,
        patterns=("gemm_fusion", "MemcpyD2D"),
    )

    assert summary["scope"] == "gpu"
    assert summary["patterns"]["gemm_fusion"] == {"total_ms": 5.0, "count": 2}
    assert summary["patterns"]["MemcpyD2D"] == {"total_ms": 1.0, "count": 1}
    assert summary["top_events_by_total_ms"] == [
        {"name": "gemm_fusion_dot", "total_ms": 5.0, "count": 2},
        {"name": "MemcpyD2D", "total_ms": 1.0, "count": 1},
    ]


def test_render_markdown_includes_pattern_and_top_event_rows(tmp_path):
    trace = tmp_path / "trace.json.gz"
    _write_trace(
        trace,
        [
            {
                "ph": "M",
                "pid": 1,
                "name": "process_name",
                "args": {"name": "/device:GPU:0"},
            },
            {"ph": "X", "pid": 1, "name": "input_reduce_fusion", "dur": 1250},
        ],
    )
    summary = {
        "traces": [
            summarize_trace(trace, scope="all", top_events=5, patterns=("input_reduce",))
        ]
    }

    markdown = render_markdown(summary)

    assert "# Profile Trace Summary" in markdown
    assert "`input_reduce`" in markdown
    assert "`input_reduce_fusion`" in markdown
    assert "1.25" in markdown


def test_summarize_trace_reports_top_hlo_ops(tmp_path):
    trace = tmp_path / "trace.json.gz"
    _write_trace(
        trace,
        [
            {
                "ph": "M",
                "pid": 1,
                "name": "process_name",
                "args": {"name": "/device:GPU:0"},
            },
            {
                "ph": "X",
                "pid": 1,
                "name": "gemm_fusion_dot_general_744",
                "dur": 2000,
                "args": {
                    "hlo_module": "jit_compiled",
                    "hlo_op": "gemm_fusion_dot_general.744",
                    "kernel_details": "regs:226 occ_pct:16.6667",
                },
            },
            {
                "ph": "X",
                "pid": 1,
                "name": "gemm_fusion_dot_general_744",
                "dur": 3000,
                "args": {
                    "hlo_module": "jit_compiled",
                    "hlo_op": "gemm_fusion_dot_general.744",
                    "kernel_details": "regs:226 occ_pct:16.6667",
                },
            },
            {
                "ph": "X",
                "pid": 1,
                "name": "MemcpyD2D",
                "dur": 1000,
                "args": {"hlo_module": "jit_copy", "hlo_op": "copy.1"},
            },
        ],
    )

    summary = summarize_trace(
        trace,
        scope="gpu",
        top_events=5,
        top_hlo_ops=2,
        patterns=("gemm_fusion",),
    )

    assert summary["top_hlo_ops_by_total_ms"] == [
        {
            "hlo_module": "jit_compiled",
            "hlo_op": "gemm_fusion_dot_general.744",
            "event": "gemm_fusion_dot_general_744",
            "total_ms": 5.0,
            "count": 2,
            "kernel_details": "regs:226 occ_pct:16.6667",
        },
        {
            "hlo_module": "jit_copy",
            "hlo_op": "copy.1",
            "event": "MemcpyD2D",
            "total_ms": 1.0,
            "count": 1,
            "kernel_details": "",
        },
    ]
    markdown = render_markdown({"traces": [summary]})
    assert "### Top HLO Ops" in markdown
    assert "`gemm_fusion_dot_general.744`" in markdown
    assert "`regs:226 occ_pct:16.6667`" in markdown


def test_summarize_trace_reports_top_gemm_kernel_shapes(tmp_path):
    trace = tmp_path / "trace.json.gz"
    _write_trace(
        trace,
        [
            {
                "ph": "M",
                "pid": 1,
                "name": "process_name",
                "args": {"name": "/device:GPU:0"},
            },
            {
                "ph": "X",
                "pid": 1,
                "name": "ampere_bf16_s16816gemm_bf16_128x64_sliced1x2",
                "dur": 2000,
                "args": {
                    "hlo_op": "command_buffer_1",
                    "kernel_details": "regs:226 grid:56,1,1 block:128,1,1",
                },
            },
            {
                "ph": "X",
                "pid": 1,
                "name": "ampere_bf16_s16816gemm_bf16_128x64_sliced1x2",
                "dur": 3000,
                "args": {
                    "hlo_op": "command_buffer_1",
                    "kernel_details": "regs:226 grid:56,1,1 block:128,1,1",
                },
            },
            {
                "ph": "X",
                "pid": 1,
                "name": "void cublasLt::splitKreduce_kernel<float>",
                "dur": 1000,
                "args": {
                    "hlo_op": "command_buffer_1",
                    "kernel_details": "regs:32 grid:56,1,1 block:128,1,1",
                },
            },
        ],
    )

    summary = summarize_trace(trace, scope="gpu", top_events=5, top_hlo_ops=3)

    assert summary["top_gemm_grids_by_total_ms"][:1] == [
        {
            "grid": "56,1,1",
            "total_ms": 6.0,
            "count": 3,
            "top_events": [
                {
                    "event": "ampere_bf16_s16816gemm_bf16_128x64_sliced1x2",
                    "count": 2,
                },
                {
                    "event": "void cublasLt::splitKreduce_kernel<float>",
                    "count": 1,
                },
            ],
        }
    ]
    assert summary["top_gemm_kernels_by_total_ms"][:2] == [
        {
            "event": "ampere_bf16_s16816gemm_bf16_128x64_sliced1x2",
            "hlo_op": "command_buffer_1",
            "grid": "56,1,1",
            "kernel_details": "regs:226 grid:56,1,1 block:128,1,1",
            "total_ms": 5.0,
            "count": 2,
        },
        {
            "event": "void cublasLt::splitKreduce_kernel<float>",
            "hlo_op": "command_buffer_1",
            "grid": "56,1,1",
            "kernel_details": "regs:32 grid:56,1,1 block:128,1,1",
            "total_ms": 1.0,
            "count": 1,
        },
    ]
    markdown = render_markdown({"traces": [summary]})
    assert "### Top GEMM Grids" in markdown
    assert "### Top GEMM Kernels" in markdown
    assert "`56,1,1`" in markdown


def test_jax_trace_profile_counters_include_scoped_gpu_cpu_rows(tmp_path):
    profile_dir = tmp_path / "plugins" / "profile" / "run"
    profile_dir.mkdir(parents=True)
    trace = profile_dir / "host.trace.json.gz"
    _write_trace(
        trace,
        [
            {
                "ph": "M",
                "pid": 1,
                "name": "process_name",
                "args": {"name": "/device:GPU:0"},
            },
            {
                "ph": "M",
                "pid": 2,
                "name": "process_name",
                "args": {"name": "/host:CPU"},
            },
            {"ph": "X", "pid": 1, "name": "gemm_fusion_dot", "dur": 4000},
            {"ph": "X", "pid": 1, "name": "MemcpyD2D", "dur": 1000},
            {
                "ph": "X",
                "pid": 2,
                "name": "$model_executor.py:501 forward_step_token_ids_jit",
                "dur": 9000,
            },
        ],
    )

    counters = _profile_counters(tmp_path)

    assert counters["ranges"]["gemm_fusion"] == {"total_ms": 4.0, "count": 1}
    assert counters["scoped_ranges"]["gpu"]["gemm_fusion"] == {
        "total_ms": 4.0,
        "count": 1,
    }
    assert counters["scoped_ranges"]["cpu"]["forward_step_token_ids_jit"] == {
        "total_ms": 9.0,
        "count": 1,
    }
    assert counters["scoped_top_events_by_total_ms"]["gpu"][0]["name"] == "gemm_fusion_dot"
    assert (
        counters["scoped_top_events_by_total_ms"]["cpu"][0]["name"]
        == "$model_executor.py:501 forward_step_token_ids_jit"
    )
