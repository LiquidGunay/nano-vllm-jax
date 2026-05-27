import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.summarize_profile_trace import render_markdown, summarize_trace


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
