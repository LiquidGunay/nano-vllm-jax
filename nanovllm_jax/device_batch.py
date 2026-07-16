"""Runner-owned materialization of host schedule plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from nanovllm_jax.batch import SchedulePlan


@dataclass(frozen=True)
class HostBatch:
    """Host facts retained beside one materialized device batch."""

    seq_ids: tuple[int, ...] = ()
    query_lens: tuple[int, ...] = ()
    seq_lens: tuple[int, ...] = ()
    block_tables: tuple[tuple[int, ...], ...] = ()
    prefill_is_final: tuple[bool, ...] = ()
    decode_steps: int = 1
    uses_static_decode_metadata: bool = False


@dataclass
class DeviceBatch:
    """Fixed-shape arrays consumed by the runner and executor."""

    tokens: jax.Array
    positions: jax.Array
    seq_ids: jax.Array
    query_start_loc: jax.Array
    is_prefill: bool
    num_prefill_tokens: int
    num_decode_tokens: int
    block_tables: jax.Array
    seq_lens: jax.Array
    host: HostBatch = field(default_factory=HostBatch)
    packed_prefill: bool = False
    token_row_ids: jax.Array | None = None

    @property
    def batch_size(self) -> int:
        if self.packed_prefill:
            return int(self.block_tables.shape[0])
        return int(self.tokens.shape[0])

    @property
    def query_lens(self) -> jax.Array:
        return jnp.diff(self.query_start_loc).astype(jnp.int32)

    @property
    def active_decode_rows(self) -> jax.Array:
        return (self.seq_ids >= 0) & (self.query_lens > 0)

    @property
    def prefill_final_flags(self) -> list[bool]:
        if not self.host.prefill_is_final:
            return [True] * self.batch_size
        return [bool(value) for value in self.host.prefill_is_final]


class BatchMaterializer:
    """Pad a host plan and reuse shape-stable decode arrays."""

    def __init__(
        self,
        *,
        execution: str,
        device_token_carry: bool,
        static_decode_metadata: bool,
        resident_decode_metadata: bool,
        static_decode_seq_lens_carry: bool,
    ) -> None:
        self.execution = execution
        self.device_token_carry = device_token_carry
        self.static_decode_metadata = static_decode_metadata
        self.resident_decode_metadata = resident_decode_metadata
        self.static_decode_seq_lens_carry = static_decode_seq_lens_carry
        self._decode_constants: dict[tuple[Any, ...], dict[str, jax.Array]] = {}
        self._decode_metadata: dict[tuple[Any, ...], dict[str, Any]] = {}

    def materialize(self, plan: SchedulePlan) -> DeviceBatch:
        return self._packed_prefill(plan) if plan.bucket.packed_prefill else self._dense(plan)

    def _dense(self, plan: SchedulePlan) -> DeviceBatch:
        width = plan.bucket.query_tokens
        block_width = plan.bucket.block_table_width
        tokens = [list(row.token_ids) + [0] * (width - row.query_len) for row in plan.rows]
        positions = [list(row.positions) + [0] * (width - row.query_len) for row in plan.rows]
        block_tables = [
            list(row.block_table) + [0] * (block_width - len(row.block_table)) for row in plan.rows
        ]
        seq_ids = [row.seq_id for row in plan.rows]
        query_lens = [row.query_len for row in plan.rows]
        seq_lens = [row.seq_len for row in plan.rows]
        padding = plan.bucket.batch_size - len(plan.rows)
        tokens.extend([[0] * width for _ in range(padding)])
        positions.extend([[0] * width for _ in range(padding)])
        block_tables.extend([[0] * block_width for _ in range(padding)])
        seq_ids.extend([-1] * padding)
        query_lens.extend([0] * padding)
        seq_lens.extend([0] * padding)
        query_start_loc = self._query_start_loc(query_lens)
        use_static = self._can_reuse_decode_arrays(plan, tokens)
        arrays = (
            self._static_decode_arrays(
                tokens=tokens,
                positions=positions,
                seq_ids=seq_ids,
                query_start_loc=query_start_loc,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_lens=query_lens,
            )
            if use_static
            else self._device_put(
                tokens, positions, seq_ids, query_start_loc, block_tables, seq_lens
            )
        )
        return DeviceBatch(
            tokens=arrays[0],
            positions=arrays[1],
            seq_ids=arrays[2],
            query_start_loc=arrays[3],
            is_prefill=plan.is_prefill,
            num_prefill_tokens=plan.num_scheduled_tokens if plan.is_prefill else 0,
            num_decode_tokens=0 if plan.is_prefill else plan.num_scheduled_tokens,
            block_tables=arrays[4],
            seq_lens=arrays[5],
            host=self._host_batch(
                seq_ids,
                query_lens,
                seq_lens,
                block_tables,
                prefill_is_final=(
                    tuple(row.prefill_is_final for row in plan.rows) if plan.is_prefill else ()
                ),
                decode_steps=1 if plan.is_prefill else plan.decode_steps,
                uses_static_decode_metadata=use_static,
            ),
        )

    def _packed_prefill(self, plan: SchedulePlan) -> DeviceBatch:
        if not plan.is_prefill:
            raise ValueError("packed layout is valid only for prefill")
        token_bucket = plan.bucket.query_tokens
        actual_tokens = plan.num_scheduled_tokens
        if actual_tokens > token_bucket:
            raise ValueError("scheduled tokens exceed the packed prefill bucket")
        block_width = plan.bucket.block_table_width
        tokens: list[int] = []
        positions: list[int] = []
        token_row_ids: list[int] = []
        block_tables: list[list[int]] = []
        seq_ids: list[int] = []
        query_lens: list[int] = []
        seq_lens: list[int] = []
        for row_index, row in enumerate(plan.rows):
            tokens.extend(row.token_ids)
            positions.extend(row.positions)
            token_row_ids.extend([row_index] * row.query_len)
            block_tables.append(list(row.block_table) + [0] * (block_width - len(row.block_table)))
            seq_ids.append(row.seq_id)
            query_lens.append(row.query_len)
            seq_lens.append(row.seq_len)
        padding_rows = plan.bucket.batch_size - len(plan.rows)
        block_tables.extend([[0] * block_width for _ in range(padding_rows)])
        seq_ids.extend([-1] * padding_rows)
        query_lens.extend([0] * padding_rows)
        seq_lens.extend([0] * padding_rows)
        token_padding = token_bucket - actual_tokens
        tokens.extend([0] * token_padding)
        positions.extend([0] * token_padding)
        token_row_ids.extend([0] * token_padding)
        query_start_loc = self._query_start_loc(query_lens)
        arrays = self._device_put(
            [tokens],
            [positions],
            seq_ids,
            query_start_loc,
            block_tables,
            seq_lens,
        )
        return DeviceBatch(
            tokens=arrays[0],
            positions=arrays[1],
            seq_ids=arrays[2],
            query_start_loc=arrays[3],
            is_prefill=True,
            num_prefill_tokens=actual_tokens,
            num_decode_tokens=0,
            block_tables=arrays[4],
            seq_lens=arrays[5],
            host=self._host_batch(
                seq_ids,
                query_lens,
                seq_lens,
                block_tables,
                prefill_is_final=tuple(row.prefill_is_final for row in plan.rows),
            ),
            packed_prefill=True,
            token_row_ids=jax.device_put(np.asarray([token_row_ids], dtype=np.int32)),
        )

    def _can_reuse_decode_arrays(self, plan: SchedulePlan, tokens: list[list[Any]]) -> bool:
        return bool(
            not plan.is_prefill
            and self.static_decode_metadata
            and self.execution in {"decode-jit", "jit"}
            and self.device_token_carry
            and plan.bucket.query_tokens == 1
            and all(
                row.carries_device_token and int(tokens[index][0]) == 0
                for index, row in enumerate(plan.rows)
            )
        )

    def _static_decode_arrays(
        self,
        *,
        tokens: list[list[Any]],
        positions: list[list[int]],
        seq_ids: list[int],
        query_start_loc: list[int],
        block_tables: list[list[int]],
        seq_lens: list[int],
        query_lens: list[int],
    ) -> tuple[jax.Array, ...]:
        token_shape = (len(tokens), len(tokens[0]))
        block_shape = (len(block_tables), len(block_tables[0]))
        device_seq_ids = (
            [index if query_len > 0 else -1 for index, query_len in enumerate(query_lens)]
            if self.resident_decode_metadata
            else seq_ids
        )
        constant_key = (token_shape, tuple(device_seq_ids), tuple(query_lens))
        constants = self._decode_constants.get(constant_key)
        if constants is None:
            constants = {
                "tokens": jax.device_put(np.asarray(tokens, dtype=np.int32)),
                "positions": jax.device_put(np.zeros_like(np.asarray(positions, dtype=np.int32))),
                "seq_ids": jax.device_put(np.asarray(device_seq_ids, dtype=np.int32)),
                "query_start_loc": jax.device_put(np.asarray(query_start_loc, dtype=np.int32)),
            }
            self._decode_constants[constant_key] = constants
        metadata_key = (
            token_shape,
            block_shape,
            tuple(query_lens),
            "resident"
            if self.resident_decode_metadata
            else (tuple(seq_ids), tuple(map(tuple, block_tables))),
        )
        metadata = self._decode_metadata.get(metadata_key)
        if metadata is None:
            metadata = {
                "block_tables": jax.device_put(
                    np.zeros(block_shape, dtype=np.int32)
                    if self.resident_decode_metadata
                    else np.asarray(block_tables, dtype=np.int32)
                ),
                "seq_lens": jax.device_put(
                    np.zeros((len(seq_lens),), dtype=np.int32)
                    if self.resident_decode_metadata
                    else np.asarray(seq_lens, dtype=np.int32)
                ),
            }
            self._decode_metadata[metadata_key] = metadata
        elif not (self.static_decode_seq_lens_carry or self.resident_decode_metadata):
            metadata = {
                **metadata,
                "seq_lens": jax.device_put(np.asarray(seq_lens, dtype=np.int32)),
            }
        return (
            constants["tokens"],
            constants["positions"],
            constants["seq_ids"],
            constants["query_start_loc"],
            metadata["block_tables"],
            metadata["seq_lens"],
        )

    @staticmethod
    def _device_put(*values: Any) -> tuple[jax.Array, ...]:
        return jax.device_put(tuple(np.asarray(value, dtype=np.int32) for value in values))

    @staticmethod
    def _query_start_loc(query_lens: list[int]) -> list[int]:
        result = [0]
        for query_len in query_lens:
            result.append(result[-1] + query_len)
        return result

    @staticmethod
    def _host_batch(
        seq_ids: list[int],
        query_lens: list[int],
        seq_lens: list[int],
        block_tables: list[list[int]],
        *,
        prefill_is_final: tuple[bool, ...] = (),
        decode_steps: int = 1,
        uses_static_decode_metadata: bool = False,
    ) -> HostBatch:
        return HostBatch(
            seq_ids=tuple(seq_ids),
            query_lens=tuple(query_lens),
            seq_lens=tuple(seq_lens),
            block_tables=tuple(tuple(row) for row in block_tables),
            prefill_is_final=prefill_is_final,
            decode_steps=decode_steps,
            uses_static_decode_metadata=uses_static_decode_metadata,
        )
