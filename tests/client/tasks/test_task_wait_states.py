"""Tests for Task.wait() state handling.

Verifies that wait() stops at non-working states (including input_required)
and that result() raises clearly when the task can't produce a result.
"""

import asyncio
from datetime import datetime, timezone

import mcp.types
import pytest
from mcp.types import GetTaskResult

from fastmcp import FastMCP
from fastmcp.client import Client


@pytest.fixture
async def blocked_tool_server():
    """Server with a task-mode tool that blocks until signaled."""
    blocker = asyncio.Event()
    mcp = FastMCP("wait-state-test")

    @mcp.tool(task=True)
    async def blocked_tool() -> str:
        await blocker.wait()
        return "done"

    yield mcp, blocker
    blocker.set()


def _synthetic_status(task_id: str, status: mcp.types.TaskStatus) -> GetTaskResult:
    now = datetime.now(timezone.utc)
    return GetTaskResult(
        taskId=task_id,
        status=status,
        createdAt=now,
        lastUpdatedAt=now,
        ttl=None,
    )


class TestWaitStopStates:
    """wait() should return for any state that isn't 'working'."""

    async def test_wait_returns_on_input_required(self, blocked_tool_server):
        mcp, blocker = blocked_tool_server

        async with Client(mcp) as client:
            task = await client.call_tool("blocked_tool", {}, task=True)

            task._status_cache = _synthetic_status(task.task_id, "input_required")

            status = await task.wait(timeout=1.0)
            assert status.status == "input_required"

            blocker.set()

    async def test_wait_returns_on_cancelled(self, blocked_tool_server):
        mcp, blocker = blocked_tool_server

        async with Client(mcp) as client:
            task = await client.call_tool("blocked_tool", {}, task=True)

            task._status_cache = _synthetic_status(task.task_id, "cancelled")

            status = await task.wait(timeout=1.0)
            assert status.status == "cancelled"

            blocker.set()

    async def test_wait_returns_on_completed(self, blocked_tool_server):
        mcp, blocker = blocked_tool_server

        async with Client(mcp) as client:
            task = await client.call_tool("blocked_tool", {}, task=True)

            task._status_cache = _synthetic_status(task.task_id, "completed")

            status = await task.wait(timeout=1.0)
            assert status.status == "completed"

            blocker.set()

    async def test_wait_returns_on_failed(self, blocked_tool_server):
        mcp, blocker = blocked_tool_server

        async with Client(mcp) as client:
            task = await client.call_tool("blocked_tool", {}, task=True)

            task._status_cache = _synthetic_status(task.task_id, "failed")

            status = await task.wait(timeout=1.0)
            assert status.status == "failed"

            blocker.set()

    async def test_wait_does_not_return_on_working(self, blocked_tool_server):
        mcp, blocker = blocked_tool_server

        async with Client(mcp) as client:
            task = await client.call_tool("blocked_tool", {}, task=True)

            task._status_cache = _synthetic_status(task.task_id, "working")

            with pytest.raises(TimeoutError):
                await task.wait(timeout=0.3)

            blocker.set()


class TestResultGuard:
    """result() should raise when the task stopped in a non-completable state."""

    async def test_result_raises_on_input_required(self, blocked_tool_server):
        mcp, blocker = blocked_tool_server

        async with Client(mcp) as client:
            task = await client.call_tool("blocked_tool", {}, task=True)

            task._status_cache = _synthetic_status(task.task_id, "input_required")

            with pytest.raises(RuntimeError, match="cannot return a result"):
                await task.result()

            blocker.set()

    async def test_result_succeeds_on_completed(self, blocked_tool_server):
        mcp, blocker = blocked_tool_server

        async with Client(mcp) as client:
            blocker.set()
            task = await client.call_tool("blocked_tool", {}, task=True)

            result = await task.result()
            assert result is not None

    async def test_result_raises_on_cancelled(self, blocked_tool_server):
        mcp, blocker = blocked_tool_server

        async with Client(mcp) as client:
            task = await client.call_tool("blocked_tool", {}, task=True)

            task._status_cache = _synthetic_status(task.task_id, "cancelled")

            with pytest.raises(RuntimeError, match="cannot return a result"):
                await task.result()

            blocker.set()
