"""Hermetic start_ipc_server / open_ipc_connection round-trip -- the socket
proof CI must execute on every OS. POSIX drives the AF_UNIX branch;
windows-latest drives the real TCP-loopback + token-handshake branch through
the same public entry points (no internal auth plumbing touched directly)."""
from __future__ import annotations

import asyncio

import pytest

from iai_mcp import _ipc


def test_ipc_server_roundtrip_echoes_request(monkeypatch, short_socket):
    # short_socket (tests/conftest.py) keeps the AF_UNIX path under the
    # platform sun_path limit -- pytest's own tmp_path is too deep on macOS.
    sock_path = str(short_socket)
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", sock_path)

    async def _echo_handler(reader, writer):
        line = await reader.readline()
        writer.write(line)
        await writer.drain()
        writer.close()

    async def scenario() -> bytes:
        server, actual_addr, _needs_cleanup = await _ipc.start_ipc_server(
            _echo_handler, sock_path
        )
        try:
            reader, writer = await _ipc.open_ipc_connection(sock_path, timeout=5)
            writer.write(b"ping\n")
            await writer.drain()
            reply = await asyncio.wait_for(reader.readline(), timeout=5)
            writer.close()
            return reply
        finally:
            server.close()
            await server.wait_closed()
            _ipc.shutdown_ipc(actual_addr)

    reply = asyncio.run(scenario())
    assert reply == b"ping\n"


def test_wrong_token_rejected_without_reaching_handler(monkeypatch, short_socket):
    """Windows-only: a bad first-line token must close the connection before
    the wrapped handler runs. POSIX has no token layer -- self-skips."""
    if not _ipc.IS_WINDOWS:
        pytest.skip("token auth gate is Windows-only")

    sock_path = str(short_socket)
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", sock_path)

    served: list[str] = []

    async def _echo_handler(reader, writer):
        line = await reader.readline()
        served.append(line.decode().strip())
        writer.write(line)
        await writer.drain()
        writer.close()

    async def scenario() -> bytes:
        server, actual_addr, _needs_cleanup = await _ipc.start_ipc_server(
            _echo_handler, sock_path
        )
        try:
            host, port = actual_addr
            reader, writer = await asyncio.open_connection(host, port)
            wrong_token = "0" * 64
            writer.write((wrong_token + "\n").encode())
            writer.write(b"stealth\n")
            await writer.drain()
            data = await asyncio.wait_for(reader.read(), timeout=5)
            writer.close()
            return data
        finally:
            server.close()
            await server.wait_closed()
            _ipc.shutdown_ipc(actual_addr)

    rejected = asyncio.run(scenario())
    assert rejected == b""
    assert served == [], "handler must never run on a wrong-token connection"


@pytest.mark.parametrize("windows", [False, True])
def test_ipc_reply_line_limit(monkeypatch, windows):
    import json
    import tempfile
    from pathlib import Path

    if not windows and _ipc.IS_WINDOWS:
        pytest.skip("Unix sockets unavailable on Windows")
    monkeypatch.setattr(_ipc, "IS_WINDOWS", windows)
    cap = 64 * 1024 * 1024
    reply = {"jsonrpc": "2.0", "id": 1, "result": {"hits": [
        {"literal_surface": "recall evidence " * 4800},
    ]}}
    large = json.dumps(reply).encode() + b"\n"
    assert 65536 < len(large) < cap
    responses = [b'{"result": "ok"}\n', large, b"x" * (cap + 1) + b"\n"]

    async def scenario(sock_path):
        async def handler(reader, writer):
            index = int(await reader.readline())
            writer.write(responses[index])
            try:
                await writer.drain()
            except (ConnectionError, OSError):
                pass
            finally:
                writer.close()
                await writer.wait_closed()

        server, addr, _ = await _ipc.start_ipc_server(handler, sock_path)
        try:
            for index, expected in enumerate(responses):
                reader, writer = await _ipc.open_ipc_connection(sock_path, timeout=5)
                try:
                    writer.write(f"{index}\n".encode())
                    await writer.drain()
                    if index == 2:
                        with pytest.raises(ValueError, match="limit"):
                            await asyncio.wait_for(reader.readline(), timeout=5)
                    else:
                        actual = await asyncio.wait_for(reader.readline(), timeout=5)
                        assert json.loads(actual) == json.loads(expected)
                finally:
                    writer.close()
                    await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()
            _ipc.shutdown_ipc(addr)

    with tempfile.TemporaryDirectory(prefix="rpc-") as directory:
        sock_path = str(Path(directory) / "s")
        monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", sock_path)
        asyncio.run(scenario(sock_path))
