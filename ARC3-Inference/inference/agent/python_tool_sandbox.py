"""Lightweight isolated runner for analyzer Python tool calls."""
from __future__ import annotations

import inspect
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import textwrap
import time
from typing import Any, Callable

from inference.utils import segmentation as _segmentation
from inference.utils.grid_utils import ARC_COLOR_CHARS


_SANDBOX_BOOTSTRAP = textwrap.dedent(
    r"""
    import builtins
    import contextlib
    import io
    import json
    import os
    import sys
    import traceback

    try:
        import resource
    except ImportError:  # pragma: no cover
        resource = None

    COLOR_CHARS = ""

    __SEGMENTATION_SOURCE__

    HOST_STDOUT = sys.stdout

    SAFE_MODULES = {
        "bisect",
        "collections",
        "copy",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
    }
    SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "Exception",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "TypeError",
        "type",
        "ValueError",
        "RuntimeError",
        "zip",
    }


    def _send(payload):
        HOST_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
        HOST_STDOUT.flush()


    def _recv():
        line = sys.stdin.readline()
        if not line:
            raise EOFError("sandbox input closed")
        return json.loads(line)


    class FrameView:
        def __init__(self, *, ascii, step, level, shape, grid):
            self.ascii = ascii
            self.step = step
            self.level = level
            self.shape = tuple(shape)
            self._grid = grid
            self._segmentation = None

        @property
        def segmentation(self):
            if self._segmentation is None:
                self._segmentation = segment_layer(self._grid, COLOR_CHARS)
            return self._segmentation

        def __str__(self):
            rows, cols = self.shape
            return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

        __repr__ = __str__


    class HistoryEntryView:
        def __init__(self, *, action, frame):
            self.action = action
            self.frame = frame

        def __str__(self):
            return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"

        __repr__ = __str__


    class TransitionView:
        def __init__(self, *, action, before_frame, after_frame, result):
            self.action = action
            self.before_frame = before_frame
            self.after_frame = after_frame
            self.frame = after_frame
            self.result = dict(result) if isinstance(result, dict) else {}

        def __str__(self):
            return (
                "ActionTransitionView("
                f"action={self.action!r}, "
                f"before_frame={self.before_frame}, "
                f"after_frame={self.after_frame})"
            )

        __repr__ = __str__


    def _frame_from_payload(payload):
        if not isinstance(payload, dict):
            return None
        return FrameView(
            ascii=str(payload.get("ascii", "")),
            step=int(payload.get("step", 0)),
            level=int(payload.get("level", 0)),
            shape=payload.get("shape", [0, 0]),
            grid=payload.get("grid", []),
        )


    def _history_from_payload(payload):
        items = []
        for entry in payload or []:
            if not isinstance(entry, dict):
                continue
            items.append(
                HistoryEntryView(
                    action=str(entry.get("action", "")),
                    frame=_frame_from_payload(entry.get("frame")),
                )
            )
        return items


    def _transitions_from_history(history, last_action_result):
        transitions = []
        for index, entry in enumerate(history):
            action = str(getattr(entry, "action", "") or "").strip()
            if not action:
                continue
            before_frame = history[index - 1].frame if index > 0 else None
            transitions.append(
                TransitionView(
                    action=action,
                    before_frame=before_frame,
                    after_frame=entry.frame,
                    result={},
                )
            )
        if transitions and isinstance(last_action_result, dict):
            transitions[-1].result = dict(last_action_result)
        return transitions


    def _json_safe(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return str(value)


    def _sanitize_exception(exc):
        extracted = traceback.extract_tb(exc.__traceback__)
        user_frames = [frame for frame in extracted if frame.filename == "<python_tool>"]
        lines = ["Traceback (most recent call last):"]
        for frame in user_frames or extracted[-1:]:
            lines.append(f'  File "<python_tool>", line {frame.lineno}, in {frame.name}')
        lines.append(f"{exc.__class__.__name__}: {exc}")
        return "\n".join(lines)


    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name or "").split(".", 1)[0]
        if root not in SAFE_MODULES:
            raise ImportError(f"Module '{name}' is not allowed in the sandbox.")
        return builtins.__import__(name, globals, locals, fromlist, level)


    def _set_limits(timeout_seconds):
        if resource is None:
            return
        cpu_limit = max(1, int(timeout_seconds)) + 1
        for limit, value in (
            (getattr(resource, "RLIMIT_CPU", None), cpu_limit),
            (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),
            (getattr(resource, "RLIMIT_NOFILE", None), 32),
        ):
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (value, value))
            except (OSError, ValueError):
                pass


    def _normalize_actions(actions):
        if isinstance(actions, str):
            items = [actions]
        elif isinstance(actions, dict):
            items = [actions]
        elif isinstance(actions, (list, tuple)):
            items = list(actions)
        else:
            raise TypeError(
                "action(actions) expects a string, an action object, or a list of action strings/objects."
            )
        if not items:
            raise ValueError("action(actions) requires at least one action.")

        normalized = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                action_name = item.strip()
                if not action_name:
                    raise ValueError(f"Action {index} is empty.")
                normalized.append({"action": action_name})
                continue
            if isinstance(item, dict):
                action_name = str(item.get("action", "")).strip()
                if not action_name:
                    raise ValueError(f"Action {index} is missing an `action` field.")
                entry = {"action": action_name}
                if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                    raise ValueError(
                        f"Action {index} uses legacy MOUSE x/y fields; use row and col."
                    )
                if "row" in item:
                    entry["row"] = item.get("row")
                if "col" in item:
                    entry["col"] = item.get("col")
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        return normalized


    def main():
        initial = _recv()
        global COLOR_CHARS
        COLOR_CHARS = str(initial.get("color_chars") or "")
        timeout_seconds = max(1, int(initial.get("timeout_seconds", 30)))
        sandbox_cwd = str(initial.get("sandbox_cwd", "")).strip()
        if sandbox_cwd:
            os.chdir(sandbox_cwd)
        _set_limits(timeout_seconds)

        action_results = []
        new_notes = []
        verdicts = []
        motif_lookups = []
        remembered_edits_holder = {}
        stdout = io.StringIO()
        runtime_globals = {
            "__builtins__": {
                name: getattr(builtins, name)
                for name in SAFE_BUILTINS
            },
            "result": None,
        }
        runtime_globals["__builtins__"]["__import__"] = _safe_import

        def _refresh_state(state_payload):
            current_frame = _frame_from_payload(state_payload.get("current_frame"))
            history = _history_from_payload(state_payload.get("history"))
            last_action_result = state_payload.get("last_action_result")
            action_result = (
                dict(last_action_result) if isinstance(last_action_result, dict) else {}
            )
            transitions = _transitions_from_history(history, action_result)
            last_transition = transitions[-1] if transitions else None

            runtime_globals["current_frame"] = current_frame
            runtime_globals["latest_frame"] = current_frame
            runtime_globals["history"] = history
            runtime_globals["transitions"] = transitions
            runtime_globals["last_transition"] = last_transition
            runtime_globals["previous_frame"] = (
                last_transition.before_frame if last_transition is not None else None
            )
            runtime_globals["last_action_frame"] = (
                last_transition.after_frame if last_transition is not None else None
            )
            runtime_globals["last_action"] = last_transition.action if last_transition is not None else None
            runtime_globals["valid_actions"] = [str(item) for item in state_payload.get("valid_actions", [])]
            runtime_globals["last_action_result"] = action_result
            # The level's own action count, which is the number the score is
            # computed from. `current_frame.step` is the whole game and says
            # nothing about how much of *this* level's budget is gone.
            level_now = getattr(current_frame, "level", None)
            runtime_globals["actions_this_level"] = (
                sum(
                    1
                    for entry in history
                    if getattr(entry, "action", "")
                    and getattr(entry.frame, "level", None) == level_now
                )
                if level_now is not None
                else 0
            )

        # What each action has actually done, counted rather than recalled.
        #
        # The strongest result on this benchmark so far came from a CNN that
        # learned nothing except which actions change the frame, and it beat
        # reasoning agents by an order of magnitude. This hands that signal
        # over directly: a button that has never moved anything is the
        # cheapest thing to stop paying for, and the score squares every press.
        #
        # `changed` counts transitions that altered any cell; `median_cells`
        # separates a real move from a HUD tick, which also reports as changed
        # but is one or two cells every single time.
        def action_effects(level=None, window=None):
            # Read through runtime_globals: current_frame and transitions are
            # locals of _refresh_state, rebuilt every time the state changes,
            # so closing over them would pin the values from before the first
            # action(...) call.
            frame_now = runtime_globals.get("current_frame")
            wanted = frame_now.level if level is None and frame_now is not None else level
            rows = {}
            selected = [
                t for t in (runtime_globals.get("transitions") or [])
                if t.before_frame is not None and t.after_frame is not None
                and (wanted is None or t.after_frame.level == wanted)
            ]
            if window:
                selected = selected[-int(window):]
            for t in selected:
                before, after = t.before_frame._grid, t.after_frame._grid
                changed = 0
                for r in range(min(len(before), len(after))):
                    row_a, row_b = before[r], after[r]
                    for c in range(min(len(row_a), len(row_b))):
                        if row_a[c] != row_b[c]:
                            changed += 1
                name = str(t.action).split("(")[0].strip() or "?"
                entry = rows.setdefault(name, {"tried": 0, "changed": 0, "_cells": []})
                entry["tried"] += 1
                if changed:
                    entry["changed"] += 1
                    entry["_cells"].append(changed)
            for entry in rows.values():
                cells = sorted(entry.pop("_cells"))
                entry["median_cells"] = cells[len(cells) // 2] if cells else 0
                entry["max_cells"] = cells[-1] if cells else 0
            return rows

        runtime_globals["action_effects"] = action_effects

        def action(actions):
            normalized_actions = _normalize_actions(actions)
            _send({"type": "action", "actions": normalized_actions})
            reply = _recv()
            if reply.get("type") == "action_error":
                raise RuntimeError(str(reply.get("error", "action failed")))
            if reply.get("type") != "action_result":
                raise RuntimeError("Invalid action response from sandbox host.")
            action_result = reply.get("action_result") or {}
            action_results.append(action_result)
            _refresh_state(reply.get("state") or {})
            return action_result

        runtime_globals["action"] = action

        # Append-only findings log. `notes` survives across tool calls (the host
        # carries it), so anything written here outlives the conversation
        # trimming that drops older turns.
        notes_log = list((initial.get("state") or {}).get("notes") or [])

        def note(text):
            entry = str(text)
            notes_log.append(entry)
            new_notes.append(entry)
            return entry

        runtime_globals["note"] = note
        runtime_globals["notes"] = notes_log

        # Framing hypotheses are written by the harness and can only be killed
        # or confirmed from here, always with evidence. A frame nobody can
        # refute is a belief, and a wrong belief costs actions.
        hypotheses_view = list((initial.get("state") or {}).get("hypotheses") or [])

        def _verdict(kind):
            def record(name, evidence):
                entry = {
                    "name": str(name),
                    "evidence": str(evidence),
                    "verdict": kind,
                }
                verdicts.append(entry)
                return entry
            return record

        # Only when framing actually produced something. With framing off the
        # parent omits the key entirely, and offering an empty `hypotheses`
        # with kill/confirm alongside it advertises a mechanism that can never
        # do anything -- which is worse than not mentioning it.
        if (initial.get("state") or {}).get("hypotheses") is not None:
            runtime_globals["hypotheses"] = hypotheses_view
            runtime_globals["kill"] = _verdict("dead")
            runtime_globals["confirm"] = _verdict("alive")

        # The motif catalog rides in the state rather than the prompt: the
        # names are in the system prompt, the entries are here, and only what
        # gets printed costs context. Lookups are recorded so a run can say
        # whether the vocabulary was used at all -- an unused feature and a
        # harmful one look identical in the score.
        motif_catalog = list((initial.get("state") or {}).get("motif_catalog") or [])
        if motif_catalog:
            def _motifs():
                return [str(m.get("slug", "")) for m in motif_catalog if m.get("slug")]

            def _motif(name):
                wanted = str(name or "").strip().lower().replace("_", "-").replace(" ", "-")
                for entry in motif_catalog:
                    if str(entry.get("slug", "")).lower() != wanted:
                        continue
                    motif_lookups.append(wanted)
                    found = {k: v for k, v in entry.items() if k != "has_detail"}
                    if not entry.get("has_detail"):
                        found["note"] = (
                            "Summary only -- no detailed entry was written for this motif."
                        )
                    return found
                motif_lookups.append(f"{wanted}?")
                return {
                    "error": f"no motif named {wanted!r}",
                    "available": _motifs(),
                }

            runtime_globals["motifs"] = _motifs
            runtime_globals["motif"] = _motif

        # Code that outlives the turn. Every python call is a fresh
        # subprocess, so a world model written in one turn evaporates before
        # the next -- the model can only carry prose forward in notes and has
        # to re-derive its dynamics from scratch each time. Sources stored here
        # are re-executed at the top of every later call, which is what makes
        # an executable world model, and therefore planning against one,
        # possible at all.
        remembered_source = dict((initial.get("state") or {}).get("remembered_code") or {})
        remembered_edits = remembered_edits_holder
        remembered_errors = []
        if (initial.get("state") or {}).get("remembered_code") is not None:
            def remember(name, source):
                key = str(name).strip()
                if not key:
                    raise ValueError("remember() needs a non-empty name")
                remembered_source[key] = str(source)
                remembered_edits[key] = str(source)
                try:
                    exec(compile(str(source), f"<remembered:{key}>", "exec"), runtime_globals, runtime_globals)
                except Exception as exc:
                    # Kept anyway: the model asked for it, and silently
                    # dropping it would leave it calling a function that is
                    # not there with no idea why.
                    return {"stored": key, "error": f"{type(exc).__name__}: {exc}"}
                return {"stored": key}

            def forget(name):
                key = str(name).strip()
                remembered_source.pop(key, None)
                remembered_edits[key] = None
                return {"forgot": key}

            for key, source in list(remembered_source.items()):
                try:
                    exec(compile(str(source), f"<remembered:{key}>", "exec"), runtime_globals, runtime_globals)
                except Exception as exc:
                    remembered_errors.append(f"{key}: {type(exc).__name__}: {exc}")

            runtime_globals["remember"] = remember
            runtime_globals["forget"] = forget
            runtime_globals["remembered"] = sorted(remembered_source)
            runtime_globals["remembered_errors"] = remembered_errors

        _refresh_state(initial.get("state") or {})

        try:
            compiled = compile(str(initial.get("code", "")), "<python_tool>", "exec")
            with contextlib.redirect_stdout(stdout):
                exec(compiled, runtime_globals, runtime_globals)
            _send(
                {
                    "type": "final",
                    "stdout": stdout.getvalue(),
                    "result": _json_safe(runtime_globals.get("result")),
                    "action_results": _json_safe(action_results),
                    "notes": _json_safe(new_notes),
                    "verdicts": _json_safe(verdicts),
                    "motif_lookups": _json_safe(motif_lookups),
                    "remembered_edits": _json_safe(remembered_edits_holder),
                }
            )
        except Exception as exc:
            _send(
                {
                    "type": "error",
                    "error": _sanitize_exception(exc),
                    "stdout": stdout.getvalue(),
                    "action_results": _json_safe(action_results),
                    "notes": _json_safe(new_notes),
                    "verdicts": _json_safe(verdicts),
                    "motif_lookups": _json_safe(motif_lookups),
                    "remembered_edits": _json_safe(remembered_edits_holder),
                }
            )


    if __name__ == "__main__":
        main()
    """
).replace("__SEGMENTATION_SOURCE__\n", inspect.getsource(_segmentation))


def _sanitize_host_error_text(text: str) -> str:
    if not str(text or "").strip():
        return "Sandbox process exited unexpectedly."
    return "Sandbox process exited unexpectedly."


def _sandbox_env() -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PATH": os.environ.get("PATH", ""),
    }


def _send_json_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        try:
            process.kill()
        except OSError:
            pass


def _wait_for_process_exit(process: subprocess.Popen[str], *, timeout: float = 1.0) -> None:
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
    except OSError:
        return

    try:
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        pass


def run_sandboxed_python(
    *,
    code: str,
    timeout_seconds: int,
    initial_state: dict[str, Any],
    action_handler: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rgb_python_tool_") as sandbox_dir:
        host_action_results: list[dict[str, Any]] = []
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _SANDBOX_BOOTSTRAP],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                cwd=sandbox_dir,
                env=_sandbox_env(),
                start_new_session=True,
            )
        except OSError:
            return {
                "error": "Sandbox process could not start.",
                "stdout": "",
                "action_results": [],
            }
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        stdout_queue: queue.Queue[str | None] = queue.Queue()

        def _stdout_reader() -> None:
            for raw_line in process.stdout:
                stdout_queue.put(raw_line)
            stdout_queue.put(None)

        threading.Thread(target=_stdout_reader, daemon=True).start()

        _send_json_line(
            process.stdin,
            {
                "code": code,
                "timeout_seconds": timeout_seconds,
                "sandbox_cwd": sandbox_dir,
                "state": initial_state,
                "color_chars": ARC_COLOR_CHARS,
            },
        )

        deadline = time.monotonic() + max(1, int(timeout_seconds))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": f"Tool timed out after {timeout_seconds}s",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                line = stdout_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                stderr = process.stderr.read()
                _wait_for_process_exit(process)
                return {
                    "error": _sanitize_host_error_text(stderr),
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                stderr = process.stderr.read()
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": "Sandbox process returned an invalid response.",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            msg_type = str(message.get("type", "")).strip()
            if msg_type == "action":
                try:
                    action_result_payload = action_handler(list(message.get("actions") or []))
                except Exception:  # noqa: BLE001
                    _send_json_line(
                        process.stdin,
                        {
                            "type": "action_error",
                            "error": "action failed in sandbox host.",
                        },
                    )
                    continue
                raw_action_result = action_result_payload.get("action_result") or {}
                if isinstance(raw_action_result, dict):
                    host_action_results.append(dict(raw_action_result))
                _send_json_line(
                    process.stdin,
                    {
                        "type": "action_result",
                        "action_result": raw_action_result,
                        "state": action_result_payload.get("state") or {},
                    },
                )
                continue

            if msg_type in {"final", "error"}:
                _wait_for_process_exit(process)
                return {
                    "stdout": str(message.get("stdout", "") or ""),
                    "result": message.get("result"),
                    "error": str(message.get("error", "") or ""),
                    "action_results": list(message.get("action_results") or host_action_results),
                    "notes": [str(item) for item in (message.get("notes") or [])],
                    "verdicts": [
                        item for item in (message.get("verdicts") or [])
                        if isinstance(item, dict)
                    ],
                    "motif_lookups": [
                        str(item) for item in (message.get("motif_lookups") or [])
                    ],
                    "remembered_edits": (
                        message.get("remembered_edits")
                        if isinstance(message.get("remembered_edits"), dict)
                        else {}
                    ),
                }

            _wait_for_process_exit(process)
            return {
                "error": "Sandbox process returned an unknown message type.",
                "stdout": "",
                "action_results": list(host_action_results),
            }
