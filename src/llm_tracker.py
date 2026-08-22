"""
LLM 调用追踪器（单例）。
统一包装所有 OpenAI/DashScope API 调用，记录耗时/次数/Token。
设计为 6 阶段：init / short_reflection / crossover_gen / evaluation / long_reflection / mutation
"""
import csv
import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI


VALID_PHASES = {
    "init", "short_reflection", "crossover_gen",
    "evaluation", "long_reflection", "mutation",
    "other",  # 兜底
}


class LLMTracker:
    """单例追踪器，线程安全"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_state()
        return cls._instance

    @classmethod
    def get_instance(cls) -> "LLMTracker":
        return cls()

    def _init_state(self):
        self.reset()

    def reset(self):
        """重置所有计数器（每次新进化前调用）"""
        self.evolution_id: Optional[str] = None
        self.model: Optional[str] = None
        self.started_at: Optional[str] = None
        self.ended_at: Optional[str] = None
        self._start_ts: Optional[float] = None
        self._end_ts: Optional[float] = None
        self.phases: Dict[str, Dict[str, float]] = {}
        self.call_details: List[Dict[str, Any]] = []
        self.total_calls: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self.failures: Dict[str, int] = {"count": 0, "retried": 0, "aborted": 0}
        self._current_phase: Optional[str] = None

    def start_evolution(self, evolution_id: str, model: str):
        self.reset()
        self.evolution_id = evolution_id
        self.model = model
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self._start_ts = time.perf_counter()

    def end_evolution(self):
        self._end_ts = time.perf_counter()
        self.ended_at = datetime.now().isoformat(timespec="seconds")

    @property
    def total_duration_sec(self) -> float:
        if self._start_ts is None:
            return 0.0
        end = self._end_ts if self._end_ts else time.perf_counter()
        return end - self._start_ts

    @contextmanager
    def phase(self, name: str):
        """阶段上下文管理器"""
        if name not in VALID_PHASES:
            name = "other"
        prev = self._current_phase
        self._current_phase = name
        if name not in self.phases:
            self.phases[name] = {"calls": 0, "duration_sec": 0.0, "tokens": 0}
        phase_start = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name]["duration_sec"] += time.perf_counter() - phase_start
            self._current_phase = prev

    def _record_call(
        self,
        phase: str,
        success: bool,
        duration: float,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        error: Optional[str],
    ):
        if phase not in VALID_PHASES:
            phase = "other"
        if phase not in self.phases:
            self.phases[phase] = {"calls": 0, "duration_sec": 0.0, "tokens": 0}

        self.phases[phase]["calls"] += 1
        # 累加每次 API 调用耗时到 phase 总耗时
        self.phases[phase]["duration_sec"] += duration
        self.phases[phase]["tokens"] += total_tokens
        self.total_calls += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        if not success:
            self.failures["count"] += 1

        self.call_details.append({
            "call_id": len(self.call_details) + 1,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "phase": phase,
            "model": self.model or "",
            "success": success,
            "duration_sec": round(duration, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "error": error or "",
        })

    def export_json(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "llm_usage.json")
        payload = {
            "evolution_id": self.evolution_id,
            "model": self.model,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_duration_sec": round(self.total_duration_sec, 2),
            "total_api_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "phases": {
                k: {kk: round(vv, 2) if isinstance(vv, float) else vv
                    for kk, vv in v.items()}
                for k, v in self.phases.items()
            },
            "failures": self.failures,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    def export_csv(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "llm_calls_detail.csv")
        if not self.call_details:
            return path
        fields = list(self.call_details[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self.call_details)
        return path


def tracked_chat_create(client, phase: str, **kwargs):
    """
    包装 client.chat.completions.create()。
    自动记录耗时、Token、调用次数。
    支持同步与异步（async）调用：当 client 是 AsyncOpenAI 时返回 awaitable。

    注意：OpenAI SDK 2.x 对 AsyncCompletions.create 做了装饰器包装，
    inspect.iscoroutinefunction() 无法可靠识别，因此改用 isinstance 判断客户端类型。
    """
    tracker = LLMTracker.get_instance()

    # 通过客户端类型判断同步/异步，避免 iscoroutinefunction 的误判
    is_async = isinstance(client, AsyncOpenAI)

    if is_async:
        async def _wrap():
            t0 = time.perf_counter()
            error = None
            try:
                resp = await client.chat.completions.create(**kwargs)
                usage = getattr(resp, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
                success = True
            except Exception as e:
                resp = None
                prompt_tokens = completion_tokens = total_tokens = 0
                success = False
                error = f"{type(e).__name__}: {e}"
            duration = time.perf_counter() - t0
            tracker._record_call(
                phase=phase, success=success, duration=duration,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                total_tokens=total_tokens, error=error,
            )
            if not success:
                raise RuntimeError(f"LLM API call failed: {error}")
            return resp
        return _wrap()

    # 同步路径
    t0 = time.perf_counter()
    error = None
    try:
        response = client.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
        success = True
    except Exception as e:
        response = None
        prompt_tokens = completion_tokens = total_tokens = 0
        success = False
        error = f"{type(e).__name__}: {e}"

    duration = time.perf_counter() - t0
    tracker._record_call(
        phase=phase, success=success, duration=duration,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        total_tokens=total_tokens, error=error,
    )
    if not success:
        raise RuntimeError(f"LLM API call failed: {error}")
    return response
