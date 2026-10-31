"""
llm.py

Everything that talks to an LLM:

  * LLM: one client for Anthropic (claude-*), OpenAI (gpt-*, o*) and DeepSeek (deepseek-*), with retries,
    a thread-safe spend tracker / budget cap (api_metrics.jsonl) and a thread pool (map)
  * Prompts: a setting's YAML prompt file (string.Template placeholders, validated at startup, lockable)
  * Blind: teacher-blind checks (static prompt audit, client-boundary abort/scrub, bank check)
  * TextGrad engine routed through LLM.call, the patched TGD-with-momentum optimizer, and a retrying step;
    textgrad is imported lazily because it creates a logs/ directory on import
"""

import hashlib
import json
import os
import random
import re
import string
import sys
import threading
import time
import warnings
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


# -------------------------------------------------------------------------
# Errors
# -------------------------------------------------------------------------
#
# BudgetExhausted, BlindViolation and LLMConfigError stop a run: LLM.map and optimizer_step re-raise them, and
# callers that catch Exception must re-raise ABORTS. None of them is a RuntimeError.

class BudgetExhausted(Exception):
    def __init__(self, max_cost_usd=None, spent_usd: Optional[float] = None):
        if isinstance(max_cost_usd, (int, float)) and spent_usd is not None:
            super().__init__(f"API budget exhausted: ${spent_usd:.6f} >= ${max_cost_usd:.6f}")
        else:   # BudgetExhausted("message")
            super().__init__(max_cost_usd or "API budget exhausted")
            max_cost_usd = None
        self.max_cost_usd = max_cost_usd
        self.spent_usd = spent_usd


class BlindViolation(Exception):
    """A forbidden (teacher-identity) token in a prompt or a bank (teacher-blind runs)."""


class LLMConfigError(Exception):
    """The API rejects the configuration (key, balance, model, parameters): every further call would fail too."""


class LLMCallError(Exception):
    """One call failed (non-transient error, empty or truncated answer, or transient errors past the retries)."""


class PromptError(ValueError):
    pass


ABORTS = (BudgetExhausted, BlindViolation, LLMConfigError)


# -------------------------------------------------------------------------
# Backends
# -------------------------------------------------------------------------
#
# Anthropic goes through the Messages API, OpenAI and DeepSeek through chat completions (DeepSeek: OpenAI SDK on
# its own endpoint). Base URLs are explicit so that OPENAI_BASE_URL / ANTHROPIC_BASE_URL in the environment cannot
# redirect a run. SDK-internal retries are off: LLM.call does the retrying and the accounting.

PROVIDERS = {   # provider: (key variable, base URL)
    "anthropic": ("ANTHROPIC_API_KEY", "https://api.anthropic.com"),
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com"),
}
OPENAI_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
DEEPSEEK_EFFORTS = ("none", "low", "high", "max")   # the API's own levels ("none" = thinking off)
LLM_FIELDS = ("model", "reasoning_effort", "max_tokens", "max_workers", "max_cost_usd",
              "price_input", "price_output", "price_cache_read")


def detect_provider(model: str) -> str:
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("deepseek-"):
        return "deepseek"
    if model.startswith("gpt-") or re.match(r"o\d", model):
        return "openai"
    raise ValueError(f"unsupported model {model!r}: expected claude-*, gpt-*, o<N>* or deepseek-*")


def is_openai_reasoning(model: str) -> bool:
    """o-series and gpt-5 or later (not the *-chat variants): reasoning_effort, no temperature."""
    m = re.match(r"gpt-(\d+)", model)
    return bool(re.match(r"o\d", model)) or (m is not None and int(m.group(1)) >= 5 and "-chat" not in model)


def resolve_api_key(provider: str) -> str:
    env = PROVIDERS[provider][0]
    value = (os.environ.get(env) or "").strip()
    if not value or not value.isascii() or value in {"yourkey", "your-key"}:
        raise ValueError(f"no valid API key for {provider}: set {env}")
    return value


def make_client(provider: str, api_key: str, timeout_s: float):
    base_url = PROVIDERS[provider][1]
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=api_key, base_url=base_url, max_retries=0, timeout=timeout_s)
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=timeout_s)


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_blocks(text: str) -> str:
    """Remove inline <think>...</think> reasoning. An unterminated <think> means the answer was cut off inside
    the reasoning: return the empty string."""
    if "<think>" not in text:
        return text
    text = _THINK_BLOCK_RE.sub("", text)
    return "" if "<think>" in text else text


def _class_names(e: BaseException) -> set:
    return {c.__name__ for c in type(e).__mro__}


def _is_api_error(e: BaseException) -> bool:
    return bool(_class_names(e) & {"OpenAIError", "AnthropicError"}) or isinstance(e, (ConnectionError, TimeoutError))


def _is_transient(e: BaseException) -> bool:
    """Rate limit, timeout, connection error, overload / 5xx."""
    if _class_names(e) & {"APIConnectionError", "APITimeoutError"} or isinstance(e, (ConnectionError, TimeoutError)):
        return True
    status = getattr(e, "status_code", None)
    return isinstance(status, int) and (status in (408, 409, 429) or status >= 500)


def _retry_after(e: BaseException) -> Optional[float]:
    headers = getattr(getattr(e, "response", None), "headers", None)
    try:
        return float(headers.get("retry-after")) if headers is not None and headers.get("retry-after") else None
    except (TypeError, ValueError):
        return None


def _usage_tokens(usage) -> Tuple[int, int, int, int]:
    """(uncached prompt, completion, cache read, cache write) tokens of a response's usage.

    OpenAI reports prompt_tokens / completion_tokens (prompt includes the cached tokens in
    prompt_tokens_details.cached_tokens); DeepSeek adds prompt_cache_hit_tokens; Anthropic reports input_tokens
    (uncached only) plus cache_read_input_tokens / cache_creation_input_tokens.
    """
    prompt = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0)
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    if cache_read == 0 and cache_write == 0:
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
        cached = cached or int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
        if cached:
            prompt, cache_read = max(prompt - cached, 0), cached
    return prompt, completion, cache_read, cache_write


def _field(section, name, default=None):
    return section.get(name, default) if isinstance(section, Mapping) else getattr(section, name, default)


# -------------------------------------------------------------------------
# LLM client with spend tracking
# -------------------------------------------------------------------------

class LLM:
    """LLM client built from cfg.llm (or the llm section itself); every call is budget-checked and logged.

    Prices are USD per 1M tokens: price_cache_read defaults to 0.1 x price_input, Anthropic cache writes are
    priced at 1.25 x price_input. With max_cost_usd set, price_input and price_output are required.

    max_cost_usd is a soft cap: a request starts only while the spend is below it (checked before every attempt),
    and its cost is charged when the response arrives. Requests already in flight when the cap is crossed still
    complete and are charged, so the final spend can exceed the cap by the cost of up to max_workers concurrent
    requests (the next call then raises BudgetExhausted).
    """
    retries = 6              # retries of a transient failure (exponential backoff with jitter)
    backoff_base = 2.0       # seconds
    backoff_cap = 60.0
    timeout_s = 600.0        # per request
    gate_wait_s = 300.0      # Anthropic: how long parallel calls wait for the first writer of a cached prefix

    def __init__(self, cfg, log_path: Optional[str] = None, blind: Optional["Blind"] = None, client=None):
        sec = _field(cfg, "llm", None) or cfg
        known = set(sec) if isinstance(sec, Mapping) else set(vars(sec)) if hasattr(sec, "__dict__") else set()
        unknown = sorted(k for k in known - set(LLM_FIELDS) if not str(k).startswith("_"))
        if unknown:
            raise ValueError(f"unknown llm option(s) {unknown}; known: {list(LLM_FIELDS)}")
        self.model = _field(sec, "model")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("llm.model is required")
        self.provider = detect_provider(self.model)
        self.reasoning_effort = _field(sec, "reasoning_effort")
        workers = _field(sec, "max_workers")
        self.max_workers = 4 if workers is None else int(workers)

        def _num(name):
            v = _field(sec, name)
            return None if v is None else float(v)
        self.max_cost_usd = _num("max_cost_usd")
        self.price_input, self.price_output = _num("price_input"), _num("price_output")
        self.price_cache_read = _num("price_cache_read")
        self.log_path = log_path
        self.blind = blind

        effort, reasoning = self.reasoning_effort, False
        if self.provider == "anthropic":
            if effort is not None:
                raise ValueError(f"reasoning_effort is not supported for {self.model} (Anthropic); remove it")
        elif self.provider == "openai":
            reasoning = is_openai_reasoning(self.model)
            if effort is not None and not reasoning:
                raise ValueError(f"reasoning_effort is not supported for {self.model} (not a reasoning model)")
            if effort is not None and effort not in OPENAI_EFFORTS:
                raise ValueError(f"reasoning_effort for OpenAI must be one of {OPENAI_EFFORTS}, got {effort!r}")
        elif effort is not None and effort not in DEEPSEEK_EFFORTS:
            raise ValueError(f"reasoning_effort for DeepSeek must be one of {DEEPSEEK_EFFORTS}, got {effort!r}")
        # thinking: the call produces reasoning tokens (provider default when effort is unset)
        self.thinking = (reasoning or self.provider == "deepseek") and effort != "none"
        max_tokens = _field(sec, "max_tokens")
        self.max_tokens = int(max_tokens) if max_tokens is not None else (16384 if self.thinking else 4096)
        if self.max_tokens <= 0 or self.max_workers < 1:
            raise ValueError("llm.max_tokens and llm.max_workers must be >= 1")

        # request options fixed per run: OpenAI counts reasoning in max_completion_tokens, DeepSeek in max_tokens
        self._options: Dict = {}
        if self.provider == "openai":
            self._options["max_completion_tokens"] = self.max_tokens
            if effort is not None:
                self._options["reasoning_effort"] = effort
        elif self.provider == "deepseek":
            self._options["max_tokens"] = self.max_tokens
            if effort == "none":
                self._options["extra_body"] = {"thinking": {"type": "disabled"}}
            elif effort is not None:
                self._options["extra_body"] = {"thinking": {"type": "enabled"}}
                self._options["reasoning_effort"] = effort
        # OpenAI reasoning models reject temperature; DeepSeek ignores it while thinking
        self.send_temperature = not (reasoning or (self.provider == "deepseek" and self.thinking))
        if not self.send_temperature:
            warnings.warn(f"{self.model} with reasoning on does not take a temperature: calls use the model's "
                          f"default sampling", stacklevel=2)

        if self.max_cost_usd is not None:
            if self.price_input is None or self.price_output is None:
                raise ValueError("llm.max_cost_usd requires llm.price_input and llm.price_output (USD per 1M tokens)")
            if self.max_cost_usd <= 0:
                raise ValueError("llm.max_cost_usd must be > 0")
        for p in (self.price_input, self.price_output, self.price_cache_read):
            if p is not None and p < 0:
                raise ValueError("llm prices must be >= 0")

        self.client = client if client is not None else make_client(
            self.provider, resolve_api_key(self.provider), self.timeout_s)
        self.spent = 0.0
        self.input_tokens = 0            # total prompt tokens, cached or not
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.request_count = 0
        self._ok_once = False            # a call has succeeded: the configuration is accepted by the API
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(self.max_workers)   # concurrent requests, also across nested maps
        self._gates: Dict[str, threading.Event] = {}

    # ---- accounting ---------------------------------------------------------

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self.spent

    def _ensure_budget(self) -> None:
        with self._lock:
            if self.max_cost_usd is not None and self.spent >= self.max_cost_usd:
                raise BudgetExhausted(self.max_cost_usd, self.spent)

    def cost(self, uncached: int, completion: int, cache_read: int = 0, cache_write: int = 0) -> float:
        usd = 0.0   # term by term, in the old tracker's order (bit-identical estimates)
        if self.price_input is not None:
            read = self.price_cache_read if self.price_cache_read is not None else 0.1 * self.price_input
            usd += uncached * self.price_input / 1_000_000.0
            usd += cache_read * read / 1_000_000.0
            usd += cache_write * (1.25 * self.price_input) / 1_000_000.0
        if self.price_output is not None:
            usd += completion * self.price_output / 1_000_000.0
        return usd

    def _record(self, source: str, usage, finish_reason) -> None:
        uncached, completion, cache_read, cache_write = _usage_tokens(usage)
        usd = self.cost(uncached, completion, cache_read, cache_write)
        with self._lock:
            self.request_count += 1
            self.input_tokens += uncached + cache_read + cache_write
            self.output_tokens += completion
            self.cache_read_tokens += cache_read
            self.cache_write_tokens += cache_write
            self.spent += usd
            if self.log_path:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "type": "api_call", "source": source, "model": self.model,
                        "prompt_tokens": uncached + cache_read + cache_write,
                        "uncached_prompt_tokens": uncached,
                        "cache_read_tokens": cache_read, "cache_write_tokens": cache_write,
                        "completion_tokens": completion,
                        "estimated_cost_usd": usd, "total_estimated_cost_usd": self.spent,
                        "finish_reason": finish_reason,
                    }, ensure_ascii=False) + "\n")

    def state_dict(self) -> Dict:
        with self._lock:
            return {"model": self.model, "spent_usd": self.spent, "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens, "cache_read_tokens": self.cache_read_tokens,
                    "cache_write_tokens": self.cache_write_tokens, "request_count": self.request_count}

    def load_state_dict(self, state: Dict) -> None:
        with self._lock:
            self.spent = float(state["spent_usd"])
            self.input_tokens = int(state.get("input_tokens", 0))
            self.output_tokens = int(state.get("output_tokens", 0))
            self.cache_read_tokens = int(state.get("cache_read_tokens", 0))
            self.cache_write_tokens = int(state.get("cache_write_tokens", 0))
            self.request_count = int(state.get("request_count", 0))

    # ---- calls --------------------------------------------------------------

    def _request(self, system: str, user: str, temperature: float):
        """One API request -> (answer text, finish reason, usage)."""
        if self.provider == "anthropic":
            kw = {"model": self.model, "max_tokens": self.max_tokens, "timeout": self.timeout_s,
                  "messages": [{"role": "user", "content": user}]}
            if system:   # cached prefix: repeated system prompts are read at the cache price
                kw["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if self.send_temperature:
                kw["temperature"] = temperature
            r = self.client.messages.create(**kw)
            text = "\n".join(b.text for b in (r.content or []) if getattr(b, "type", None) == "text")
            return text, r.stop_reason, getattr(r, "usage", None)
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        kw = {"model": self.model, "messages": messages, "timeout": self.timeout_s, **self._options}
        if self.send_temperature:
            kw["temperature"] = temperature
        r = self.client.chat.completions.create(**kw)
        # reasoning arrives in message.reasoning_content (DeepSeek) and is discarded
        choice = r.choices[0] if getattr(r, "choices", None) else None
        text = (choice.message.content or "") if choice is not None else ""
        return text, getattr(choice, "finish_reason", None), getattr(r, "usage", None)

    def _gate(self, system: str):
        """Anthropic first-writer gate: N parallel requests with a new identical system prompt would all pay the
        cache-write price; the first proceeds alone, the others wait for it and then read the cache.
        Returns the event to set when this call is the first writer, else None."""
        key = hashlib.sha1(f"{self.model}\0{system}".encode("utf-8")).hexdigest()
        with self._lock:
            gate = self._gates.get(key)
            if gate is None:
                gate = self._gates[key] = threading.Event()
                return gate
        gate.wait(timeout=self.gate_wait_s)
        return None

    def call(self, system: str, user: str, source: str, temperature: float = 0.7) -> str:
        """One completion: the answer text, stripped. Raises BudgetExhausted before the call when the budget is
        spent (a soft cap: calls already running when it is crossed are still charged), LLMConfigError when the API
        rejects the configuration, LLMCallError when the call fails."""
        if not isinstance(system, str) or not isinstance(user, str) or not user:
            raise ValueError("LLM.call needs a system string and a non-empty user string")
        self._ensure_budget()
        if self.blind is not None:
            system, user = self.blind.check(system, user, source)
        first = self._gate(system) if self.provider == "anthropic" and system else None
        try:
            return self._call_with_retries(system, user, source, temperature)
        finally:
            if first is not None:
                first.set()

    def _call_with_retries(self, system, user, source, temperature) -> str:
        last, delay = None, 0.0
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(delay)
            self._ensure_budget()
            try:
                with self._slots:
                    text, finish, usage = self._request(system, user, temperature)
            except Exception as e:
                if not _is_api_error(e):
                    raise
                status = getattr(e, "status_code", None)
                if status in (401, 402, 403) or getattr(e, "code", None) == "insufficient_quota" or (
                        status in (400, 404, 422) and not self._ok_once):
                    raise LLMConfigError(f"{self.model}: {type(e).__name__} ({status}): {e}") from e
                if not _is_transient(e):
                    raise LLMCallError(f"{source}: {type(e).__name__} ({status}): {e}") from e
                last = e
                delay = min(self.backoff_cap, self.backoff_base * 2 ** attempt) * (0.5 + 0.5 * random.random())
                delay = max(delay, min(_retry_after(e) or 0.0, 300.0))
            else:
                self._record(source, usage, finish)
                text = _strip_think_blocks(text).strip()
                if text:
                    self._ok_once = True
                    if finish in ("length", "max_tokens"):
                        print(f"  [llm] {source}: answer truncated at max_tokens={self.max_tokens}")
                    return text
                if finish in ("length", "max_tokens"):
                    raise LLMCallError(f"{source}: no answer within max_tokens={self.max_tokens} "
                                       f"(raise llm.max_tokens or lower reasoning_effort)")
                last = LLMCallError(f"empty answer (finish_reason={finish})")
                delay = min(self.backoff_cap, self.backoff_base * 2 ** attempt)
            if attempt < self.retries:
                print(f"  [llm] {source}: {type(last).__name__}: {str(last)[:160]} -- retry {attempt + 1}/"
                      f"{self.retries} in {delay:.1f}s")
        raise LLMCallError(f"{source}: failed after {self.retries + 1} attempts: {last}") from last

    def textgrad_engine(self, source: str = "textgrad"):
        """A textgrad EngineLM whose calls go through this client (imports textgrad)."""
        return textgrad_classes()[0](self, source=source)

    def map(self, fn: Callable, items: Iterable) -> List:
        """[fn(item) for item in items] on max_workers threads, in order. An exception of fn is returned in its
        item's place, except ABORTS (BudgetExhausted, ...): queued items are cancelled, running ones finish,
        and the first abort is re-raised."""
        items = list(items)
        results: List = [None] * len(items)
        if self.max_workers == 1 or len(items) <= 1:
            for i, item in enumerate(items):
                try:
                    results[i] = fn(item)
                except ABORTS:
                    raise
                except Exception as e:  # noqa: BLE001
                    results[i] = e
            return results
        abort = None
        pool = ThreadPoolExecutor(max_workers=min(self.max_workers, len(items)))
        try:
            futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
            for fut in as_completed(futures):
                try:
                    results[futures[fut]] = fut.result()
                except ABORTS as e:
                    abort = e
                    for f in futures:
                        f.cancel()
                    break
                except Exception as e:  # noqa: BLE001
                    results[futures[fut]] = e
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        if abort is not None:
            raise abort
        return results


# -------------------------------------------------------------------------
# Prompts
# -------------------------------------------------------------------------
#
# One YAML file per setting. Values: strings (string.Template: ${name}, $$ = literal $), lists of strings, or
# nested sections of those, addressed with dotted keys ("textgrad.code_role"). Every template sees the global
# variables (the setting's prompt_vars plus `context`, the rendered context entry) and its role variables below
# (the per-call values; a role variable overrides a global of the same name). "a.*" matches every entry below a
# (any depth). The budget entries (layout per setting, e.g. budget.template.<tier>) take Budget.prompt_vars(), as
# role variables or merged into the globals by the caller.

BUDGET_VARS = ("tier", "max_concepts", "max_lines", "max_branches", "feats_module", "raw_inputs", "primitives")
ROLE_VARS: Dict[str, frozenset] = {k: frozenset(v) for k, v in {
    "context": (), "idea_system": (), "idea_user": ("direction", "existing"), "seeds": (),
    "implement_system": (), "implement_user": ("strategy", "template"), "template": (), "fallback": (),
    "backward_code_system": (), "backward_code_user": ("code", "kl", "case"),
    "backward_desc_system": (), "backward_desc_user": ("description", "code", "kl", "case"),
    "optimizer_system": (), "describe_system": (), "describe_user": ("code",),
    "cluster_system": ("n_modes",), "cluster_user": ("n_modes", "frozen", "summary", "cases"),
    "add_implement_system": ("frozen", "demos"), "add_implement_user": ("frozen", "strategy", "template", "demos"),
    "retry_lowhit": ("best_hit",), "retry_reject": ("reason",),
    "textgrad.code_role": (), "textgrad.desc_role": (),
    "textgrad.code_constraints": (), "textgrad.desc_constraints": (),
    "budget.*": BUDGET_VARS + ("reason",),   # block, template, fallback, seeds, tg_reject, ... (+ per tier)
}.items()}


def role_vars_of(key: str, table: Optional[Dict] = None) -> Optional[frozenset]:
    """Role variables of an entry (exact key, else the closest "section.*" pattern); None for an unknown entry."""
    table = ROLE_VARS if table is None else table
    if key in table:
        return frozenset(table[key])
    parts = key.split(".")
    for i in range(len(parts) - 1, 0, -1):
        wild = ".".join(parts[:i]) + ".*"
        if wild in table:
            return frozenset(table[wild])
    return None


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate keys (YAML silently keeps the last one)."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep)


def _flatten(tree, prefix: str = "") -> Dict:
    out = {}
    for k, v in tree.items():
        if not isinstance(k, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            raise PromptError(f"bad prompt key {prefix + str(k)!r}")
        key = prefix + k
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, str) or (isinstance(v, list) and v and all(isinstance(s, str) for s in v)):
            out[key] = v
        else:   # e.g. a seed "name: text" parsed as a mapping, or an unquoted yes / 0.5
            raise PromptError(f"prompt entry {key!r} must be a string or a non-empty list of strings "
                              f"(got {type(v).__name__}; quote it)")
    return out


def _placeholders(text: str, key: str) -> set:
    names = set()
    for m in string.Template.pattern.finditer(text):
        if m.group("invalid") is not None:
            line = text.count("\n", 0, m.start("invalid")) + 1
            raise PromptError(f"{key}: invalid '$' on line {line} (write $$ for a literal $)")
        name = m.group("named") or m.group("braced")
        if name:
            names.add(name)
    return names


class Prompts:
    def __init__(self, path: str, variables: Dict):
        self.path = path
        with open(path, "rb") as f:
            self._raw = f.read()
        try:
            tree = yaml.load(self._raw.decode("utf-8"), Loader=_UniqueKeyLoader)
        except yaml.YAMLError as e:
            raise PromptError(f"{path}: {e}") from e
        if not isinstance(tree, dict):
            raise PromptError(f"{path}: top level must be a mapping")
        self.entries = _flatten(tree)
        if "context" in variables:
            raise PromptError("'context' is reserved for the rendered context entry")
        self.variables = {k: str(v) for k, v in variables.items()}
        if "context" in self.entries:
            self.variables["context"] = self._substitute("context", self.entries["context"], dict(self.variables))

    def apply_budget(self, budget) -> None:
        """Program-budget overlay (program_budget.tier != off), as the budget-tier runs used it: the rendered
        budget.block is appended to the context, template / fallback / seeds become the tier's, and
        budget.code_constraint is added to TextGrad's code constraints. The budget's prompt_vars() become globals."""
        e, tier = self.entries, budget.tier
        need = ["budget.block", "budget.fallback", "budget.code_constraint", "budget.tg_reject",
                f"budget.template.{tier}", f"budget.seeds.{tier}"]
        missing = [k for k in need if k not in e]
        if missing:
            raise PromptError(f"{self.path}: program budget '{tier}' needs the entries {missing}")
        self.variables.update({k: str(v) for k, v in budget.prompt_vars().items()})
        block = self._substitute("budget.block", e["budget.block"], self.variables)
        e["template"], e["fallback"] = e[f"budget.template.{tier}"], e["budget.fallback"]
        e["seeds"] = e[f"budget.seeds.{tier}"]
        e["textgrad.code_constraints"] = list(e.get("textgrad.code_constraints", [])) + [e["budget.code_constraint"]]
        if "context" in e:
            self.variables["context"] = self.variables["context"] + "\n\n" + block

    def keys(self) -> List[str]:
        return list(self.entries)

    def get(self, key: str, default=None):
        """The raw (unrendered) entry."""
        return self.entries.get(key, default)

    def render_text(self, text: str, **role_vars) -> str:
        """Substitute the global and role variables into an arbitrary template string."""
        return self._substitute("<text>", text, {**self.variables, **{k: str(v) for k, v in role_vars.items()}})

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    @staticmethod
    def _substitute(key: str, text: str, mapping: Dict) -> str:
        missing = _placeholders(text, key) - set(mapping)
        if missing:
            raise PromptError(f"prompt {key!r}: missing variable(s) {sorted(missing)}")
        return string.Template(text).substitute(mapping)

    def placeholders(self, key: str) -> set:
        v = self.entries[key]
        return set().union(*(_placeholders(s, key) for s in (v if isinstance(v, list) else [v])))

    def render(self, key: str, **role_vars):
        """The entry with global and role variables substituted (a list entry -> list of strings)."""
        if key not in self.entries:
            kind = "a section, address its entries as key.name" if any(
                k.startswith(key + ".") for k in self.entries) else "missing"
            raise PromptError(f"prompt {key!r} is {kind} ({self.path})")
        mapping = {**self.variables, **{k: str(v) for k, v in role_vars.items()}}   # unused extras are ignored
        v = self.entries[key]
        if isinstance(v, list):
            return [self._substitute(f"{key}[{i}]", s, mapping) for i, s in enumerate(v)]
        return self._substitute(key, v, mapping)

    def validate(self, role_vars: Optional[Dict[str, Iterable[str]]] = None,
                 required: Sequence[str] = ()) -> None:
        """Startup check: every entry is known, every placeholder is a global or role variable of its entry,
        `required` entries exist, and optimizer_system survives textgrad's str.format. Raises PromptError listing
        all problems."""
        problems = [f"missing entry {k!r}" for k in required if k not in self.entries]
        for key in self.entries:
            roles = role_vars_of(key, role_vars)
            if roles is None:
                problems.append(f"unknown entry {key!r}")
                continue
            known = (set(self.variables) - ({"context"} if key == "context" else set())) | roles
            try:
                unknown = self.placeholders(key) - known
            except PromptError as e:
                problems.append(str(e))
                continue
            if unknown:
                problems.append(f"{key}: unknown variable(s) {sorted(unknown)}")
        if "optimizer_system" in self.entries and not any(p.startswith("optimizer_system") for p in problems):
            try:   # textgrad formats the optimizer system prompt with the two tags: other braces must be doubled
                self.render("optimizer_system").format(new_variable_start_tag="<A>", new_variable_end_tag="</A>")
            except (KeyError, IndexError, ValueError) as e:
                problems.append(f"optimizer_system: textgrad's str.format fails ({type(e).__name__}: {e}); "
                                f"write literal braces as {{{{ }}}}")
        if problems:
            raise PromptError(f"{self.path}:\n  " + "\n  ".join(problems))

    def lock(self, path: str) -> Dict:
        """Write sha256 of every rendered static entry (no role variables) and of every raw role template."""
        def sha(v):
            return hashlib.sha256((v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
                                  .encode("utf-8")).hexdigest()
        static, templates = {}, {}
        for key, v in self.entries.items():
            if self.placeholders(key) <= set(self.variables):
                static[key] = sha(self.render(key))
            else:
                templates[key] = sha(v)
        out = {"file": os.path.basename(self.path), "file_sha256": hashlib.sha256(self._raw).hexdigest(),
               "variables": {k: v for k, v in self.variables.items() if k != "context"},
               "static": static, "templates": templates}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        return out


# -------------------------------------------------------------------------
# Teacher-blind runs
# -------------------------------------------------------------------------
#
# The teacher's identity is kept out of every text that reaches the LLM: audit() renders every prompt entry at
# startup, check() is called by LLM.call on every outgoing system/user text ("abort" raises, "scrub" replaces and
# logs), check_bank() refuses a bank whose code or descriptions name the teacher (a bank written with the teacher
# in the prompts carries that information).
#
# Matching ignores case and separators: a forbidden token's alphanumeric parts may be joined by any run of spaces,
# "_" or "-" (or nothing), so "POMO-CVRP", "pomo_cvrp", "Pomo cvrp" and "POMOCVRP" are one match; a match may not
# touch a letter on either side ("pomodoro" is not one), while digits, "_" and punctuation delimit it ("pomo_tsp50",
# "LEHD's", "lehd_feats": budget-tier programs import the LEHD-TSP primitives as cand_feats, epb/feats.py).

BLIND_FORBIDDEN = ("POMO-CVRP", "POMO-TSP", "POMO", "LEHD-CVRP", "LEHD-TSP", "LEHD")


def _blind_token(token: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", token)
    if not parts:
        raise ValueError(f"forbidden token {token!r} has no letters or digits")
    return r"[\s_\-]*".join(re.escape(p) for p in parts)


class Blind:
    def __init__(self, forbidden: Sequence[str] = BLIND_FORBIDDEN, replacement: str = "expert policy",
                 mode: str = "abort", log_path: Optional[str] = None):
        if mode not in ("abort", "scrub"):
            raise ValueError(f"Blind mode must be 'abort' or 'scrub', got {mode!r}")
        # one alternation, longest token first: "POMO-CVRP" is one match, not "POMO" + "CVRP"
        tokens = sorted({t for t in forbidden if t}, key=lambda t: len(re.findall(r"[A-Za-z0-9]", t)), reverse=True)
        if not tokens:
            raise ValueError("Blind needs at least one forbidden token")
        self.pattern = re.compile(r"(?<![A-Za-z])(?:" + "|".join(_blind_token(t) for t in tokens) + r")(?![A-Za-z])",
                                  re.IGNORECASE)
        self.replacement, self.mode, self.log_path = replacement, mode, log_path
        if self.find(replacement):
            raise ValueError(f"replacement {replacement!r} contains a forbidden token")
        self.n_checked = 0       # outgoing calls checked
        self.n_scrubbed = 0      # tokens replaced at the client boundary
        self._lock = threading.Lock()

    def find(self, text: str) -> List[str]:
        return [m.group(0) for m in self.pattern.finditer(text)]

    def scrub(self, text: str) -> Tuple[str, int]:
        return self.pattern.subn(self.replacement, text)

    def audit(self, prompts: Prompts, path: Optional[str] = None) -> str:
        """Render every entry with dummy role variables; raise BlindViolation listing every leak."""
        lines, leaks = [f"forbidden: {self.pattern.pattern}"], 0
        for key in prompts.keys():
            rendered = prompts.render(key, **{v: f"<{v}>" for v in (role_vars_of(key) or ())})
            found = [t for s in (rendered if isinstance(rendered, list) else [rendered]) for t in self.find(s)]
            leaks += len(found)
            lines.append(f"{key:28s} {'LEAK ' + str(found) if found else 'clean'}")
        lines.append(f"static-prompt leaks: {leaks}")
        report = "\n".join(lines)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(report + "\n")
        if leaks:
            raise BlindViolation(f"{leaks} forbidden token(s) in {prompts.path}:\n{report}")
        return report

    def _log(self, rec: Dict) -> None:
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def check(self, system: str, user: str, source: str) -> Tuple[str, str]:
        found = self.find(system) + self.find(user)
        with self._lock:
            self.n_checked += 1
            if not found:
                return system, user
            if self.mode == "abort":
                self._log({"call": self.n_checked, "source": source, "found": found, "action": "abort"})
                raise BlindViolation(f"forbidden token(s) {found} in an outgoing {source!r} prompt")
            system, n_sys = self.scrub(system)
            user, n_user = self.scrub(user)
            self.n_scrubbed += n_sys + n_user
            self._log({"call": self.n_checked, "source": source, "found": found, "scrubbed_this_call": n_sys + n_user,
                       "scrubbed_total": self.n_scrubbed, "sample": (system if n_sys else user)[:160]})
        return system, user

    def check_bank(self, entries) -> None:
        """entries: bank entries (dicts with code / description), or a snapshot dict with "heuristics"."""
        if isinstance(entries, Mapping):
            entries = entries["heuristics"]
        leaks = [f"entry {i} {k}: {self.find(v)}" for i, e in enumerate(entries)
                 for k, v in e.items() if isinstance(v, str) and self.find(v)]
        if leaks:
            raise BlindViolation("bank names the teacher:\n  " + "\n  ".join(leaks))


# -------------------------------------------------------------------------
# TextGrad engine / optimizer
# -------------------------------------------------------------------------

TEXTGRAD_VERSION = "0.1.8"   # the optimizer patch below copies this version's _update_prompt
_TG: Dict = {}
_TG_LOCK = threading.Lock()


def import_textgrad(log_dir: Optional[str] = None):
    """The textgrad module, imported on first use (it writes its prompt log into $TEXTGRAD_LOG_DIR, default
    ./logs/; log_dir sets it for this import). Also defines the engine / optimizer classes."""
    with _TG_LOCK:
        if "tg" in _TG:
            return _TG["tg"]
        old = os.environ.get("TEXTGRAD_LOG_DIR")
        if log_dir is not None:
            if "textgrad" in sys.modules:
                warnings.warn("textgrad was imported before: its log directory is unchanged", stacklevel=2)
            os.environ["TEXTGRAD_LOG_DIR"] = log_dir
        try:
            import textgrad as tg
            from textgrad.optimizer import TextualGradientDescentwithMomentum
            from textgrad.optimizer.optimizer import construct_tgd_prompt, logger
        finally:
            if log_dir is not None:
                if old is None:
                    os.environ.pop("TEXTGRAD_LOG_DIR", None)
                else:
                    os.environ["TEXTGRAD_LOG_DIR"] = old
        if tg.__version__ != TEXTGRAD_VERSION:
            raise ImportError(f"textgrad=={TEXTGRAD_VERSION} is required (found {tg.__version__})")

        class BudgetedTextGradEngine(tg.EngineLM):
            """textgrad engine that routes through LLM.call, so every call is budget-tracked (and blind-checked)."""
            system_prompt = "You are a helpful, creative, and smart assistant."

            def __init__(self, llm: LLM, source: str = "textgrad", system_prompt: str = system_prompt):
                self.llm, self.source = llm, source
                self.model_string = llm.model
                self.system_prompt = system_prompt

            def generate(self, prompt, system_prompt=None, temperature=0, max_tokens=None, top_p=None, **kwargs):
                if max_tokens is not None or top_p is not None or kwargs:
                    raise TypeError("BudgetedTextGradEngine takes no max_tokens / top_p / extra options "
                                    "(set llm.max_tokens in the config)")
                return self.llm.call(system_prompt or self.system_prompt, prompt,
                                     source=self.source, temperature=temperature)

            def __call__(self, prompt, **kwargs):
                return self.generate(prompt, **kwargs)

        class SafeTextualGradientDescentwithMomentum(TextualGradientDescentwithMomentum):
            """Work around a bug in textgrad 0.1.x where TextualGradientDescentwithMomentum._update_prompt()
            builds the prompt but forgets to return it, causing requests with empty user content."""

            def _update_prompt(self, variable, momentum_storage_idx: int):
                past_values = ""
                past_n_steps = self.momentum_storage[momentum_storage_idx]
                for i, step_info in enumerate(past_n_steps):
                    past_values += f"\n{variable.get_role_description()} at Step {i + 1}: {step_info['value']}.\n"

                optimizer_information = {
                    "variable_desc": variable.get_role_description(),
                    "variable_value": variable.value,
                    "variable_grad": variable.get_gradient_text(),
                    "variable_short": variable.get_short_value(),
                    "constraint_text": self.constraint_text,
                    "past_values": past_values,
                    "new_variable_start_tag": self.new_variable_tags[0],
                    "new_variable_end_tag": self.new_variable_tags[1],
                    "in_context_examples": "\n".join(self.in_context_examples),
                }
                prompt = construct_tgd_prompt(
                    do_momentum=(self.do_momentum and (past_values != "")),
                    do_constrained=self.do_constrained,
                    do_in_context_examples=(self.do_in_context_examples and (len(self.in_context_examples) > 0)),
                    **optimizer_information,
                )
                logger.info("SafeTextualGradientDescentwithMomentum prompt for update", extra={"prompt": prompt})
                return prompt

        _TG.update(tg=tg, engine=BudgetedTextGradEngine, tgd=SafeTextualGradientDescentwithMomentum)
        return tg


def textgrad_classes() -> Tuple[type, type]:
    """(BudgetedTextGradEngine, SafeTextualGradientDescentwithMomentum)."""
    import_textgrad()
    return _TG["engine"], _TG["tgd"]


def textgrad_engine(llm: LLM, source: str = "textgrad"):
    return llm.textgrad_engine(source)


def __getattr__(name: str):
    """`from epb.llm import SafeTGD` (or BudgetedTextGradEngine) imports textgrad on first use."""
    if name in ("SafeTGD", "SafeTextualGradientDescentwithMomentum"):
        return textgrad_classes()[1]
    if name == "BudgetedTextGradEngine":
        return textgrad_classes()[0]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_optimizer(engine, variable, momentum_window: int = 2, constraints: Optional[List[str]] = None,
                   optimizer_system_prompt: Optional[str] = None):
    """Patched TGD-with-momentum on one variable; optimizer_system_prompt None = textgrad's default (it must
    contain {new_variable_start_tag} / {new_variable_end_tag}, other braces doubled)."""
    if momentum_window < 1:   # textgrad 0.1.8 crashes on momentum_window=0 (__init__ returns a value)
        raise ValueError("momentum_window must be >= 1")
    kw = {} if optimizer_system_prompt is None else {"optimizer_system_prompt": optimizer_system_prompt}
    return textgrad_classes()[1](engine=engine, parameters=[variable], momentum_window=momentum_window,
                                 constraints=constraints, **kw)


def optimizer_step(optimizer, label: str, max_retries: int = 2) -> bool:
    """optimizer.step(), retried when the answer lacks the variable tags. False when the step fails; ABORTS
    propagate. A failed attempt is taken out of the momentum window again (textgrad stores the value first)."""
    for attempt in range(max_retries + 1):
        saved = [list(s) for s in getattr(optimizer, "momentum_storage", [])]
        try:
            optimizer.step()
            return True
        except ABORTS:
            raise
        except Exception as e:  # noqa: BLE001
            if saved:
                optimizer.momentum_storage = saved
            if "could not be indexed" in str(e) and attempt < max_retries:
                print(f"  [{label}] optimizer.step() format error, retrying ({attempt + 1}/{max_retries})...")
                continue
            print(f"  [{label}] optimizer.step() error: {str(e)[:300]}")
            return False
    return False


optimizer_step_with_retry = optimizer_step


def strip_code_fences(text: str) -> str:
    if "```" not in text:
        return text.strip()
    return "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("```")
    ).strip()
