"""
Unified LLM provider abstraction (v3.0.138) — stdlib only (urllib, no new deps).

The framework's agents (harvest reasoning, dataset curation, labeling guidance,
VLM captioning) shouldn't be hard-wired to one model. Route a chat completion to
a LOCAL model (Ollama) or a HOSTED API by a single model id:

  ollama:gemma2:9b          -> local Ollama  (OLLAMA_HOST, default 127.0.0.1:11434)
  deepseek:deepseek-chat    -> DeepSeek API  (OpenAI-compatible)
  glm:glm-4.6               -> Zhipu GLM API (OpenAI-compatible)
  openai:gpt-4o-mini        -> OpenAI
  vllm@http://NODE:8000/v1:MODEL -> a self-hosted vLLM/SGLang OpenAI endpoint
                                    (e.g. a cluster batch job serving a big model)
  anthropic:claude-...      -> Anthropic

Why this design (cluster reality, 2026-06-27): Bridges-2 has H100-80GB nodes that
CAN host DeepSeek-V3/V4 & GLM-4.6, but only as time-limited BATCH jobs on a shared
queue (no persistent always-on server). So: sporadic agent reasoning -> APIs;
bulk LLM/VLM over a dataset -> a cluster batch job exposing a vllm@ endpoint;
small always-on -> local ollama. This module makes all of them interchangeable.

Keys: environment, or ~/.llm_keys (KEY=VALUE per line):
  DEEPSEEK_API_KEY, ZHIPU_API_KEY (or GLM_API_KEY), OPENAI_API_KEY, ANTHROPIC_API_KEY
"""
import json
import os
import urllib.request

_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
_KEYS_FILE = os.path.expanduser("~/.llm_keys")

# provider -> (env key names, OpenAI-compatible base url or None)
_OPENAI_COMPAT = {
    "deepseek": (["DEEPSEEK_API_KEY"], "https://api.deepseek.com/v1"),
    "glm":      (["ZHIPU_API_KEY", "GLM_API_KEY"], "https://open.bigmodel.cn/api/paas/v4"),
    "openai":   (["OPENAI_API_KEY"], "https://api.openai.com/v1"),
}


def _load_keys() -> dict:
    keys = {}
    try:
        if os.path.isfile(_KEYS_FILE):
            for line in open(_KEYS_FILE):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    keys[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    # env overrides file
    for k in ("DEEPSEEK_API_KEY", "ZHIPU_API_KEY", "GLM_API_KEY",
              "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if os.environ.get(k):
            keys[k] = os.environ[k]
    return keys


def _key_for(env_names, keys) -> str:
    for n in env_names:
        if keys.get(n):
            return keys[n]
    return ""


def _ollama_up() -> bool:
    try:
        with urllib.request.urlopen(_OLLAMA_HOST + "/api/tags", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def provider_status() -> dict:
    """Which backends are usable right now (key present / ollama reachable)."""
    keys = _load_keys()
    st = {}
    for prov, (envs, _base) in _OPENAI_COMPAT.items():
        st[prov] = {"configured": bool(_key_for(envs, keys)),
                    "kind": "api", "needs": envs[0]}
    st["anthropic"] = {"configured": bool(_key_for(["ANTHROPIC_API_KEY"], keys)),
                       "kind": "api", "needs": "ANTHROPIC_API_KEY"}
    st["ollama"] = {"configured": _ollama_up(), "kind": "local",
                    "needs": f"Ollama running at {_OLLAMA_HOST}"}
    st["vllm"] = {"configured": True, "kind": "local",
                  "needs": "a vllm@<url>:<model> endpoint (cluster batch job)"}
    return st


def _http_json(url, payload, headers, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(model_id: str, prompt: str, system: str = "", max_tokens: int = 256,
         timeout: int = 60) -> dict:
    """Send one prompt to `model_id`. Returns {ok, text, error, model, provider}.
    Never raises — errors come back in the dict."""
    keys = _load_keys()
    try:
        prov, _, rest = model_id.partition(":")
        prov = prov.strip().lower()

        if prov == "ollama":
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
            out = _http_json(_OLLAMA_HOST + "/api/chat",
                             {"model": rest, "messages": msgs, "stream": False,
                              "options": {"num_predict": max_tokens}},
                             {"Content-Type": "application/json"}, timeout)
            return {"ok": True, "text": (out.get("message") or {}).get("content", ""),
                    "model": model_id, "provider": prov}

        if prov in _OPENAI_COMPAT or prov == "vllm":
            if prov == "vllm":
                # vllm@http://host:8000/v1:model
                base, _, model = rest.partition(":")
                base = base.lstrip("@")
                api_key = "EMPTY"
            else:
                envs, base = _OPENAI_COMPAT[prov]
                model = rest
                api_key = _key_for(envs, keys)
                if not api_key:
                    return {"ok": False, "error": f"missing {envs[0]} (set in ~/.llm_keys)",
                            "model": model_id, "provider": prov}
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
            out = _http_json(base.rstrip("/") + "/chat/completions",
                             {"model": model, "messages": msgs, "max_tokens": max_tokens},
                             {"Content-Type": "application/json",
                              "Authorization": "Bearer " + api_key}, timeout)
            return {"ok": True,
                    "text": (out["choices"][0]["message"]["content"]),
                    "model": model_id, "provider": prov}

        if prov == "anthropic":
            api_key = _key_for(["ANTHROPIC_API_KEY"], keys)
            if not api_key:
                return {"ok": False, "error": "missing ANTHROPIC_API_KEY",
                        "model": model_id, "provider": prov}
            out = _http_json("https://api.anthropic.com/v1/messages",
                             {"model": rest, "max_tokens": max_tokens,
                              "system": system or None,
                              "messages": [{"role": "user", "content": prompt}]},
                             {"Content-Type": "application/json",
                              "x-api-key": api_key,
                              "anthropic-version": "2023-06-01"}, timeout)
            parts = out.get("content") or []
            txt = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
            return {"ok": True, "text": txt, "model": model_id, "provider": prov}

        return {"ok": False, "error": f"unknown provider '{prov}' in '{model_id}'",
                "model": model_id, "provider": prov}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}",
                "model": model_id, "provider": model_id.split(':', 1)[0]}
