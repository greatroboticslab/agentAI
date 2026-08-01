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
  gemini:gemini-2.0-flash   -> Google Gemini (generateContent API)

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
import urllib.error
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
              "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
              "GOOGLE_API_KEY"):
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
    st["gemini"] = {"configured": bool(_key_for(["GEMINI_API_KEY", "GOOGLE_API_KEY"], keys)),
                    "kind": "api", "needs": "GEMINI_API_KEY"}
    st["ollama"] = {"configured": _ollama_up(), "kind": "local",
                    "needs": f"Ollama running at {_OLLAMA_HOST}"}
    st["vllm"] = {"configured": True, "kind": "local",
                  "needs": "a vllm@<url>:<model> endpoint (cluster batch job)"}
    return st


def _http_json(url, payload, headers, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        # Surface the provider's own error text (e.g. "Incorrect API key
        # provided") so bring-your-own-key users see WHY it failed, not a bare
        # "HTTP Error 401". Never include the request (which holds the key).
        detail = ""
        try:
            body = e.read().decode("utf-8", "replace")[:4000]
            try:
                j = json.loads(body)
                err = j.get("error")
                if isinstance(err, dict):
                    detail = err.get("message") or ""
                elif isinstance(err, str):
                    detail = err
                detail = detail or j.get("message") or body
            except Exception:
                detail = body
        except Exception:
            detail = ""
        raise RuntimeError(f"HTTP {e.code}: {(detail or e.reason)[:240]}")


def _http_get_json(url, headers, timeout=20):
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            body = e.read().decode("utf-8", "replace")[:4000]
            j = json.loads(body)
            err = j.get("error")
            detail = (err.get("message") if isinstance(err, dict) else err) or j.get("message") or body
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code}: {(detail or e.reason)[:200]}")


def list_models(provider: str, keys: dict = None) -> dict:
    """v3.14.4 — ask the PROVIDER which models this account can actually use, so
    the UI can offer a dropdown instead of asking the user to type a model id.
    Returns {ok, models:[{id,label}], error}."""
    merged = _load_keys()
    if keys:
        merged.update({k: v for k, v in keys.items() if v})
    prov = (provider or "").strip().lower()
    try:
        if prov == "openai":
            k = _key_for(["OPENAI_API_KEY"], merged)
            if not k:
                return {"ok": False, "error": "no OpenAI key saved", "models": []}
            out = _http_get_json("https://api.openai.com/v1/models",
                                 {"Authorization": "Bearer " + k})
            ids = sorted({m.get("id", "") for m in (out.get("data") or [])})
            keep = [i for i in ids if i.startswith(("gpt-", "o1", "o3", "o4", "chatgpt"))
                    and not any(x in i for x in ("audio", "realtime", "transcribe", "tts",
                                                 "image", "embedding", "moderation"))]
            return {"ok": True, "models": [{"id": i, "label": i} for i in keep] or
                    [{"id": i, "label": i} for i in ids[:60]]}
        if prov == "anthropic":
            k = _key_for(["ANTHROPIC_API_KEY"], merged)
            if not k:
                return {"ok": False, "error": "no Anthropic key saved", "models": []}
            out = _http_get_json("https://api.anthropic.com/v1/models?limit=100",
                                 {"x-api-key": k, "anthropic-version": "2023-06-01"})
            ms = [{"id": m.get("id", ""), "label": m.get("display_name") or m.get("id", "")}
                  for m in (out.get("data") or []) if m.get("id")]
            return {"ok": True, "models": ms}
        if prov in ("gemini", "google"):
            k = _key_for(["GEMINI_API_KEY", "GOOGLE_API_KEY"], merged)
            if not k:
                return {"ok": False, "error": "no Gemini key saved", "models": []}
            out = _http_get_json(
                "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                {"x-goog-api-key": k})
            ms = []
            for m in (out.get("models") or []):
                if "generateContent" not in (m.get("supportedGenerationMethods") or []):
                    continue
                mid = (m.get("name") or "").replace("models/", "")
                if mid:
                    ms.append({"id": mid, "label": m.get("displayName") or mid})
            return {"ok": True, "models": ms}
        return {"ok": False, "error": f"unknown provider '{provider}'", "models": []}
    except Exception as e:
        msg = str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}"
        return {"ok": False, "error": msg[:240], "models": []}


def chat(model_id: str, prompt: str, system: str = "", max_tokens: int = 256,
         timeout: int = 60, keys: dict = None) -> dict:
    """Send one prompt to `model_id`. Returns {ok, text, error, model, provider}.
    Never raises — errors come back in the dict.

    `keys` (v3.10): a per-request key map (e.g. {"OPENAI_API_KEY": "..."}) that
    OVERRIDES the server-global ~/.llm_keys. This is how a user's OWN commercial
    key (bring-your-own-key) is injected for one request without ever being
    written to the shared key file."""
    merged = _load_keys()
    if keys:
        merged.update({k: v for k, v in keys.items() if v})
    keys = merged
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

        if prov in ("gemini", "google"):
            api_key = _key_for(["GEMINI_API_KEY", "GOOGLE_API_KEY"], keys)
            if not api_key:
                return {"ok": False, "error": "missing GEMINI_API_KEY",
                        "model": model_id, "provider": "gemini"}
            body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": max_tokens}}
            if system:
                body["system_instruction"] = {"parts": [{"text": system}]}
            url = ("https://generativelanguage.googleapis.com/v1beta/models/"
                   + rest + ":generateContent")
            out = _http_json(url, body,
                             {"Content-Type": "application/json",
                              "x-goog-api-key": api_key}, timeout)
            cands = out.get("candidates") or []
            txt = ""
            if cands:
                for p in ((cands[0].get("content") or {}).get("parts") or []):
                    txt += p.get("text", "")
            return {"ok": True, "text": txt, "model": model_id, "provider": "gemini"}

        return {"ok": False, "error": f"unknown provider '{prov}' in '{model_id}'",
                "model": model_id, "provider": prov}
    except Exception as e:
        # RuntimeError carries our own already-clean provider message (e.g. the
        # OpenAI/Gemini auth text); other exceptions get their type for context.
        msg = str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {str(e)}"
        return {"ok": False, "error": msg[:240],
                "model": model_id, "provider": model_id.split(':', 1)[0]}
