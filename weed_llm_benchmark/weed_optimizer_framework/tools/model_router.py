"""Role-based model router — Phase 2 of the platform-optimization plan.

The one place that answers "for THIS job, which model, running WHERE?" — instead
of every call site hardcoding a model id. See docs/PLATFORM_OPTIMIZATION_PLAN.md.

`place` semantics (the "what runs where" doctrine):
  * "lab"     — must answer SYNCHRONOUSLY to a web request, so it has to be
                reachable from the always-on lab server: local Ollama on the 3060
                (today only `qwen2.5:3b` is pulled) OR a cloud OpenAI-compat API
                if a key is set. Keep it small + fast (it blocks the request).
  * "cluster" — consumed inside a queued cluster job (harvest brain, curation,
                labeling VLM, training), where the cluster's own Ollama / vLLM is
                reachable. NOT callable synchronously from the lab; the dashboard
                injects the chosen model into the job, it does not `chat()` it inline.

Honesty: the big models (glm-4.7-flash, deepseek-v3:671b) are deployed on the
CLUSTER and/or reachable only via a cloud key. For a lab-synchronous role we
therefore prefer the small local model and only use a bigger "deep" model when it
is actually reachable (key present / endpoint up) — otherwise we degrade to the
small one and SAY SO (via the `source` field). We never pretend a model answered
when it could not be reached.

resolve() is pure: pass in `provider_status` (from llm_providers.provider_status())
so it can be unit-tested without any network.
"""
from __future__ import annotations

# The only model pulled on the lab 3060 right now — the safe lab-sync default.
LAB_SMALL = "ollama:qwen2.5:3b"

# role -> spec. `deep` (lab roles only) is an optional bigger model used ONLY when
# reachable. `fallbacks` are tried in order when earlier choices are unreachable.
ROLES: dict = {
    "interactive_plan": {
        "place": "lab", "model": LAB_SMALL, "latency_budget_s": 60,
        "deep": "", "fallbacks": [],
        "desc": "Turn a user's intent into a project + agents plan (New Project)."},
    "analysis_summary": {
        "place": "lab", "model": LAB_SMALL, "latency_budget_s": 45,
        "deep": "", "fallbacks": [],
        "desc": "Summarize dataset EDA and give a training-readiness review."},
    "harvest_brain": {
        "place": "cluster", "model": "ollama:gemma4", "latency_budget_s": 0,
        "fallbacks": ["ollama:qwen2.5:7b"],
        "desc": "Decide what datasets to collect next (runs in the harvest job)."},
    "curation": {
        "place": "cluster", "model": "ollama:qwen2.5:7b", "latency_budget_s": 0,
        "fallbacks": ["ollama:gemma4"],
        "desc": "Judge dataset quality / on-topic-ness during curation."},
    "labeling_vlm": {
        "place": "cluster", "model": "ollama:minicpm-v", "latency_budget_s": 0,
        "fallbacks": ["ollama:llama3.2-vision", "ollama:moondream"],
        "desc": "VLM captioning / label assistance."},
    "deep_review": {
        "place": "cluster", "model": "vllm:glm-4.7-flash", "latency_budget_s": 0,
        "is_async": True, "fallbacks": ["ollama:gemma4"],
        "desc": "On-demand deeper review with a bigger cluster model (glm-4.7-flash)."},
    "hard_reasoning": {
        "place": "cluster", "model": "vllm:deepseek-v3:671b", "latency_budget_s": 0,
        "is_async": True, "rare": True, "fallbacks": ["vllm:glm-4.7-flash", "ollama:gemma4"],
        "desc": "Rare, genuinely-hard one-off reasoning (deepseek-v3 671B; SU-expensive)."},
}


def _provider_of(model_id: str) -> str:
    return str(model_id or "").partition(":")[0].strip().lower()


def _reachable(model_id: str, place: str, status: dict) -> bool:
    """Is `model_id` usable right now? Only meaningful for lab (synchronous) roles;
    cluster roles are consumed inside the job (reachable in-context) → always True."""
    if place != "lab":
        return True
    if not status:
        return True  # no info → don't block
    prov = _provider_of(model_id)
    if prov == "ollama":
        return bool((status.get("ollama") or {}).get("configured"))
    if prov == "vllm":
        # a cluster vLLM/SGLang endpoint is not reachable from the lab synchronously
        return False
    # cloud OpenAI-compat / anthropic → reachable iff a key is configured
    return bool((status.get(prov) or {}).get("configured"))


def resolve(role: str, domain_config: dict | None = None,
            global_roles: dict | None = None, provider_status: dict | None = None,
            allow_deep: bool = True) -> dict:
    """Resolve a role to a concrete model + placement, honestly.

    Precedence (first reachable wins):
      1. per-domain override  domain_config['model_routing'][role]
      2. global role config   global_roles[role]  (results/framework/model_config.json)
      3. deep tier            spec['deep']  (lab roles only, opt-in, if reachable)
      4. role default         spec['model']
      5. fallbacks            spec['fallbacks'] in order
    Returns {ok, role, model, place, latency_budget_s, is_async, source, reachable,
             desc}. `source` names WHY that model was chosen (which layer), or
             'unreachable_default' when nothing reachable was found.
    """
    spec = ROLES.get(role)
    if not spec:
        return {"ok": False, "role": role, "error": f"unknown role '{role}'",
                "model": LAB_SMALL, "place": "lab"}
    place = spec.get("place", "lab")

    cands: list = []
    dov = ((domain_config or {}).get("model_routing") or {}).get(role)
    if isinstance(dov, dict):
        dov = dov.get("model")
    if dov:
        cands.append(("domain", str(dov)))
    gov = (global_roles or {}).get(role)
    if gov:
        cands.append(("global", str(gov)))
    if allow_deep and place == "lab" and spec.get("deep"):
        cands.append(("deep", str(spec["deep"])))
    cands.append(("default", str(spec["model"])))
    for fb in spec.get("fallbacks", []):
        cands.append(("fallback", str(fb)))

    chosen, source = None, ""
    for src, mid in cands:
        if _reachable(mid, place, provider_status or {}):
            chosen, source = mid, src
            break
    if chosen is None:
        chosen, source = str(spec["model"]), "unreachable_default"

    return {"ok": True, "role": role, "model": chosen, "place": place,
            "latency_budget_s": spec.get("latency_budget_s", 60),
            "is_async": bool(spec.get("is_async")), "rare": bool(spec.get("rare")),
            "source": source,
            "reachable": _reachable(chosen, place, provider_status or {}),
            "desc": spec.get("desc", "")}


def role_table() -> list:
    """The full role table for display (settings UI / console)."""
    out = []
    for r, spec in ROLES.items():
        out.append({"role": r, "place": spec.get("place", "lab"),
                    "default": spec.get("model", ""), "deep": spec.get("deep", ""),
                    "is_async": bool(spec.get("is_async")), "rare": bool(spec.get("rare")),
                    "latency_budget_s": spec.get("latency_budget_s", 60),
                    "fallbacks": list(spec.get("fallbacks", [])),
                    "desc": spec.get("desc", "")})
    return out
