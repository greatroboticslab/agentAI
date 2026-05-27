"""Classify a botanical/biological class name into one of:
  cwd12 | weed | disease | pest | crop | other

Hybrid strategy:
  1. Keyword heuristic (no LLM) — fast, deterministic, works offline.
  2. Brain LLM (Gemma 4 via Ollama) — handles novel names like 'Kochia',
     'Velvetleaf', 'Buffelgrass' that the keyword list doesn't know.
  3. Confidence-based merge: if keyword == LLM → use that with high confidence.
     If they disagree → use LLM (it has world knowledge) but log for audit.
  4. If LLM unavailable (no ollama) → keyword-only fallback. Still safe.

Used by:
  - dataset_discovery.harvest_new_datasets() — auto-classify each new
    class_name after extracting from HF features, persist to overrides file
  - tools/dashboard_server.py — server-side topic resolution (already there)
  - tools/topic_backfill_all.py (TODO) — one-shot reclassify all 348 known
    classes using LLM, mass-fix the keyword-heuristic misclassifications
"""
from __future__ import annotations
import logging
import re
from typing import Optional, Tuple

from .class_topic_store import (
    VALID_TOPICS, load_overrides, save_override,
)

logger = logging.getLogger(__name__)


# ---------- Keyword tables (same as dashboard_server.py's heuristic) ----------
_CWD12 = {
    "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass", "Morningglory",
    "Nutsedge", "PalmerAmaranth", "PricklySida", "Purslane", "Ragweed",
    "Sicklepod", "SpottedSpurge",
}
_WEED_KEYWORDS = (
    "weed", "grass", "purslane", "amaranth", "morningglory", "ragweed",
    "sicklepod", "spurge", "nutsedge", "lantana", "parthenium", "carpetweed",
    "crabgrass", "goosegrass", "eclipta", "sida", "siamweed", "snakeweed",
    "pigweed", "smartweed", "chickweed", "fathen", "mayweed", "shepherd",
    "cranesbill", "knotweed", "silkybent", "blackgrass", "cleavers",
    "charlock", "kochia", "buttercup", "thistle", "nightshade",
    "velvetleaf", "buffelgrass", "johnsongrass", "bermudagrass",
    "dandelion", "foxtail", "ryegrass",
)
_DISEASE_KEYWORDS = (
    "blight", "rot", "mildew", "rust", "spot", "scab", "virus", "mosaic",
    "healthy", "disease", "bacterial", "septoria", "anthrac", "canker",
    "yellow", "scald", "smut", "hispa", "blast", "esca", "powdery",
    "leafminer", "monilia", "phytoph", "fusarium",
)
_PEST_KEYWORDS = (
    "ant", "bee", "beetle", "caterpillar", "earthworm", "earwig",
    "grasshopper", "moth", "slug", "snail", "wasp", "weevil", "aphid",
    "thrip", "armyworm", "borer", "looper", "fly", "mite", "bug",
    "insect", "pest",
)
_CROP_KEYWORDS = (
    "apple", "tomato", "potato", "pepper", "corn", "maize", "rice",
    "wheat", "grape", "peach", "cherry", "strawberry", "cassava",
    "guava", "coconut", "lemon", "banana", "olive", "cucumber",
    "almond", "cardamom", "chilli", "clove", "tobacco", "coffee",
    "aloevera", "ginger", "galangal", "curcuma", "eggplant", "bilimbi",
    "cantaloupe", "papaya", "mango", "soybean", "cotton", "sugarcane",
    "groundnut", "bellpepper", "watermelon", "pineapple", "carrot",
)


def classify_keyword(cls: str) -> Tuple[str, float]:
    """Returns (topic, confidence_0_to_1) from keyword matching only.
    confidence is 1.0 if hit, 0.0 if 'other' fallback."""
    if cls in _CWD12:
        return ("cwd12", 1.0)
    cl = cls.lower()
    if any(k in cl for k in _WEED_KEYWORDS):
        return ("weed", 1.0)
    if any(k in cl for k in _DISEASE_KEYWORDS):
        return ("disease", 1.0)
    if any(k in cl for k in _PEST_KEYWORDS):
        return ("pest", 1.0)
    if any(k in cl for k in _CROP_KEYWORDS):
        return ("crop", 1.0)
    return ("other", 0.0)


_LLM_PROMPT = """You categorize botanical and biological class names for a weed-detection research dataset.

Classify the given class name into EXACTLY ONE of these categories:
  - weed: undesirable plants in agricultural fields (e.g. Goosegrass, Kochia, Velvetleaf, Pigweed)
  - crop: cultivated plants harvested for food/fiber (e.g. Tomato, Apple, Soybean, Wheat)
  - disease: plant diseases or symptoms (e.g. Apple_Scab, Tomato_Bacterial_Spot, Rust)
  - pest: insects or animals that damage plants (e.g. Aphids, Beetles, Caterpillars)
  - other: anything else (image variants, augmentation flags, miscellaneous labels)

Rules:
  - Respond with the category word and NOTHING ELSE.
  - For ambiguous botanical names, use your knowledge of agricultural ecology.
  - 'Healthy' or healthy-leaf labels = disease (it's a disease-dataset class).
  - Crop-cultivar names (e.g. 'Arborio rice') = crop.

Class name: {cls}
Answer:"""


def classify_llm(cls: str, model: str = "gemma4:latest",
                 timeout_s: float = 10.0) -> Tuple[Optional[str], float]:
    """Returns (topic_str_or_None, confidence). None if LLM unreachable."""
    try:
        import ollama
    except ImportError:
        logger.debug("ollama not installed — LLM classifier unavailable")
        return (None, 0.0)
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "Answer with one word only."},
                {"role": "user", "content": _LLM_PROMPT.format(cls=cls)},
            ],
            options={"temperature": 0.0, "num_predict": 8},
        )
        text = (resp.get("message", {}).get("content") or "").strip().lower()
    except Exception as e:
        logger.warning(f"LLM classify fail for {cls!r}: {e}")
        return (None, 0.0)

    # Extract first valid topic word
    first = re.split(r"[^a-z0-9]+", text)
    for token in first:
        if token in VALID_TOPICS and token != "cwd12":
            # CWD12 is detection-specific; LLM shouldn't predict it directly.
            return (token, 0.9)
    logger.warning(f"LLM returned unparsable topic for {cls!r}: {text[:60]!r}")
    return (None, 0.0)


def classify(cls: str, use_llm: bool = True,
             model: str = "gemma4:latest",
             persist: bool = True) -> dict:
    """Resolve topic for `cls` using overrides → LLM (optional) → keywords.

    Returns:
      {topic, source, confidence, keyword_topic, llm_topic}
      source ∈ override | llm | keyword | fallback
    """
    out: dict = {
        "cls": cls, "topic": None, "source": None, "confidence": 0.0,
        "keyword_topic": None, "llm_topic": None,
    }

    # Layer 0 (v3.0.43.16): CWD12 inviolable — never override these 12 species
    # to 'weed' just because LLM said so. They have their own UI filter tab.
    if cls in _CWD12:
        out.update({"topic": "cwd12", "source": "cwd12_canonical",
                    "confidence": 1.0})
        return out

    # Layer 1: existing override (already classified before)
    overrides = load_overrides()
    if cls in overrides:
        out.update({"topic": overrides[cls], "source": "override",
                    "confidence": 1.0})
        return out

    # Compute keyword answer first (cheap, always)
    kw_topic, kw_conf = classify_keyword(cls)
    out["keyword_topic"] = kw_topic

    # Layer 2: LLM (if enabled + reachable)
    if use_llm:
        llm_topic, llm_conf = classify_llm(cls, model=model)
        out["llm_topic"] = llm_topic
        if llm_topic is not None:
            # Hybrid merge: agreement → high confidence; disagreement → trust LLM
            if llm_topic == kw_topic:
                out.update({"topic": llm_topic, "source": "llm",
                            "confidence": max(llm_conf, kw_conf)})
            else:
                logger.info(
                    f"[topic] disagree on {cls!r}: keyword={kw_topic} "
                    f"llm={llm_topic} — using LLM"
                )
                out.update({"topic": llm_topic, "source": "llm",
                            "confidence": llm_conf})
            if persist:
                save_override(cls, out["topic"])
            return out

    # Layer 3: keyword-only fallback
    out.update({
        "topic": kw_topic,
        "source": "keyword" if kw_conf > 0 else "fallback",
        "confidence": kw_conf,
    })
    # Don't persist pure-keyword fallbacks — they're the floor, not commitments
    return out


def classify_batch(class_names: list, use_llm: bool = True,
                    model: str = "gemma4:latest",
                    persist: bool = True) -> list:
    """Classify many names. Skips ones already in overrides."""
    overrides = load_overrides()
    results = []
    for cls in class_names:
        if cls in overrides:
            results.append({
                "cls": cls, "topic": overrides[cls], "source": "override",
                "confidence": 1.0, "keyword_topic": None, "llm_topic": None,
            })
            continue
        r = classify(cls, use_llm=use_llm, model=model, persist=persist)
        results.append(r)
    return results
