"""Architecture — Framework diagram and component overview."""

import streamlit as st


def render(data):
    st.title("🏗️ Framework Architecture")

    st.markdown("""
    ## Agent Loop Pattern
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │                  SuperBrain (Gemma 4 / Qwen3)                │
    │                                                              │
    │   See state → Think → Choose tool → See result → Repeat     │
    │                      (14 tools)                              │
    └─────────────────────────┬────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────────┐
          ▼                   ▼                       ▼
     Label Tools          Training Tools         Eval + Analysis
     ├ VLM inference      ├ train_yolo           ├ evaluate
     ├ consensus          ├ freeze_train         ├ analyze_failure
     ├ filter_labels      ├ lora_train           ├ inspect_labels
     ├ plant.id API       ├ distill_train        └ search_models
     └ run_external       └ two_pass_train

          ↓                   ↓                       ↓
    ┌──────────────────────────────────────────────────────────────┐
    │                    Memory (Persistent JSON)                   │
    │   10 hard lessons + learned lessons + experiment history      │
    │   Survives across jobs → Brain reads before each decision    │
    └──────────────────────────────────────────────────────────────┘
    ```
    """)

    st.divider()

    # Component details
    st.subheader("Components")

    components = {
        "🧠 Brain (brain.py)": "Gemma 4 31B via Ollama. Native function calling. 14 tools. "
                                "Three backends: Ollama (default), HuggingFace, fallback pipeline.",
        "🔄 Orchestrator (orchestrator.py)": "Main while loop. Agent mode (Brain controls flow) and "
                                              "strategy mode (rigid pipeline). Job chaining. Forced progression.",
        "💾 Memory (memory.py)": "Persistent JSON. 10 hard lessons. Experiment history. "
                                  "Baselines. Atomic writes (.tmp → os.replace).",
        "🛡️ Monitor (monitor.py)": "Strategy validation. Forgetting detection (F1 < 0.90). "
                                    "Per-class drift analysis.",
        "🔗 LoRA (lora_yolo.py)": "Conv2d LoRA adapters. 4 modes: head, backbone, hybrid, all. "
                                   "Hybrid: backbone LoRA + head fully trained.",
        "🏷️ Label Gen (label_gen.py)": "Multi-VLM consensus. External model integration. "
                                         "Adaptive min_votes per image.",
        "🔍 Label Filter (label_filter.py)": "YOLO self-training filter. Conf>0.8 removes 22.8% noise.",
        "📊 Evaluator (evaluator.py)": "Dual-conf: mAP@conf=0.001 (full PR curve), F1@conf=0.25 (practical).",
        "🌐 Web Identifier (web_identifier.py)": "plant.id API. Cache-first. 49 credits remaining.",
        "🔎 Model Discovery (model_discovery.py)": "HuggingFace search + download + inference. "
                                                     "DETR, YOLOv8s pre-researched.",
    }

    for name, desc in components.items():
        with st.expander(name):
            st.markdown(desc)

    st.divider()

    # File inventory
    st.subheader("File Inventory")
    st.markdown("""
    ```
    weed_optimizer_framework/    (17 files, ~4,500 lines)
    ├── brain.py           (650)  SuperBrain: 14 tools, 3 backends
    ├── orchestrator.py    (780)  Agent loop, job chain, forced progression
    ├── memory.py          (270)  10 hard lessons, experiment history
    ├── monitor.py         (198)  Strategy validation, forgetting detection
    ├── config.py          (175)  VLM registry, Brain models, thresholds
    ├── run.py             (119)  CLI entry point
    ├── precache.py        (112)  Offline API caching
    ├── tools/
    │   ├── evaluator.py   (296)  Dual-conf mAP + F1
    │   ├── vlm_pool.py    (358)  Live VLM inference
    │   ├── model_discovery.py (338) HuggingFace search
    │   ├── label_gen.py   (240)  Multi-VLM consensus
    │   ├── web_identifier.py (252) plant.id API
    │   ├── lora_yolo.py   (211)  Conv2d LoRA
    │   ├── label_filter.py (158) YOLO self-training filter
    │   └── yolo_trainer.py (196) Training + replay buffer
    ```
    """)
