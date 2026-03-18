# agentAI

**A Universal Embodied Multi-Agent Framework for Agricultural Robotics**

Developed at [MTSU Great Robotics Lab](https://github.com/greatroboticslab), this repository implements **EMACF** (Embodied Multi-Agent Cognitive Framework) — a domain-agnostic agent architecture where an LLM serves as the robot's "brain" and specialized real-time agents serve as its "body". The framework is designed to be universal: the same core agent infrastructure handles weed detection, robot navigation, human-robot interaction, and future applications.

## Repository Structure

| Directory | Description | Status |
|-----------|-------------|--------|
| [`multagent/`](multagent/) | **EMACF core framework** — Event-driven multi-agent system extending MetaGPT for real-time robot control. Includes BrainAgent (LLM), PerceptionAgent (YOLO), TargetingAgent (laser), NavigationAgent, cloud-edge communication, and real-time dashboard. | Active |
| [`weed_llm_benchmark/`](weed_llm_benchmark/) | **Vision LLM Benchmark** — Evaluates 19 open-source vision LLMs against YOLO for weed detection on CottonWeedDet12 (5,648 images, 12 species). Results feed back to optimize EMACF's PerceptionAgent. | Active |
| `robot_navigation/` | **Robot Navigation** *(planned)* — Autonomous navigation module sharing the same EMACF agent framework. | Planned |

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    EMACF (multagent/)                             │
│                                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  ┌────────────┐  │
│  │  Brain    │  │  Perception  │  │ Targeting  │  │ Navigation │  │
│  │  Agent    │  │  Agent       │  │ Agent      │  │ Agent      │  │
│  │  (LLM)   │  │  (YOLO)      │  │ (Laser)    │  │ (Movement) │  │
│  └────┬─────┘  └──────┬───────┘  └─────┬─────┘  └──────┬─────┘  │
│       │               │                │               │         │
│       └───────────────┴────────────────┴───────────────┘         │
│                           EventBus                               │
│                              │                                   │
│  ┌───────────────────────────┴────────────────────────────────┐  │
│  │                    Edge Bridge (WebSocket)                  │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │   Edge Device       │
                    │   (LaserCar)        │
                    │   Camera + Laser    │
                    │   + Safety Monitor  │
                    └─────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│               weed_llm_benchmark/                                │
│                                                                  │
│   19 Vision LLMs  vs  YOLO11n  on  CottonWeedDet12              │
│   ──────────────────────────────────────────────────             │
│   Benchmark results → Optimize PerceptionAgent YOLO model        │
└──────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

1. **Universal Agent Framework** — Domain logic lives in agent implementations, not the core. The same `EmbodiedTeam`, `EventBus`, and `AgentRegistry` handle any robotic task.
2. **Event-Driven Architecture** — Agents react to events in real-time, not turn-based. The BrainAgent (LLM) only intervenes on significant events, minimizing latency.
3. **Hot-Pluggable Agents** — Add or remove agents at runtime via `AgentRegistry`. New tasks (e.g., obstacle avoidance) just require a new agent plugin.
4. **LLM-Agnostic** — Switch between vLLM, Ollama, OpenAI, or any LLM backend via config.
5. **Cloud-Edge Separation** — Compute-heavy processing (YOLO, LLM) runs on cloud GPU; edge device handles only hardware I/O with independent safety fallback.

## Current Progress

### EMACF Framework (`multagent/`)

| Component | Status | Description |
|-----------|--------|-------------|
| Core Framework | Done | `EmbodiedTeam`, `EventBus`, `AgentRegistry`, `EdgeBridge` |
| BrainAgent | Done | LLM-based cognitive center with memory and self-optimization |
| PerceptionAgent | Done | YOLO11n detection + tracking + noise filtering + trajectory prediction |
| TargetingAgent | Done | Coordinate transform + firing control + laser patterns |
| NavigationAgent | Done | Mode management + vehicle commands |
| Dashboard | Done | Vue 3 real-time visualization (live feed, agent status, metrics) |
| Edge Client | Done | Camera streaming, command execution, safety monitor |

### Weed LLM Benchmark (`weed_llm_benchmark/`)

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0: Evaluation Module | Done | `evaluate.py` (mAP, precision, recall), `datasets.py`, format converters |
| Phase 1: YOLO Baseline | **Done** | YOLO11n fine-tuned on CottonWeedDet12 — mAP@0.5=**0.929**, P=0.930, R=0.850 |
| Phase 2: Full LLM Benchmark | **Done** | 11 models evaluated on CottonWeedDet12 |
| Phase 3: YOLO+LLM Fusion | Planned | 3 strategies: supplement, filter, weighted |
| Phase 4: Ablation Studies | Planned | Prompt engineering, model size, grounding capability |
| Phase 5: Paper Writing | Planned | Figures, tables, manuscript |

#### Benchmark Results (CottonWeedDet12, 848 test images)

| Model | Type | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Time |
|-------|------|---------|--------------|-----------|--------|-----|------|
| **YOLO11n** (fine-tuned) | Detector | **0.929** | **0.865** | 0.930 | 0.850 | 0.888 | — |
| Florence-2-large | VLM | 0.329 | 0.302 | **0.692** | 0.431 | 0.531 | 662s |
| InternVL2-8B | VLM | 0.208 | 0.091 | 0.545 | 0.354 | 0.429 | 3799s |
| Qwen2.5-VL-3B | VLM | 0.196 | 0.068 | 0.333 | 0.249 | 0.285 | 5898s |
| MiniCPM-V-4.5 | VLM | 0.178 | 0.041 | 0.393 | 0.334 | 0.361 | 6695s |
| Qwen2.5-VL-7B | VLM | 0.176 | 0.059 | 0.334 | 0.214 | 0.261 | 6047s |
| Grounding-DINO-base | Detector | 0.000 | 0.000 | — | — | — | 682s |
| Llama 3.2 Vision 11B | VLM | 0.000 | 0.000 | 0.005 | 0.007 | 0.006 | 11370s |
| Moondream 1.8B | VLM | 0.000 | 0.000 | — | — | — | 1184s |
| LLaVA 7B / 13B / BakLLaVA | VLM | 0.000 | 0.000 | — | — | — | — |

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | MetaGPT (extended for embodied AI) |
| Cloud Backend | FastAPI + asyncio + WebSocket |
| Object Detection | Ultralytics YOLO11n (fine-tuned on CottonWeedDet12) |
| LLM Integration | vLLM / Ollama / OpenAI-compatible APIs |
| Dashboard | Vue 3 + Vite + ECharts |
| Hardware | HeliosDAC (laser) + ESP32 (motor/laser control) |
| Compute Cluster | PSC Bridges-2 (V100 GPUs) |
| Benchmark Models | 19 models: Qwen2.5/3-VL, Grounding DINO, Florence-2, PaliGemma2, YOLO-World, InternVL2, Molmo, DeepSeek-VL2, etc. |

## Papers

1. **"Universal Embodied Multi-Agent Cognitive Framework for Agricultural Robotics"** — EMACF architecture paper (targeting *Scientific Reports*)
2. **"Can Vision LLMs Detect Weeds? A Benchmark of Open-Source Multimodal Models for Agricultural Object Detection"** — Vision LLM benchmark (targeting *Computers and Electronics in Agriculture*)

## Getting Started

See individual project READMEs:
- [`multagent/README.md`](multagent/README.md) — EMACF setup and usage
- [`weed_llm_benchmark/README.md`](weed_llm_benchmark/README.md) — Benchmark framework documentation

## License

Research use. MTSU Great Robotics Lab.
