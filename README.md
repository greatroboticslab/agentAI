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
| Phase 2: Full LLM Benchmark | **Done** | 15 models evaluated on CottonWeedDet12 (9 with mAP > 0) |
| Phase 3: YOLO+LLM Fusion | **Done** | Only OWLv2 filter improves YOLO (+0.018 F1); LLMs cannot rescue YOLO misses |
| Phase 3B: Cross-Species Generalization | **Done** | YOLO drops 27% on unseen species; Florence-2 precision exceeds YOLO; LLM augmentation +0.009 F1 |
| Phase 3C: Anti-Forgetting Methods | **Done** | All simple methods fail; label quality (SAM+Depth) is the bottleneck |
| Phase 3D: SAM+Depth Enhanced Labeling | **Done** | SAM+Florence-2 caption: worse (-6.8% old, -11% new); caption classification too noisy |
| Phase 3E: Agent Optimizer | **Done** | **First improvement!** Florence+OWLv2 consensus: +0.016 F1 on unseen species, -0.020 forgetting |
| Phase 3F: Florence-2 Fine-tune | **Done** | Negative: fine-tuning degraded both old (-11%) and new species |
| Phase 4: HyperAgent Closed-Loop | **Done** | Qwen2.5-7B Brain: 3 rounds executed, system works but Brain needs stronger reasoning |
| Phase 4B: Weed Optimizer Framework | **Done** | 14 files, 3,522 lines. Ollama function calling, job chain, plant.id API, HuggingFace model discovery. |
| Phase 4C: Clone + Train External Models | **Done** | YOLOv8s trained from COCO→CottonWeed: F1=0.888; DETR zero-shot: F1=0; YOLO11n baseline: F1=0.917 |
| Phase 4D: plant.id Integration | **Done** | API key configured, local test OK (Status 201). Cluster needs pre-cache (network blocked). 49 credits left. |
| Phase 4E: DeepSeek-R1 Brain | **Done** | 7 action types (vs Qwen's 1). Autonomously searched HuggingFace + downloaded models. |
| Phase 4F: Extended Run (6h48m) | **Done** | 7 rounds autonomous. Filter removed 16.3% label noise. Brain reasoning loop validated. |
| Phase 4: Ablation Studies | Planned | Prompt engineering, model size, grounding capability |
| Phase 5: Paper Writing | Planned | Figures, tables, manuscript |

#### Benchmark Results (CottonWeedDet12, 848 test images)

| Model | Type | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Time |
|-------|------|---------|--------------|-----------|--------|-----|------|
| **YOLO11n** (fine-tuned) | Detector | **0.929** | **0.865** | 0.930 | 0.850 | 0.888 | — |
| Florence-2-base (0.23B) | VLM | **0.434** | **0.392** | **0.789** | 0.519 | 0.626 | 558s |
| Florence-2-large (0.77B) | VLM | 0.329 | 0.302 | 0.692 | 0.431 | 0.531 | 662s |
| InternVL2-8B | VLM | 0.208 | 0.091 | 0.545 | 0.354 | 0.429 | 3838s |
| Qwen2.5-VL-3B | VLM | 0.196 | 0.068 | 0.333 | 0.249 | 0.285 | 5898s |
| MiniCPM-V-4.5 | VLM | 0.192 | 0.043 | 0.407 | 0.340 | 0.371 | 6595s |
| OWLv2-large | Detector | 0.184 | 0.117 | 0.194 | **0.943** | 0.322 | 2519s |
| Qwen2.5-VL-7B | VLM | 0.176 | 0.059 | 0.334 | 0.214 | 0.261 | 6047s |
| InternVL2-2B | VLM | 0.002 | 0.001 | 0.038 | 0.025 | 0.031 | 2094s |
| InternVL2.5-8B | VLM | 0.000 | 0.000 | 0.016 | 0.001 | 0.001 | 6238s |
| Grounding-DINO-base | Detector | 0.000 | 0.000 | — | — | — | 843s |
| Llama 3.2 Vision 11B | VLM | 0.000 | 0.000 | 0.005 | 0.007 | 0.006 | 11370s |
| Moondream / Molmo / LLaVA | VLM | 0.000 | 0.000 | — | — | — | — |

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
