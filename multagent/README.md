# EMACF — Embodied Multi-Agent Cognitive Framework

A universal, event-driven multi-agent framework that extends [MetaGPT](https://github.com/geekan/MetaGPT) from software development to **real-time embodied robot control**. An LLM serves as the robot's "brain" for high-level cognition, while specialized agents handle real-time perception, targeting, and navigation.

**First application**: LaserCar — an autonomous laser weeding vehicle.
**Design goal**: Domain-agnostic framework reusable for weed detection, robot navigation, human-robot interaction, and future robotic tasks.

## Architecture

```
                        ┌─────────────────────────────┐
                        │        Cloud (GPU)           │
                        │                              │
  User ──► Chat ──►     │  ┌────────────────────────┐  │
                        │  │      BrainAgent         │  │
                        │  │   (LLM Cognition)       │  │
                        │  │  Memory · Optimization  │  │
                        │  └──────────┬─────────────┘  │
                        │             │                 │
                        │       ┌─────┴─────┐          │
                        │       │  EventBus │          │
                        │       └─┬───┬───┬─┘          │
                        │         │   │   │            │
                        │  ┌──────┘   │   └───────┐    │
                        │  ▼          ▼           ▼    │
                        │ Perception Targeting  Navigation
                        │ Agent      Agent      Agent  │
                        │ (YOLO)     (Laser)    (Move) │
                        │                              │
                        │  ┌────────────────────────┐  │
                        │  │     Edge Bridge         │  │
                        │  │   (WebSocket Server)    │  │
                        │  └──────────┬─────────────┘  │
                        └─────────────┼────────────────┘
                                      │ WebSocket
                        ┌─────────────┼────────────────┐
                        │  Edge Device (LaserCar)      │
                        │                              │
                        │  Camera → Stream to Cloud    │
                        │  Commands ← Execute locally  │
                        │  Safety Monitor (independent) │
                        │  HeliosDAC + ESP32 + FlySky  │
                        └──────────────────────────────┘
```

## Agents

### BrainAgent (LLM-based Cognitive Center)
- Subscribes to all significant events (detections, firing results, mode changes)
- Periodically calls LLM to analyze system state and make decisions
- Publishes parameter adjustments (YOLO confidence, firing duration, etc.)
- Manages long-term memory and self-optimization strategies
- Handles natural language user interaction

### PerceptionAgent (Real-time Object Detection)
- **YOLO11n** inference on video frames (fine-tuned on CottonWeedDet12)
- Persistent weed tracking across frames (spatial matching, unique IDs)
- Multi-level noise filtering (movement detection, outlier removal, smoothing)
- Trajectory prediction for firing planning
- Configurable parameters: confidence threshold, area/aspect ratio filters

### TargetingAgent (Laser Control)
- Camera pixel → laser DAC coordinate transformation (KDTree interpolation)
- Target selection and priority scoring
- Firing state machine with safety interlocks
- Laser pattern generation (zigzag, circle, cross, etc.)

### NavigationAgent (Vehicle Movement)
- Operation mode management (manual, autonomous patrol, stop-and-fire)
- Motor control command generation
- Coordinates movement with firing events (stop during active firing)

## Core Framework

| Module | Description |
|--------|-------------|
| `core/embodied_team.py` | Top-level orchestrator — initializes agents, starts EventBus and EdgeBridge |
| `core/event_bus.py` | Async pub/sub event system — all inter-agent communication |
| `core/edge_bridge.py` | WebSocket server for cloud-edge communication |
| `core/agent_registry.py` | Hot-pluggable agent management — add/remove at runtime |
| `core/embodied_role.py` | Base class for all agents (extends MetaGPT Role) |
| `core/safety.py` | Safety policy enforcement |
| `core/config_manager.py` | YAML-based configuration loading |

## Dashboard

Real-time web dashboard for monitoring and control:

- **Live Video Feed** — Annotated video stream with detection overlays
- **Agent Status Panel** — Per-agent health, message counts, latency
- **Message Flow** — Real-time event stream visualization
- **Brain Thoughts** — LLM reasoning display
- **Chat Interface** — Natural language interaction with BrainAgent
- **Performance Board** — Detection rate, firing accuracy, system metrics
- **Parameter Control** — Adjust agent parameters in real-time

**Stack**: Vue 3 + Vite + ECharts (frontend), FastAPI + WebSocket (backend)

## Edge Device

Independent edge client with dual-layer safety:

| Component | Description |
|-----------|-------------|
| `edge/edge_client.py` | Main client — manages all edge components |
| `edge/camera_streamer.py` | MJPEG video streaming to cloud |
| `edge/command_executor.py` | Executes cloud control commands |
| `edge/hardware_driver.py` | HeliosDAC (laser) + ESP32 (motor/laser) drivers |
| `edge/flysky_receiver.py` | RC remote control receiver (10 channels) |
| `edge/safety_monitor.py` | **Independent** safety fallback — operates without cloud connection, auto emergency stop on 500ms heartbeat timeout |

## Configuration

All configuration in `config/` directory:

| File | Description |
|------|-------------|
| `default.yaml` | Global settings (ports, intervals, history size) |
| `agents.yaml` | Per-agent parameters (YOLO confidence, firing duration, patrol speed) |
| `hardware.yaml` | Edge device config (camera, DAC, ESP32, FlySky) |
| `llm.yaml` | LLM backend config (vLLM/Ollama/OpenAI endpoints) |
| `dashboard.yaml` | Dashboard server settings |

## Connection to weed_llm_benchmark

The `weed_llm_benchmark/` project benchmarks 19 vision LLMs against YOLO on the CottonWeedDet12 dataset. Results directly improve EMACF:

1. **Better YOLO model** — Fine-tuned YOLO11n (mAP@0.5=0.929) replaces the baseline model in PerceptionAgent
2. **LLM-YOLO fusion** — Benchmark identifies which VLMs can supplement YOLO detections
3. **BrainAgent intelligence** — Understanding VLM capabilities informs how BrainAgent should reason about detections

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure LLM backend
vim config/llm.yaml

# Start the system (cloud side)
python -m multagent.core.embodied_team

# Start dashboard
python scripts/run_dashboard.py

# Start edge client (on LaserCar)
python -m multagent.edge.edge_client
```

## Project Structure

```
multagent/
├── core/                    # Universal framework (domain-agnostic)
│   ├── embodied_team.py     # Top-level orchestrator
│   ├── event_bus.py         # Async event pub/sub
│   ├── edge_bridge.py       # Cloud-edge WebSocket
│   ├── agent_registry.py    # Hot-plug agent management
│   ├── embodied_role.py     # Agent base class
│   ├── safety.py            # Safety policies
│   └── config_manager.py    # Config loading
│
├── agents/                  # Domain-specific agents
│   ├── brain/               # LLM cognitive agent
│   ├── perception/          # YOLO detection + tracking
│   ├── targeting/           # Laser control
│   └── navigation/          # Vehicle movement
│
├── edge/                    # Edge device code
│   ├── edge_client.py       # Main client
│   ├── camera_streamer.py   # Video streaming
│   ├── hardware_driver.py   # DAC + ESP32 drivers
│   └── safety_monitor.py    # Independent safety
│
├── dashboard/               # Real-time web UI
│   ├── backend/             # FastAPI server
│   └── frontend/            # Vue 3 + Vite
│
├── config/                  # YAML configuration
├── tests/                   # Test suites
├── PLAN.md                  # Architecture plan
└── requirements.txt         # Dependencies
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (for YOLO inference)
- MetaGPT >= 0.8.0
- Ultralytics >= 8.0.0
- FastAPI, uvicorn, websockets, aiohttp
- Vue 3 + Node.js 18+ (for dashboard frontend)
