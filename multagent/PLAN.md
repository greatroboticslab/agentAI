# Embodied Multi-Agent Cognitive Framework (EMACF)
# Complete Project Plan

> Extending MetaGPT from software development multi-agent framework to embodied AI (real-time robot control).
> First application: LaserCar (autonomous laser weeding vehicle).
> Framework designed to be domain-agnostic, reusable for humanoid robots and other robotic systems.

---

## 1. Project Overview

### 1.1 Core Concept

LLM serves as the robot's "brain"; real-time Agents serve as its "body".
Users interact via natural language. The system runs autonomously, provides real-time feedback, and self-optimizes.

### 1.2 Architecture Principles

1. **Full Cloud Computation**: Local device retains only hardware I/O + safety fallback
2. **Event-Driven**: Brain doesn't think every frame; it only intervenes on significant events
3. **Hot-Pluggable Agents**: Add/remove agents at runtime via AgentRegistry
4. **LLM-Agnostic**: Framework is not tied to any specific LLM; switch via config
5. **Domain-Agnostic**: Core framework is reusable; domain logic lives in agent implementations

### 1.3 Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| Agent Framework | MetaGPT (fork + extend) | Professor's requirement; high academic citation value |
| Cloud Backend | FastAPI + asyncio | Async high-performance, native WebSocket support |
| Cloud-Edge Comm | WebSocket (JSON commands) + WebSocket (MJPEG video) | Simple, reliable, bidirectional real-time |
| Video Transport | OpenCV JPEG encode -> WebSocket binary | Controllable compression, bandwidth-adaptive |
| Frontend | Vue 3 + Vite | Reactive, component-based |
| Visualization | ECharts + Canvas | Real-time charts + video overlay |
| YOLO | ultralytics (cloud GPU) | Migrated from local to cloud |
| LLM | OpenAI-compatible API (vLLM/Ollama) | Deploy open-source models on supercomputer |
| Database | SQLite (lightweight) / PostgreSQL (production) | Store firing logs, performance data |

---

## 2. System Architecture

### 2.1 Overall Architecture Diagram

```
User (any device browser)
    |
    | HTTP/WebSocket
    v
+--------------------------------------------------------------+
|                        CLOUD (Supercomputer)                  |
|                                                               |
|  +-------------------------------------------------------+   |
|  |              Dashboard (FastAPI + Vue)                 |   |
|  |  Live Video | Agent Status | Message Flow | Brain      |   |
|  |  Thinking | User Chat | Performance Metrics            |   |
|  +-------------------------+-----------------------------+   |
|                            |                                  |
|  +-------------------------+-----------------------------+   |
|  |            MetaGPT EmbodiedTeam                       |   |
|  |                                                       |   |
|  |  +---------+ +----------+ +----------+ +----------+  |   |
|  |  |  Brain  | |Perception| |Targeting | |Navigation|  |   |
|  |  |  Agent  | |  Agent   | |  Agent   | |  Agent   |  |   |
|  |  |  (LLM)  | |  (YOLO)  | |  (Laser) | | (Vehicle)|  |   |
|  |  +----+----+ +----+-----+ +----+-----+ +----+-----+  |   |
|  |       |           |            |             |        |   |
|  |  +----+-----------+------------+-------------+-----+  |   |
|  |  |              EventBus (Event Bus)               |  |   |
|  |  +------------------------+------------------------+  |   |
|  +---------------------------+---------------------------+   |
|                              |                                |
|  +---------------------------+---------------------------+   |
|  |              EdgeBridge (cloud side)                   |   |
|  |   Receive video | Send commands | Heartbeat monitor   |   |
|  +---------------------------+---------------------------+   |
|                              |                                |
+------------------------------+--------------------------------+
                               | WebSocket (Internet)
                               |
+------------------------------+--------------------------------+
|                        LOCAL (Laser Car)                       |
|                              |                                |
|  +---------------------------+---------------------------+    |
|  |              EdgeClient (local side)                  |    |
|  |   Stream video | Receive commands | Send heartbeat    |    |
|  |   Safety fallback (independent of cloud)              |    |
|  +--+----------------+------------------+------------+---+    |
|     |                |                   |                    |
|  +--+----+   +-------+--------+   +-----+------+             |
|  |Camera |   | Helios DAC     |   |   ESP32    |             |
|  |       |   | (dual laser    |   | (motor +   |             |
|  |       |   |  motors)       |   |  laser)    |             |
|  +-------+   +----------------+   +------------+             |
|                                                               |
|  +-----------------------------------------------------------+|
|  |         SafetyMonitor (independent process)                ||
|  |  Heartbeat timeout -> shut laser + stop motors             ||
|  |  (operates WITHOUT cloud connection)                       ||
|  +-----------------------------------------------------------+|
+---------------------------------------------------------------+
```

### 2.2 Data Flow

```
[Camera] --MJPEG frames--> [EdgeClient] --WebSocket--> [EdgeBridge]
                                                            |
                                                            v
                                                    [PerceptionAgent]
                                                    YOLO detection + tracking
                                                            |
                                               WeedDetectionEvent
                                                            |
                                         +------------------+
                                         v                  v
                                  [BrainAgent]       [TargetingAgent]
                                  analyze+decide     target selection +
                                         |           coord transform
                                ParamAdjustment           |
                                         |          LaserCommand
                                         v                |
                                  [Agent params]          v
                                                    [EdgeBridge]
                                                    --WebSocket--> [EdgeClient]
                                                                   --serial--> [Hardware]
```

---

## 3. Directory Structure

```
multagent/
|
|-- PLAN.md                              # This document (project blueprint)
|-- lasercar.py                          # Original monolithic code (kept for reference, no longer runs)
|-- requirements.txt                     # Cloud dependencies
|-- setup.py                             # Package installation config
|
|-- config/
|   |-- default.yaml                     # Default global configuration
|   |-- agents.yaml                      # Agent parameter configuration
|   |-- hardware.yaml                    # Hardware-related configuration
|   |-- llm.yaml                         # LLM configuration
|   +-- dashboard.yaml                   # Dashboard configuration
|
|-- core/                                # ===== Generic Framework Layer (domain-agnostic) =====
|   |-- __init__.py
|   |-- embodied_role.py                 # EmbodiedRole: extends MetaGPT Role
|   |-- embodied_action.py               # EmbodiedAction: extends MetaGPT Action
|   |-- embodied_team.py                 # EmbodiedTeam: event-driven Team
|   |-- event_bus.py                     # Event bus: publish-subscribe pattern
|   |-- events.py                        # Standard event type definitions
|   |-- agent_registry.py                # Agent registration/discovery/hot-plug
|   |-- edge_bridge.py                   # Cloud-side: cloud-edge communication manager
|   |-- safety.py                        # Cloud-side safety policies
|   +-- config_manager.py                # Configuration management
|
|-- agents/                              # ===== Domain Implementation Layer (LaserCar specific) =====
|   |-- __init__.py
|   |
|   |-- perception/                      # --- Perception Agent (Eyes) ---
|   |   |-- __init__.py
|   |   |-- agent.py                     # PerceptionAgent main class
|   |   |-- yolo_detector.py             # YOLO inference wrapper
|   |   |-- noise_filter.py              # Noise filtering (extracted from lasercar.py)
|   |   |-- weed_tracker.py              # Weed tracking + ID assignment
|   |   +-- trajectory_predictor.py      # Trajectory prediction (extracted from lasercar.py)
|   |
|   |-- targeting/                       # --- Targeting Agent (Hands) ---
|   |   |-- __init__.py
|   |   |-- agent.py                     # TargetingAgent main class
|   |   |-- coordinate_transform.py      # Camera->Laser coordinate transform (with calibration)
|   |   |-- target_selector.py           # Target priority ranking + selection strategy
|   |   |-- firing_controller.py         # Firing control (static targeting + trajectory tracking)
|   |   +-- laser_patterns.py            # Laser firing pattern generation
|   |
|   |-- navigation/                      # --- Navigation Agent (Legs) ---
|   |   |-- __init__.py
|   |   |-- agent.py                     # NavigationAgent main class
|   |   |-- vehicle_commands.py          # Vehicle movement command generation
|   |   |-- mode_manager.py              # Operation mode management (SwA/SwB/SwC/SwD)
|   |   +-- path_planner.py             # Path planning (future extension)
|   |
|   +-- brain/                           # --- Brain Agent (Cognitive Core) ---
|       |-- __init__.py
|       |-- agent.py                     # BrainAgent main class
|       |-- prompts/                     # LLM prompts
|       |   |-- system_prompt.py         # System role prompt
|       |   |-- event_analysis.py        # Event analysis prompt templates
|       |   +-- optimization.py          # Self-optimization prompt templates
|       |-- memory.py                    # Brain memory system (short-term + long-term)
|       |-- optimizer.py                 # Self-optimization engine
|       +-- user_interface.py            # Natural language interaction handler
|
|-- edge/                                # ===== Edge Client (runs on laser car) =====
|   |-- __init__.py
|   |-- edge_client.py                   # Main program entry point
|   |-- camera_streamer.py               # Camera capture + MJPEG streaming
|   |-- hardware_driver.py               # Hardware driver (Helios DAC + ESP32)
|   |-- command_executor.py              # Receive cloud commands and execute
|   |-- flysky_receiver.py               # FlySky remote control signal receiver
|   |-- safety_monitor.py                # Local safety monitoring (independent process)
|   +-- requirements_edge.txt            # Edge dependencies (minimal)
|
|-- dashboard/                           # ===== Web Visualization =====
|   |-- backend/
|   |   |-- __init__.py
|   |   |-- server.py                    # FastAPI main server
|   |   |-- ws_handlers.py              # WebSocket endpoints
|   |   |-- api_routes.py                # REST API endpoints
|   |   +-- stream_proxy.py             # Video stream proxy (edge -> frontend)
|   |
|   +-- frontend/
|       |-- package.json
|       |-- vite.config.js
|       |-- index.html
|       +-- src/
|           |-- App.vue                  # Main application
|           |-- main.js                  # Entry point
|           |-- stores/                  # Pinia state management
|           |   |-- agents.js            # Agent state
|           |   |-- messages.js          # Message flow
|           |   +-- system.js            # System state
|           |-- components/
|           |   |-- LiveFeed.vue         # Live video + YOLO overlay
|           |   |-- AgentPanel.vue       # Agent list + status
|           |   |-- BrainThought.vue     # Brain thinking process display
|           |   |-- MessageFlow.vue      # Agent message flow visualization
|           |   |-- PerformanceBoard.vue # Performance dashboard
|           |   |-- ParamControl.vue     # Parameter adjustment panel
|           |   +-- ChatPanel.vue        # User conversation panel
|           +-- utils/
|               |-- websocket.js         # WebSocket client
|               +-- formatters.js        # Data formatters
|
|-- tests/
|   |-- __init__.py
|   |-- test_core/
|   |   |-- test_event_bus.py
|   |   |-- test_agent_registry.py
|   |   +-- test_edge_bridge.py
|   |-- test_agents/
|   |   |-- test_perception.py
|   |   |-- test_targeting.py
|   |   |-- test_navigation.py
|   |   +-- test_brain.py
|   +-- test_integration/
|       |-- test_agent_coordination.py
|       +-- test_cloud_edge.py
|
+-- scripts/
    |-- start_cloud.py                   # Start all cloud services
    |-- start_edge.py                    # Start edge client
    |-- simulate.py                      # Simulation mode (no hardware, uses recorded video)
    +-- benchmark.py                     # Performance benchmark
```

---

## 4. Core Framework Layer Detailed Design (core/)

> This layer is fully domain-agnostic. It contains NO LaserCar-specific logic.
> When applying to humanoid robots in the future, only new agents/ need to be written; core/ is fully reusable.

### 4.1 EmbodiedRole (core/embodied_role.py)

Extends MetaGPT's Role class with real-time embodied capabilities.

```python
"""
Inherits from: metagpt.roles.Role
New capabilities:
  1. Event-driven response (on_event)
  2. Hardware I/O interface (send_hardware_command)
  3. Latency awareness (latency tracking)
  4. Periodic heartbeat (heartbeat)
  5. Runtime parameter dynamic adjustment (update_params)
"""

class EmbodiedRole(Role):
    # --- New attributes ---
    event_bus: EventBus           # Event bus reference
    edge_bridge: EdgeBridge       # Cloud-edge communication reference
    agent_registry: AgentRegistry # Agent registry reference
    params: dict                  # Runtime adjustable parameters
    latency_tracker: dict         # Latency tracking {metric: deque}

    # --- MetaGPT original methods (retained for compatibility) ---
    async def _act(self) -> Message:
        """MetaGPT's turn-based action, retained for compatibility"""
        pass

    async def _react(self) -> Message:
        """MetaGPT's message reaction, retained for compatibility"""
        pass

    # --- New methods ---
    async def on_event(self, event: Event) -> None:
        """
        Event-driven response (core extension).
        Unlike MetaGPT's turn-based approach, this is real-time.
        Subclasses must implement this method.
        """
        raise NotImplementedError

    async def run_loop(self) -> None:
        """
        Agent main loop.
        Listens for relevant events on event_bus, calls on_event to handle.
        Also executes periodic tasks (e.g., PerceptionAgent's detection cycle).
        """
        pass

    async def send_hardware_command(self, command: HardwareCommand) -> None:
        """Send hardware command via EdgeBridge"""
        await self.edge_bridge.send_command(command)

    def update_params(self, new_params: dict) -> None:
        """Brain dynamically adjusts this Agent's parameters"""
        self.params.update(new_params)
        self.on_params_updated(new_params)

    def on_params_updated(self, changed_params: dict) -> None:
        """Parameter change callback, subclasses can override"""
        pass

    def report_latency(self, metric: str, value_ms: float) -> None:
        """Record latency metrics for Brain analysis"""
        self.latency_tracker[metric].append(value_ms)

    def get_status(self) -> dict:
        """Return current Agent status summary (for Dashboard display)"""
        return {
            "name": self.name,
            "profile": self.profile,
            "params": self.params,
            "latency": {k: list(v)[-5:] for k, v in self.latency_tracker.items()},
        }
```

### 4.2 EventBus (core/event_bus.py)

```python
"""
Async event bus, the core of Agent inter-communication.
Supports:
  - publish/subscribe pattern (one-to-many)
  - request/response pattern (one-to-one, synchronous wait)
  - Event filtering (by type, by source Agent)
  - Event history recording (for Brain review)
  - Dashboard event stream push
"""

class EventBus:
    def __init__(self, history_size: int = 1000):
        self._subscribers: dict[str, list[Callable]] = {}
        self._history: deque[Event] = deque(maxlen=history_size)
        self._dashboard_callback: Optional[Callable] = None

    async def publish(self, event: Event) -> None:
        """
        Publish event to the bus.
        All Agents subscribed to this event type will be notified.
        Also pushes to Dashboard.
        """
        pass

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to a specific event type"""
        pass

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Unsubscribe from an event type"""
        pass

    async def request(self, target_agent: str, event: Event, timeout: float = 5.0) -> Event:
        """Synchronous request-response pattern (with timeout)"""
        pass

    def get_recent_events(self, event_type: str = None, count: int = 50) -> list[Event]:
        """Get recent events (for Brain analysis)"""
        pass

    def set_dashboard_callback(self, callback: Callable) -> None:
        """Register Dashboard push callback"""
        pass
```

### 4.3 Events (core/events.py)

```python
"""
Standard event type definitions.
All events inherit from Event base class.
Domain-specific events are defined in agents/; only generic events here.
"""

@dataclass
class Event:
    event_type: str               # Event type identifier
    source: str                   # Source Agent name
    timestamp: float              # Event timestamp
    data: dict                    # Event data payload

# --- Generic System Events ---

class AgentRegisteredEvent(Event):
    """New Agent registered"""
    # data: {"agent_name": str, "agent_profile": str}

class AgentRemovedEvent(Event):
    """Agent removed"""
    # data: {"agent_name": str, "reason": str}

class ParamUpdateEvent(Event):
    """Parameter update command (typically issued by Brain)"""
    # data: {"target_agent": str, "params": dict, "reason": str}

class EdgeConnectedEvent(Event):
    """Edge device connected"""
    # data: {"edge_id": str, "capabilities": list}

class EdgeDisconnectedEvent(Event):
    """Edge device disconnected"""
    # data: {"edge_id": str, "reason": str}

class HeartbeatEvent(Event):
    """Heartbeat event"""
    # data: {"edge_id": str, "latency_ms": float}

class UserChatEvent(Event):
    """User sends a message"""
    # data: {"message": str, "user_id": str}

class SystemStatusEvent(Event):
    """System status summary (published periodically)"""
    # data: {"agents": dict, "edge_status": dict, "performance": dict}
```

### 4.4 AgentRegistry (core/agent_registry.py)

```python
"""
Agent registry: manages the lifecycle of all active Agents.
Supports:
  - Dynamic registration/deregistration
  - Agent discovery (by name, by capability)
  - Health checks
  - Hot-plugging (add/remove Agents at runtime)
"""

class AgentRegistry:
    def __init__(self, event_bus: EventBus):
        self._agents: dict[str, EmbodiedRole] = {}
        self._event_bus = event_bus

    async def register(self, agent: EmbodiedRole) -> None:
        """
        Register a new Agent.
        Auto-connects event_bus, edge_bridge.
        Publishes AgentRegisteredEvent to notify all Agents (especially Brain).
        """
        pass

    async def unregister(self, agent_name: str, reason: str = "") -> None:
        """Unregister an Agent"""
        pass

    def get_agent(self, name: str) -> Optional[EmbodiedRole]:
        """Get Agent by name"""
        pass

    def list_agents(self) -> list[dict]:
        """List all active Agents and their status"""
        pass

    def discover(self, capability: str) -> list[EmbodiedRole]:
        """Discover Agents by capability (e.g., find all video-processing Agents)"""
        pass

    async def health_check(self) -> dict:
        """Check health status of all Agents"""
        pass
```

### 4.5 EdgeBridge (core/edge_bridge.py)

```python
"""
Cloud-side edge communication manager.
Responsibilities:
  1. Manage WebSocket connections with edge devices
  2. Receive video stream and dispatch to PerceptionAgent
  3. Receive sensor data
  4. Send control commands to edge devices
  5. Heartbeat monitoring
"""

class EdgeBridge:
    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._connections: dict[str, WebSocket] = {}
        self._video_callbacks: list[Callable] = []
        self._heartbeat_interval: float = 0.1  # 100ms

    async def start(self, host: str, port: int) -> None:
        """Start WebSocket server, wait for edge device connections"""
        pass

    async def on_edge_connected(self, edge_id: str, ws: WebSocket) -> None:
        """Edge device connection callback"""
        pass

    async def on_edge_message(self, edge_id: str, message: bytes) -> None:
        """
        Handle messages from edge device.
        Message types:
          - VIDEO_FRAME: video frame -> dispatch to video callbacks
          - SENSOR_DATA: sensor data -> publish event
          - HEARTBEAT: heartbeat -> update connection status
          - REMOTE_CONTROL: remote control data -> publish event
        """
        pass

    async def send_command(self, command: HardwareCommand) -> None:
        """
        Send control command to edge device.
        Command types:
          - LASER_ON / LASER_OFF / LASER_POWER(value)
          - DAC_POSITION(motor_idx, x, y)
          - DAC_PATTERN(motor_idx, points[])
          - VEHICLE_FORWARD / VEHICLE_STOP / VEHICLE_SPEED(value)
          - CAMERA_RESOLUTION(w, h) / CAMERA_FPS(fps)
        """
        pass

    def register_video_callback(self, callback: Callable) -> None:
        """Register video frame callback (PerceptionAgent will register)"""
        pass

    def get_connection_status(self) -> dict:
        """Get connection status of all edge devices"""
        pass
```

### 4.6 EmbodiedTeam (core/embodied_team.py)

```python
"""
Inherits from: metagpt.team.Team
Modification: from turn-based to event-driven + parallel execution.
"""

class EmbodiedTeam(Team):
    def __init__(self, config_path: str = "config/default.yaml"):
        self.config = ConfigManager(config_path)
        self.event_bus = EventBus()
        self.edge_bridge = EdgeBridge(self.event_bus)
        self.agent_registry = AgentRegistry(self.event_bus)
        self.dashboard_server = None

    async def startup(self) -> None:
        """
        Start the entire system:
        1. Load configuration
        2. Start EdgeBridge (wait for edge device connections)
        3. Register all Agents
        4. Start all Agent run_loops
        5. Start Dashboard
        """
        pass

    async def add_agent(self, agent: EmbodiedRole) -> None:
        """Hot-add Agent at runtime"""
        await self.agent_registry.register(agent)
        asyncio.create_task(agent.run_loop())

    async def remove_agent(self, agent_name: str) -> None:
        """Hot-remove Agent at runtime"""
        await self.agent_registry.unregister(agent_name)

    async def shutdown(self) -> None:
        """Safely shut down all systems"""
        pass
```

---

## 5. Domain Agent Detailed Design (agents/)

### 5.1 PerceptionAgent (Perception Agent / Eyes)

**Source code mapping** (extracted from lasercar.py):
- `AdvancedNoiseFilter` -> `agents/perception/noise_filter.py` (extract as-is)
- `WeedTrajectoryPredictor` -> `agents/perception/trajectory_predictor.py` (extract as-is)
- `EnhancedWeedTargeting.detection_thread()` YOLO detection -> `agents/perception/yolo_detector.py`
- `EnhancedWeedTargeting._update_weed_tracking()` -> `agents/perception/weed_tracker.py`

**Agent main class (agents/perception/agent.py)**:

```python
class PerceptionAgent(EmbodiedRole):
    name = "Perception"
    profile = "Weed detection, tracking, and trajectory prediction"

    # Adjustable parameters (Brain can modify dynamically)
    default_params = {
        "yolo_confidence": 0.4,
        "yolo_iou": 0.4,
        "max_area_fraction": 0.18,
        "min_area_fraction": 0.0008,
        "max_aspect_ratio": 4.0,
        "min_aspect_ratio": 0.25,
        "noise_filter_strength": 0.3,
        "noise_smoothing_window": 5,
        "noise_movement_threshold": 3.0,
        "noise_outlier_threshold": 50.0,
    }

    def __init__(self):
        self.detector = YoloDetector(model_path=config.yolo_model)
        self.tracker = WeedTracker()
        self.noise_filter = AdvancedNoiseFilter()
        self.trajectory_predictor = WeedTrajectoryPredictor()

    async def run_loop(self):
        """
        Main loop:
        1. Receive video frames from EdgeBridge
        2. YOLO inference
        3. Area/aspect ratio filtering
        4. Weed tracking + ID assignment
        5. Noise filtering
        6. Trajectory prediction
        7. Publish WeedDetectionEvent to EventBus
        """
        self.edge_bridge.register_video_callback(self._on_video_frame)
        while self.running:
            await asyncio.sleep(0.001)  # yield to event loop

    async def _on_video_frame(self, frame: np.ndarray, timestamp: float):
        """Process each video frame"""
        # 1. YOLO detection
        yolo_start = time.time()
        detections = self.detector.detect(frame, self.params)
        yolo_latency = (time.time() - yolo_start) * 1000
        self.report_latency("yolo_inference_ms", yolo_latency)

        # 2. Track + filter + predict
        tracked_weeds = self.tracker.update(detections, timestamp)
        for weed in tracked_weeds:
            filtered_pos = self.noise_filter.filter_position(
                weed.id, weed.x, weed.y
            )
            weed.filtered_x, weed.filtered_y = filtered_pos
            self.trajectory_predictor.update(weed.id, filtered_pos, timestamp)

        # 3. Publish event
        await self.event_bus.publish(WeedDetectionEvent(
            source="Perception",
            data={
                "weeds": tracked_weeds,
                "frame": frame,  # for Dashboard display
                "yolo_latency_ms": yolo_latency,
                "frame_timestamp": timestamp,
            }
        ))

    async def on_event(self, event: Event):
        """Respond to events from other Agents"""
        if isinstance(event, ParamUpdateEvent):
            if event.data["target_agent"] == self.name:
                self.update_params(event.data["params"])

    # Query methods for other Agents
    def get_trajectory_prediction(self, weed_id, duration, speed_scale=1.0):
        """Get trajectory prediction for a specific weed"""
        return self.trajectory_predictor.predict_complete_trajectory(
            weed_id, duration, speed_scale
        )
```

**Published event types**:

| Event | Trigger Condition | Data Content |
|---|---|---|
| `WeedDetectionEvent` | Each frame detection completed | weeds[], frame, yolo_latency |
| `NewWeedEvent` | New weed first appears | weed_id, position, confidence |
| `WeedLostEvent` | Weed disappears | weed_id, last_position |
| `WeedStationaryEvent` | Weed stationary beyond threshold | weed_id, position, stationary_duration |

### 5.2 TargetingAgent (Targeting Agent / Hands)

**Source code mapping**:
- `EnhancedWeedTargeting.targeting_thread()` -> `agents/targeting/agent.py`
- `EnhancedWeedTargeting.transform_coordinates()` -> `agents/targeting/coordinate_transform.py`
- `EnhancedWeedTargeting.load_calibration_data()` -> `agents/targeting/coordinate_transform.py`
- `StaticTargetingSystem` -> `agents/targeting/firing_controller.py`
- `DualMotorAutonomousTrajectoryFollower` -> `agents/targeting/firing_controller.py`
- `LaserShapeGenerator` -> `agents/targeting/laser_patterns.py`
- `EnhancedWeedTargeting._select_new_target()` -> `agents/targeting/target_selector.py`

**Agent main class (agents/targeting/agent.py)**:

```python
class TargetingAgent(EmbodiedRole):
    name = "Targeting"
    profile = "Laser targeting, coordinate transformation, and firing control"

    default_params = {
        "observation_time": 1.0,
        "prediction_duration": 8.0,
        "prediction_delay": 1.5,
        "speed_scaling_factor": 0.85,
        "static_firing_duration": 15.0,
        "min_confidence_for_execution": 0.25,
        "pattern_enabled": True,
        "pattern_type": "zigzag",
        "pattern_size": 80,
    }

    def __init__(self):
        self.coord_transform = CoordinateTransform(
            calibration_files=config.calibration_files
        )
        self.target_selector = TargetSelector()
        self.firing_controller = FiringController(self)
        self.laser_patterns = LaserPatterns()
        self.current_target = None
        self.is_firing = False

    async def run_loop(self):
        """
        Main loop:
        Listen for WeedDetectionEvent -> select target -> compute firing params -> send laser commands
        """
        self.event_bus.subscribe("WeedDetectionEvent", self._on_weed_detected)
        self.event_bus.subscribe("VehicleStoppedEvent", self._on_vehicle_stopped)
        while self.running:
            await self._process_firing_cycle()
            await asyncio.sleep(0.01)

    async def _on_weed_detected(self, event: WeedDetectionEvent):
        """Process weed detection results from Perception Agent"""
        weeds = event.data["weeds"]
        yolo_latency = event.data["yolo_latency_ms"]

        if not self.is_firing:
            target = self.target_selector.select(
                weeds, self.current_target, self.params
            )
            if target:
                self.current_target = target
                await self._start_targeting(target, yolo_latency)

    async def _start_targeting(self, target, yolo_latency):
        """Begin firing sequence"""
        # Compute coordinate transform (camera pixels -> laser DAC)
        for motor_idx in range(self.coord_transform.num_motors):
            laser_x, laser_y = self.coord_transform.transform(
                motor_idx, target.x, target.y
            )
            # Generate laser frame and send
            if self.params["pattern_enabled"]:
                points = self.laser_patterns.generate(
                    laser_x, laser_y, self.params
                )
            else:
                points = [(laser_x, laser_y)]

            await self.send_hardware_command(DACCommand(
                motor_idx=motor_idx, points=points
            ))

        # Notify other Agents
        await self.event_bus.publish(FiringStartedEvent(
            source="Targeting",
            data={"weed_id": target.id, "position": (target.x, target.y)}
        ))

    async def _on_firing_complete(self, weed_id):
        """Firing completed"""
        self.is_firing = False
        self.current_target = None
        await self.event_bus.publish(FiringCompleteEvent(
            source="Targeting",
            data={"weed_id": weed_id, "duration": firing_duration}
        ))
```

**Published event types**:

| Event | Trigger Condition | Data Content |
|---|---|---|
| `TargetSelectedEvent` | New target selected | weed_id, position, priority |
| `FiringStartedEvent` | Firing begins | weed_id, position, mode |
| `FiringCompleteEvent` | Firing completed | weed_id, duration, result |
| `LaserStatusEvent` | Laser status change | enabled, power, position |

### 5.3 NavigationAgent (Navigation Agent / Legs)

**Source code mapping**:
- `FlySkyRemoteControl._handle_switch_changes()` -> `agents/navigation/mode_manager.py`
- `FlySkyRemoteControl.send_vehicle_command()` -> `agents/navigation/vehicle_commands.py`
- SwA auto-patrol logic in `EnhancedWeedTargeting` -> `agents/navigation/agent.py`

**Agent main class (agents/navigation/agent.py)**:

```python
class NavigationAgent(EmbodiedRole):
    name = "Navigation"
    profile = "Vehicle movement, mode management, and remote control"

    default_params = {
        "forward_speed": 50,        # 0-100
        "stationary_timeout": 5.0,  # SwA mode detection wait time
        "post_strike_advance": 0.2, # Post-strike forward time
        "stabilization_time": 3.0,  # Post-stop stabilization wait time
    }

    def __init__(self):
        self.mode_manager = ModeManager()
        self.vehicle_cmds = VehicleCommands(self)
        self.current_mode = "IDLE"  # IDLE / SWA / SWB / SWC / SWD
        self.is_moving = False
        self.is_stopped_for_firing = False

    async def run_loop(self):
        """
        Main loop:
        - Listen for remote control events -> switch modes
        - Listen for firing events -> coordinate stop/go
        - SwA mode auto-patrol logic
        """
        self.event_bus.subscribe("RemoteControlEvent", self._on_remote_control)
        self.event_bus.subscribe("FiringStartedEvent", self._on_firing_started)
        self.event_bus.subscribe("FiringCompleteEvent", self._on_firing_complete)
        self.event_bus.subscribe("WeedStationaryEvent", self._on_weed_stationary)

        while self.running:
            await self._patrol_logic()  # SwA auto-patrol
            await asyncio.sleep(0.05)

    async def _on_firing_started(self, event):
        """Firing started -> stop vehicle"""
        if self.current_mode == "SWA" and self.is_moving:
            await self.vehicle_cmds.stop()
            self.is_stopped_for_firing = True
            await self.event_bus.publish(VehicleStoppedEvent(
                source="Navigation",
                data={"reason": "firing", "weed_id": event.data["weed_id"]}
            ))

    async def _on_firing_complete(self, event):
        """Firing completed -> resume forward"""
        if self.current_mode == "SWA" and self.is_stopped_for_firing:
            await asyncio.sleep(self.params["post_strike_advance"])
            await self.vehicle_cmds.forward(self.params["forward_speed"])
            self.is_stopped_for_firing = False

    async def _on_remote_control(self, event):
        """Handle remote control mode switches"""
        new_mode = self.mode_manager.process(event.data)
        if new_mode != self.current_mode:
            old_mode = self.current_mode
            self.current_mode = new_mode
            await self.event_bus.publish(ModeChangeEvent(
                source="Navigation",
                data={"old_mode": old_mode, "new_mode": new_mode}
            ))
```

**Published event types**:

| Event | Trigger Condition | Data Content |
|---|---|---|
| `VehicleStoppedEvent` | Vehicle stops | reason, position |
| `VehicleMovingEvent` | Vehicle starts moving | speed, direction |
| `ModeChangeEvent` | Operation mode switches | old_mode, new_mode |
| `RemoteControlEvent` | Remote control input | channels, switches |

### 5.4 BrainAgent (Brain Agent / Cognitive Core)

**This is the core innovation of the entire system. No corresponding old code; built entirely new.**

**Agent main class (agents/brain/agent.py)**:

```python
class BrainAgent(EmbodiedRole):
    name = "Brain"
    profile = "Cognitive decision maker with natural language interface and self-optimization"

    def __init__(self):
        self.llm = LLMInterface(config.llm)
        self.memory = BrainMemory()
        self.optimizer = SelfOptimizer(self)
        self.user_interface = UserInterface(self)
        self.pending_events: asyncio.Queue = asyncio.Queue()

        # Event aggregation (don't call LLM for every event; aggregate first)
        self.event_buffer: list[Event] = []
        self.analysis_interval: float = 2.0  # Analyze every 2 seconds

    async def run_loop(self):
        """
        Brain main loop (event-driven + periodic analysis)
        """
        # Subscribe to all key events
        self.event_bus.subscribe("WeedDetectionEvent", self._buffer_event)
        self.event_bus.subscribe("FiringCompleteEvent", self._buffer_event)
        self.event_bus.subscribe("ModeChangeEvent", self._buffer_event)
        self.event_bus.subscribe("EdgeDisconnectedEvent", self._on_emergency)
        self.event_bus.subscribe("UserChatEvent", self._on_user_chat)
        self.event_bus.subscribe("AgentRegisteredEvent", self._on_agent_change)

        while self.running:
            # Periodically analyze buffered events
            await asyncio.sleep(self.analysis_interval)
            if self.event_buffer:
                await self._analyze_events()

    async def _buffer_event(self, event: Event):
        """Buffer events (don't process immediately; wait for periodic analysis)"""
        self.event_buffer.append(event)

        # But certain high-priority events are handled immediately
        if event.event_type in ("EdgeDisconnectedEvent", "EmergencyEvent"):
            await self._on_emergency(event)

    async def _analyze_events(self):
        """
        Periodic event analysis (core cognitive function).
        Aggregates buffered events into a summary, lets LLM analyze and decide.
        """
        events = self.event_buffer.copy()
        self.event_buffer.clear()

        # Build event summary
        summary = self._build_event_summary(events)

        # Get current system status
        system_status = self._get_system_status()

        # Let LLM analyze
        prompt = self._build_analysis_prompt(summary, system_status)
        decision = await self.llm.analyze(prompt)

        # Parse and execute LLM's decisions
        await self._execute_decision(decision)

        # Record to memory
        self.memory.record(summary, decision)

        # Push to Dashboard
        await self.event_bus.publish(BrainThoughtEvent(
            source="Brain",
            data={
                "summary": summary,
                "thought": decision.reasoning,
                "actions": decision.actions,
            }
        ))

    async def _on_user_chat(self, event: UserChatEvent):
        """
        User natural language interaction (highest priority, immediate response)
        """
        user_message = event.data["message"]
        system_status = self._get_system_status()
        recent_memory = self.memory.get_recent(count=20)

        response = await self.llm.chat(
            user_message=user_message,
            system_status=system_status,
            memory=recent_memory,
        )

        # Execute LLM's decided actions
        for action in response.actions:
            await self._execute_action(action)

        # Reply to user
        await self.event_bus.publish(BrainResponseEvent(
            source="Brain",
            data={"reply": response.text, "actions_taken": response.actions}
        ))

    async def _execute_decision(self, decision):
        """
        Execute LLM's decisions.
        Decision types:
          - ADJUST_PARAM: Adjust a specific Agent's parameters
          - CHANGE_MODE: Switch operation mode
          - NOTIFY_USER: Notify the user
          - NO_ACTION: Current state is normal, no intervention needed
        """
        for action in decision.actions:
            if action.type == "ADJUST_PARAM":
                await self.event_bus.publish(ParamUpdateEvent(
                    source="Brain",
                    data={
                        "target_agent": action.target,
                        "params": action.params,
                        "reason": action.reason,
                    }
                ))
            elif action.type == "CHANGE_MODE":
                await self.event_bus.publish(ModeCommandEvent(
                    source="Brain",
                    data={"mode": action.mode, "reason": action.reason}
                ))

    def _build_event_summary(self, events: list[Event]) -> dict:
        """
        Aggregate raw events into an LLM-friendly summary.
        Example: Instead of sending 100 detection events, send:
        "Over the past 2 seconds: avg 3.2 weeds/frame detected, avg YOLO latency 45ms,
         1 weed hit, 0 misses."
        """
        summary = {
            "time_window": "last 2 seconds",
            "detection_count": 0,
            "avg_weeds_per_frame": 0,
            "avg_yolo_latency_ms": 0,
            "firings_completed": 0,
            "firings_successful": 0,
            "mode_changes": [],
            "anomalies": [],
        }
        # ... aggregation logic
        return summary

    def _get_system_status(self) -> dict:
        """Get a snapshot of the entire system's current state"""
        agents = self.agent_registry.list_agents()
        edge = self.edge_bridge.get_connection_status()
        return {
            "agents": agents,
            "edge": edge,
            "current_mode": self._get_current_mode(),
            "performance": self.optimizer.get_metrics(),
        }
```

**Brain LLM Prompt Design (agents/brain/prompts/system_prompt.py)**:

```python
BRAIN_SYSTEM_PROMPT = """
You are the cognitive brain of a laser weeding robot. You are responsible for high-level
decision making and self-optimization.

Your body parts (Agents) that you manage:
- Perception (Eyes): YOLO weed detection + tracking + trajectory prediction
- Targeting (Hands): Laser aiming + firing control
- Navigation (Legs): Vehicle movement + mode management

Your responsibilities:
1. Analyze system events and adjust parameters when needed
2. Understand user's natural language commands and translate to system operations
3. Monitor system performance; proactively intervene when anomalies detected
4. Periodically self-optimize: analyze firing effectiveness, adjust strategies

Parameters you can adjust:
- Perception: yolo_confidence, noise_filter_strength, noise_smoothing_window, ...
- Targeting: prediction_duration, prediction_delay, speed_scaling_factor, pattern_size, ...
- Navigation: forward_speed, stationary_timeout, stabilization_time, ...

Your decision output format (JSON):
{
  "reasoning": "Your thinking process",
  "actions": [
    {"type": "ADJUST_PARAM", "target": "Perception", "params": {"noise_filter_strength": 0.5}, "reason": "..."},
    {"type": "NOTIFY_USER", "message": "..."},
    {"type": "NO_ACTION"}
  ]
}

Important rules:
- Don't adjust parameters too frequently (observe for at least 10 seconds before re-adjusting)
- Safety first: any anomaly should prioritize shutting down the laser
- Parameter adjustments should be incremental; don't change too much at once
- Communicate with the user in whatever language they use
"""
```

---

## 6. Edge Client Detailed Design (edge/)

### 6.1 EdgeClient Main Program (edge/edge_client.py)

```python
"""
Minimal client running on the laser car locally.
Responsibilities:
  1. Camera capture -> stream to cloud
  2. Receive cloud commands -> execute hardware operations
  3. FlySky remote control -> forward to cloud
  4. Safety fallback -> independent of cloud
"""

class EdgeClient:
    def __init__(self, cloud_url: str):
        self.cloud_url = cloud_url  # ws://supercomputer:8765
        self.camera = CameraStreamer()
        self.hardware = HardwareDriver()
        self.flysky = FlySkyReceiver()
        self.safety = SafetyMonitor(self.hardware)

    async def run(self):
        """Main loop"""
        # 1. Connect to cloud
        ws = await self._connect_cloud()

        # 2. Start safety monitoring (independent coroutine)
        asyncio.create_task(self.safety.run())

        # 3. Run in parallel
        await asyncio.gather(
            self._stream_video(ws),        # Push video frames
            self._receive_commands(ws),     # Receive and execute commands
            self._forward_remote(ws),       # Forward remote control data
            self._send_heartbeat(ws),       # Send heartbeat
        )
```

### 6.2 Communication Protocol

**Edge -> Cloud (upstream)**:

```json
{"type": "VIDEO_FRAME", "timestamp": 1234567890.123, "data": "<base64_jpeg>"}
{"type": "HEARTBEAT", "timestamp": 1234567890.123, "hw_status": {"cpu_temp": 65, "gpu_temp": 72}}
{"type": "REMOTE_CONTROL", "channels": [1500,1500,...], "switches": "0001010"}
{"type": "HARDWARE_STATUS", "laser_enabled": true, "laser_power": 128, "dac_connected": true}
```

**Cloud -> Edge (downstream)**:

```json
{"type": "LASER_CONTROL", "action": "ON", "power": 200}
{"type": "LASER_CONTROL", "action": "OFF"}
{"type": "DAC_POSITION", "motor": 0, "x": 2048, "y": 2048}
{"type": "DAC_PATTERN", "motor": 0, "points": [{"x":2000,"y":2000},{"x":2100,"y":2100}], "duration": 30000}
{"type": "VEHICLE", "action": "FORWARD", "speed": 50}
{"type": "VEHICLE", "action": "STOP"}
{"type": "CAMERA", "resolution": [1280, 720], "fps": 30}
{"type": "HEARTBEAT_ACK"}
```

### 6.3 SafetyMonitor (edge/safety_monitor.py)

```python
"""
Local safety monitor - independent process, does NOT depend on cloud.
Rules:
  1. Cloud heartbeat timeout (500ms without HEARTBEAT_ACK) -> shut laser + stop motors
  2. Hardware temperature too high -> reduce power or shut down
  3. Remote control SwC (emergency mode) -> immediately stop everything
  4. Can always be triggered via local button for emergency stop
"""

class SafetyMonitor:
    HEARTBEAT_TIMEOUT_MS = 500
    MAX_CPU_TEMP = 85
    MAX_GPU_TEMP = 90

    async def run(self):
        while True:
            if self._heartbeat_timeout():
                self._emergency_stop("Cloud connection lost")
            if self._temperature_too_high():
                self._reduce_power()
            await asyncio.sleep(0.05)  # 50ms check interval

    def _emergency_stop(self, reason: str):
        """Emergency stop: shut laser + stop motors"""
        self.hardware.laser_off()
        self.hardware.vehicle_stop()
        print(f"[SAFETY] EMERGENCY STOP: {reason}")
```

---

## 7. Dashboard Detailed Design (dashboard/)

### 7.1 Backend API Design (dashboard/backend/)

```
FastAPI server runs on cloud, same process as the Agent system.

WebSocket endpoints:
  /ws/video        -> Real-time video frame push (MJPEG)
  /ws/agents       -> Agent status real-time push
  /ws/messages     -> Agent message flow real-time push
  /ws/brain        -> Brain thinking process real-time push
  /ws/chat         -> User conversation (bidirectional)
  /ws/performance  -> Performance metrics real-time push

REST API endpoints:
  GET  /api/agents               -> Get all Agent list and status
  POST /api/agents/{name}/params -> Manually adjust Agent parameters
  GET  /api/performance          -> Get historical performance data
  GET  /api/brain/memory         -> Get Brain memory
  POST /api/brain/chat           -> Send message to Brain (fallback, non-WebSocket)
  GET  /api/system/status        -> System status
  POST /api/system/mode          -> Switch mode
  GET  /api/config               -> Get current configuration
  PUT  /api/config               -> Update configuration
```

### 7.2 Frontend Page Layout

```
Main page layout (responsive, draggable panels):
+--------------------------------------------------------+
| Top bar: System Name | Connection Status | Current Mode |
|                                     | EMERGENCY STOP btn |
+--------+-----------------------------------------------+
|        |                                                |
| Agent  |   +-------------------------------------+      |
| Panel  |   |        LiveFeed.vue                 |      |
|        |   |   Live camera feed                  |      |
| Shows: |   |   + YOLO detection boxes overlay    |      |
| -name  |   |   + Laser position markers          |      |
| -status|   |   + Trajectory prediction lines     |      |
| -params|   |   + Firing zone annotations         |      |
| -delay |   +-------------------------------------+      |
|        |                                                |
| [+Add] |   +------------+  +-----------------+         |
| Agent  |   |BrainThought|  | PerformanceBoard|         |
|        |   |            |  |                 |         |
|        |   | What the   |  | Hit rate chart  |         |
|        |   | brain is   |  | Latency chart   |         |
|        |   | thinking   |  | YOLO time       |         |
|        |   |            |  | Network delay   |         |
|        |   | Recent     |  | Weed detection  |         |
|        |   | decisions  |  | statistics      |         |
|        |   | & reasons  |  |                 |         |
|        |   +------------+  +-----------------+         |
|        |                                                |
|        |   +-------------------------------------+      |
|        |   |        MessageFlow.vue               |      |
|        |   |  Perception --WeedDetected--> Brain  |      |
|        |   |  Brain --AdjustParam--> Targeting    |      |
|        |   |  Targeting --FireCmd--> EdgeBridge   |      |
|        |   |  (real-time animation, message flow) |      |
|        |   +-------------------------------------+      |
|        |                                                |
+--------+------------------------------------------------+
|  ChatPanel.vue                                          |
|  [User input box_________________________________][Send]|
|  Brain: "Detected 3 weeds, firing at #1..."             |
|  You: "It's very windy today"                           |
|  Brain: "Adjusted filter strength and firing area..."   |
+---------------------------------------------------------+
```

---

## 8. Development Phases & Milestones

### Phase 0: Environment Setup (Prerequisite)

```
Tasks:
  [ ] Fork MetaGPT repository locally
  [ ] Create project directory structure (per Section 3)
  [ ] Configure Python virtual environment
  [ ] Install MetaGPT + dependencies
  [ ] Create all config file templates under config/
  [ ] Create requirements.txt
  [ ] Verify MetaGPT basic functionality works (can create a simple Role)

Acceptance Criteria:
  - Can import metagpt and create a simple Role
  - Directory structure is complete
  - Config file templates are in place
```

### Phase 1: Core Framework (core/)

```
Tasks:
  [ ] 1.1 Implement core/events.py - All event type definitions
  [ ] 1.2 Implement core/event_bus.py - Publish/subscribe/request-response/history
  [ ] 1.3 Implement core/embodied_role.py - EmbodiedRole extending MetaGPT Role
  [ ] 1.4 Implement core/embodied_action.py - EmbodiedAction extending MetaGPT Action
  [ ] 1.5 Implement core/agent_registry.py - Agent registration/discovery/health check
  [ ] 1.6 Implement core/config_manager.py - YAML config loading
  [ ] 1.7 Implement core/edge_bridge.py - WebSocket server (receive edge data + send commands)
  [ ] 1.8 Implement core/embodied_team.py - Event-driven Team, system startup entry
  [ ] 1.9 Write tests/test_core/ - Unit tests

Acceptance Criteria:
  - EventBus can correctly publish/subscribe
  - EmbodiedRole can inherit MetaGPT Role and run normally
  - AgentRegistry can register/unregister/discover Agents
  - EdgeBridge can accept WebSocket connections
  - All tests pass

Dependencies: Phase 0
Estimated files: ~10 Python files
```

### Phase 2: Edge Client (edge/)

```
Tasks:
  [ ] 2.1 Implement edge/camera_streamer.py - OpenCV capture + JPEG encode + WebSocket stream
  [ ] 2.2 Implement edge/hardware_driver.py - Helios DAC + ESP32 driver
        - Extract from lasercar.py: HeliosPoint, DAC initialization, frame sending
        - Extract from lasercar.py: ESP32 serial communication, laser on/off/power control
  [ ] 2.3 Implement edge/command_executor.py - Receive cloud JSON commands -> call hardware_driver
  [ ] 2.4 Implement edge/flysky_receiver.py - Extract FlySky communication from lasercar.py
        - Keep only data receiving and parsing; mode logic moves to cloud NavigationAgent
  [ ] 2.5 Implement edge/safety_monitor.py - Heartbeat timeout + temp monitoring + emergency stop
  [ ] 2.6 Implement edge/edge_client.py - Main program, integrates all above modules
  [ ] 2.7 Write edge/requirements_edge.txt - Minimal dependency list

Acceptance Criteria:
  - EdgeClient can connect to cloud EdgeBridge
  - Video frames transport from edge to cloud (verify: cloud can decode and display)
  - Cloud sends LASER_CONTROL command -> edge correctly executes
  - Network loss for 500ms -> safety stops laser and motors
  - FlySky data forwards to cloud

Dependencies: Phase 1 (requires EdgeBridge)
Source code: Hardware-related parts of lasercar.py
```

### Phase 3: Perception Agent (agents/perception/)

```
Tasks:
  [ ] 3.1 Extract agents/perception/noise_filter.py
        - Extract AdvancedNoiseFilter class from lasercar.py as-is
        - Remove dependency on parent system; make standalone interface
  [ ] 3.2 Extract agents/perception/trajectory_predictor.py
        - Extract WeedTrajectoryPredictor class from lasercar.py as-is
  [ ] 3.3 Implement agents/perception/yolo_detector.py
        - Wrap YOLO inference (extract from detection_thread)
        - Include area filtering, aspect ratio filtering logic
  [ ] 3.4 Implement agents/perception/weed_tracker.py
        - Extract from _update_weed_tracking() and _cleanup_old_weeds()
        - Weed ID assignment, cross-frame matching, disappearance detection
  [ ] 3.5 Implement agents/perception/agent.py - PerceptionAgent main class
        - Register video frame callback
        - Chain: YOLO -> tracking -> filtering -> prediction
        - Publish events to EventBus
  [ ] 3.6 Write tests

Acceptance Criteria:
  - Receive video frames from cloud EdgeBridge -> YOLO detect -> publish WeedDetectionEvent
  - Noise filtering and trajectory prediction function identically to original code
  - YOLO latency is correctly recorded and reported

Dependencies: Phase 1, Phase 2 (requires video stream)
Source code: lasercar.py AdvancedNoiseFilter, WeedTrajectoryPredictor, detection_thread
```

### Phase 4: Targeting Agent (agents/targeting/)

```
Tasks:
  [ ] 4.1 Extract agents/targeting/coordinate_transform.py
        - Extract from lasercar.py: load_calibration_data, prepare_kdtree, transform_coordinates
        - Support dual motors
  [ ] 4.2 Extract agents/targeting/laser_patterns.py
        - Extract from lasercar.py LaserShapeGenerator
  [ ] 4.3 Implement agents/targeting/target_selector.py
        - Extract from _select_new_target() and weed_priority()
        - Target priority ranking logic
  [ ] 4.4 Implement agents/targeting/firing_controller.py
        - Merge StaticTargetingSystem and DualMotorAutonomousTrajectoryFollower
        - Static firing mode + trajectory tracking firing mode
        - Send DAC commands via EdgeBridge (no longer direct hardware access)
  [ ] 4.5 Implement agents/targeting/agent.py - TargetingAgent main class
        - Listen for WeedDetectionEvent
        - Chain: target selection -> coordinate transform -> firing control
        - Publish FiringStartedEvent / FiringCompleteEvent
  [ ] 4.6 Write tests

Acceptance Criteria:
  - Receive WeedDetectionEvent -> select target -> send DAC commands via EdgeBridge
  - Coordinate transform accuracy matches original code
  - Both static and trajectory tracking firing modes work

Dependencies: Phase 1, Phase 3 (requires WeedDetectionEvent)
Source code: lasercar.py targeting_thread, transform_coordinates, StaticTargetingSystem, etc.
```

### Phase 5: Navigation Agent (agents/navigation/)

```
Tasks:
  [ ] 5.1 Implement agents/navigation/vehicle_commands.py
        - Generate vehicle control commands (FORWARD/STOP/SPEED)
        - Send to ESP32 via EdgeBridge
  [ ] 5.2 Implement agents/navigation/mode_manager.py
        - Extract mode switching logic from FlySkyRemoteControl._handle_switch_changes()
        - SwA/SwB/SwC/SwD four-mode state machine
  [ ] 5.3 Implement agents/navigation/agent.py - NavigationAgent main class
        - Listen for RemoteControlEvent -> mode switching
        - Listen for FiringStartedEvent/FiringCompleteEvent -> stop/go coordination
        - SwA auto-patrol logic
        - SwA quadruple protection system (skip already-struck weeds)
  [ ] 5.4 Implement agents/navigation/path_planner.py - Stub (future extension)
  [ ] 5.5 Write tests

Acceptance Criteria:
  - FlySky switches SwA -> system enters auto-patrol mode
  - Weed detected -> stop -> fire -> resume forward
  - SwC emergency mode works correctly

Dependencies: Phase 1, Phase 4 (requires firing events)
Source code: lasercar.py FlySkyRemoteControl, SwA-related logic
```

### Phase 6: Brain Agent (agents/brain/)

```
Tasks:
  [ ] 6.1 Implement agents/brain/prompts/ - System prompts and templates
  [ ] 6.2 Implement agents/brain/memory.py - Short-term memory (recent events) + long-term memory (learned patterns)
  [ ] 6.3 Implement agents/brain/optimizer.py - Self-optimization engine
        - Performance metric tracking (hit rate, latency, efficiency)
        - LLM-based parameter tuning suggestions
        - Tuning effect verification (before/after comparison)
  [ ] 6.4 Implement agents/brain/user_interface.py - Natural language interaction handler
  [ ] 6.5 Implement agents/brain/agent.py - BrainAgent main class
        - Event buffering + periodic analysis
        - LLM invocation + decision parsing
        - Parameter adjustment command publishing
        - User conversation
  [ ] 6.6 Write tests (using mock LLM)

Acceptance Criteria:
  - Brain receives event summaries and produces reasonable parameter adjustments
  - User says "it's windy" -> Brain adjusts noise filter parameters
  - Consecutive misses -> Brain analyzes cause and adjusts
  - User says "start weeding" -> Brain activates SwA mode

Dependencies: Phase 1, Phase 3-5 (requires events from all Agents)
Source code: Entirely new
```

### Phase 7: Dashboard (dashboard/)

```
Tasks:
  [ ] 7.1 Implement dashboard/backend/server.py - FastAPI main server
  [ ] 7.2 Implement dashboard/backend/ws_handlers.py - All WebSocket endpoints
  [ ] 7.3 Implement dashboard/backend/api_routes.py - REST API
  [ ] 7.4 Implement dashboard/backend/stream_proxy.py - Video stream forwarding
  [ ] 7.5 Initialize Vue 3 + Vite frontend project
  [ ] 7.6 Implement LiveFeed.vue - Video + YOLO overlay
  [ ] 7.7 Implement AgentPanel.vue - Agent status
  [ ] 7.8 Implement BrainThought.vue - Brain thinking display
  [ ] 7.9 Implement MessageFlow.vue - Message flow animation
  [ ] 7.10 Implement PerformanceBoard.vue - Performance dashboard
  [ ] 7.11 Implement ChatPanel.vue - Conversation panel
  [ ] 7.12 Implement ParamControl.vue - Parameter adjustment
  [ ] 7.13 Integrate all components into App.vue

Acceptance Criteria:
  - Open Dashboard in browser; live video is visible
  - Agent message flow displays in real-time
  - Brain thinking process is visible
  - Can chat with system via Chat panel
  - Performance charts update in real-time

Dependencies: Phase 1-6 (requires all backend functionality)
```

### Phase 8: Integration Testing & Optimization

```
Tasks:
  [ ] 8.1 Full pipeline integration test (simulation mode, using recorded video)
  [ ] 8.2 Full pipeline integration test (actual hardware)
  [ ] 8.3 Performance benchmark
        - End-to-end latency (camera -> YOLO -> decision -> laser action)
        - Per-Agent processing time
        - Network latency impact on system performance
  [ ] 8.4 Comparison experiments
        - Monolithic architecture (original lasercar.py) vs multi-agent architecture
        - With Brain vs without Brain (fixed parameters)
        - Performance under different network latencies
  [ ] 8.5 Self-optimization experiments
        - Let Brain run for 1 hour; observe parameter adjustment trajectory
        - Verify whether hit rate improves
  [ ] 8.6 Script: scripts/simulate.py - Run without hardware
  [ ] 8.7 Script: scripts/benchmark.py - Automated benchmark

Acceptance Criteria:
  - System functionality matches original lasercar.py (no regression)
  - Brain's self-optimization has measurable effect
  - Dashboard display is clear and visually appealing
  - Comparison experiment data is available

Dependencies: Phase 0-7
```

---

## 9. Code Migration Mapping Table

> Precise mapping from lasercar.py to new architecture to prevent omissions.

| Original Location | Original Class/Function | New Location | Notes |
|---|---|---|---|
| L20-26 | `HeliosPoint` | `edge/hardware_driver.py` | Move as-is |
| L29-448 | `FlySkyRemoteControl` | Split | See below |
| L29-148 | - Network communication | `edge/flysky_receiver.py` | Data receive + parse |
| L149-367 | - Mode switching logic | `agents/navigation/mode_manager.py` | SwA/B/C/D logic |
| L368-428 | - Manual control | `agents/navigation/agent.py` | Manual mode handling |
| L451-632 | `AdvancedNoiseFilter` | `agents/perception/noise_filter.py` | Extract as-is |
| L634-669 | `LaserShapeGenerator` | `agents/targeting/laser_patterns.py` | Extract as-is |
| L671-965 | `WeedTrajectoryPredictor` | `agents/perception/trajectory_predictor.py` | Extract as-is |
| L967-1190 | `StaticTargetingSystem` | `agents/targeting/firing_controller.py` | Merge into firing control |
| L1191-1411 | `DualMotorAutonomousTrajectoryFollower` | `agents/targeting/firing_controller.py` | Merge into firing control |
| L1412-1639 | `EnhancedWeedTargeting.__init__` | Split across Agents + config/ | Params distributed to each Agent |
| L1640-1687 | `connect_to_esp32` | `edge/hardware_driver.py` | Move to edge |
| L1688-1741 | `send_laser_command, toggle_laser` | `edge/hardware_driver.py` | Move to edge |
| L1734-1818 | `set_laser_power, draw_power_slider` | `dashboard/` | UI moves to Dashboard |
| L1819-1918 | `load_calibration_data, prepare_kdtree` | `agents/targeting/coordinate_transform.py` | Coordinate transform |
| L1879-1919 | `transform_coordinates` | `agents/targeting/coordinate_transform.py` | Coordinate transform |
| L1920-1986 | `create_frame_data, create_pattern_frame, send_frame_to_motor` | `agents/targeting/firing_controller.py` + `edge/hardware_driver.py` | Frame gen on cloud, send on edge |
| L1987-2013 | `is_point_in_region` | `agents/targeting/coordinate_transform.py` | Region check |
| L2022-2174 | `_is_weed_already_struck_swa, _record_struck_weed_swa` | `agents/navigation/agent.py` | SwA quadruple protection |
| L2175-2368 | `_check_stationary_weeds` | `agents/perception/agent.py` | Stationary weed detection |
| L2369-2527 | `detection_thread` | `agents/perception/agent.py` | Perception Agent main loop |
| L2533-2604 | `_update_weed_tracking, _cleanup_old_weeds` | `agents/perception/weed_tracker.py` | Tracking logic |
| L2605-2744 | `targeting_thread` | `agents/targeting/agent.py` | Targeting Agent main loop |
| L2745-3049 | `control_thread` | `agents/brain/agent.py` + Dashboard | Keyboard control -> Brain + UI |
| L3050-3391 | `_display_frame` | `dashboard/` | All moves to Dashboard |
| L3392-3418 | `run` | `core/embodied_team.py` | System startup entry |
| L3421-3492 | `toggle_simulation_mode` | `scripts/simulate.py` | Simulation mode |
| L3494-3530 | `shutdown` | `core/embodied_team.py` | System shutdown |

---

## 10. Configuration File Design

### config/default.yaml

```yaml
system:
  name: "LaserCar EMACF"
  version: "1.0.0"
  log_level: "INFO"

edge:
  bridge_host: "0.0.0.0"
  bridge_port: 8765
  heartbeat_interval_ms: 100
  heartbeat_timeout_ms: 500
  video_quality: 80          # JPEG quality 0-100
  video_max_fps: 30

dashboard:
  host: "0.0.0.0"
  port: 8080
  update_rate_ms: 100
```

### config/agents.yaml

```yaml
perception:
  yolo_model: "weed4.pt"
  yolo_confidence: 0.4
  yolo_iou: 0.4
  yolo_max_det: 100
  max_area_fraction: 0.18
  min_area_fraction: 0.0008
  max_aspect_ratio: 4.0
  min_aspect_ratio: 0.25
  noise_filter_strength: 0.3
  noise_smoothing_window: 5
  noise_movement_threshold: 3.0
  noise_outlier_threshold: 50.0
  weed_cleanup_interval_s: 5.0
  weed_max_invisible_time_s: 3.0

targeting:
  observation_time_s: 1.0
  prediction_duration_s: 8.0
  prediction_delay_s: 1.5
  speed_scaling_factor: 0.85
  yolo_processing_delay_s: 1.0
  min_confidence_for_execution: 0.25
  static_firing_duration_s: 15.0
  pattern_enabled: true
  pattern_type: "zigzag"
  pattern_size: 80
  pattern_density: 0.7
  laser_max: 0xFFF
  points_per_frame: 1000
  frame_duration: 30000
  calibration_files:
    motor_0: "calibration_data_motor_0.json"
    motor_1: "calibration_data_motor_1.json"

navigation:
  forward_speed: 50
  stationary_timeout_s: 5.0
  post_strike_advance_s: 0.2
  stabilization_time_s: 3.0
  swa_struck_zone_radius: 150
  swa_zone_lifetime_s: 20.0
  swa_ignore_duration_s: 20.0

brain:
  analysis_interval_s: 2.0
  event_buffer_max: 200
  optimization_interval_s: 30.0
  min_observations_before_adjust: 10
  max_param_change_per_cycle: 0.2     # Max 20% change per cycle
```

### config/llm.yaml

```yaml
llm:
  provider: "openai_compatible"
  model: "your-model-name"
  base_url: "http://your-supercomputer:8000/v1"
  api_key: "your-api-key"
  temperature: 0.3
  max_tokens: 1024
  timeout_s: 10
```

### config/hardware.yaml

```yaml
camera:
  index: 0
  width: 1920
  height: 1080
  backend: "DSHOW"    # Windows: DSHOW, Linux: V4L2

helios_dac:
  dll_path: "HeliosLaserDAC.dll"
  num_motors: 2

esp32:
  baudrate: 115200
  laser_default_power: 128

flysky:
  esp32_ip: "192.168.1.104"
  port: 10001
```

---

## 11. Key Technical Risks & Mitigations

| Risk | Impact | Mitigation Strategy |
|---|---|---|
| Network latency too high (>200ms) | Firing accuracy decreases | Latency compensation algorithm + adaptive prediction lead |
| Network interruption | System loses control | SafetyMonitor local fallback |
| Video stream bandwidth insufficient | Frame rate drops / quality degrades | Adaptive resolution + dynamic JPEG quality adjustment |
| LLM response slow (>5s) | Brain decision delay | Async non-blocking + only non-urgent decisions |
| MetaGPT turn-based conflicts with real-time | Agents can't run in parallel | EmbodiedRole adds run_loop to override turn-based |
| YOLO model accuracy mediocre | False positives / missed detections | Brain dynamically adjusts thresholds based on statistics |

---

## 12. Execution Standards

### Development Guidelines

When developing, contributors should:

1. **Read this document first** (`PLAN.md`) to understand current progress and overall architecture
2. **Confirm current Phase**; only work on tasks in the current Phase
3. **When migrating code**: Read the corresponding section of `lasercar.py` first, understand the logic, then migrate to new location
4. **Maintain interface consistency**: Strictly implement classes and method signatures as defined in this document
5. **After completing each sub-task**: Update the corresponding checkbox in this document ([ ] -> [x])
6. **Test-driven**: Write corresponding tests after each module is completed
7. **Do not cross Phases**: Unless all tasks in the current Phase are completed and tests pass

### Naming Conventions

- File names: snake_case (e.g., `event_bus.py`)
- Class names: PascalCase (e.g., `EmbodiedRole`)
- Method names: snake_case (e.g., `on_event`)
- Event types: PascalCase + Event suffix (e.g., `WeedDetectionEvent`)
- Config keys: snake_case (e.g., `noise_filter_strength`)

### Code Style

- Python 3.9+ compatible
- Async-first (async/await)
- Type hints required
- Brief comment at top of each module explaining its responsibility
- Don't over-comment; code should be self-explanatory

---

## 13. Current Progress

```
Phase 0: Environment Setup      [ ] Not started
Phase 1: Core Framework          [ ] Not started
Phase 2: Edge Client             [ ] Not started
Phase 3: Perception Agent        [ ] Not started
Phase 4: Targeting Agent         [ ] Not started
Phase 5: Navigation Agent        [ ] Not started
Phase 6: Brain Agent             [ ] Not started
Phase 7: Dashboard               [ ] Not started
Phase 8: Integration Testing     [ ] Not started
```

---

*Document Version: v1.0*
*Created: 2026-03-07*
*Last Updated: 2026-03-07*
