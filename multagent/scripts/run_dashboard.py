"""
Launch Dashboard with all agents in demo mode (no hardware required).
Backend runs on port 8001, frontend dev server on port 5175.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-20s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("demo")


async def generate_demo_events(event_bus):
    """Generate fake detection events for demo visualization."""
    from core.events import Event
    import random
    import math

    frame_count = 0
    while True:
        frame_count += 1
        t = time.time()

        # Simulate 1-3 weeds moving across the frame
        weeds = []
        num_weeds = random.randint(0, 3)
        for i in range(num_weeds):
            weeds.append({
                "weed_id": (frame_count + i) % 10 + 1,
                "pixel_x": int(320 + 200 * math.sin(t * 0.5 + i)),
                "pixel_y": int(240 + 100 * math.cos(t * 0.3 + i)),
                "filtered_x": int(320 + 200 * math.sin(t * 0.5 + i)),
                "filtered_y": int(240 + 100 * math.cos(t * 0.3 + i)),
                "confidence": round(random.uniform(0.6, 0.95), 2),
                "box": (100, 100, 200, 200),
                "visible": True,
            })

        await event_bus.publish(Event(
            event_type="weed_detection",
            source="Perception",
            data={
                "weeds": weeds,
                "frame_shape": (480, 640),
                "yolo_latency_ms": round(random.uniform(20, 80), 1),
                "frame_timestamp": t,
            },
        ))

        # Occasionally simulate firing events
        if frame_count % 30 == 0 and weeds:
            w = weeds[0]
            await event_bus.publish(Event(
                event_type="firing_started",
                source="Targeting",
                data={"weed_id": w["weed_id"], "position": (w["pixel_x"], w["pixel_y"]), "mode": "static"},
            ))
            await asyncio.sleep(0.5)
            await event_bus.publish(Event(
                event_type="firing_complete",
                source="Targeting",
                data={"weed_id": w["weed_id"], "duration": round(random.uniform(3, 10), 1), "mode": "static"},
            ))

        await asyncio.sleep(0.2)  # ~5 FPS demo rate


async def main():
    from core.event_bus import EventBus
    from core.agent_registry import AgentRegistry
    from agents.perception.agent import PerceptionAgent
    from agents.targeting.agent import TargetingAgent
    from agents.navigation.agent import NavigationAgent
    from agents.brain.agent import BrainAgent
    from dashboard.backend.server import DashboardServer

    # Create core components
    event_bus = EventBus()
    registry = AgentRegistry(event_bus=event_bus)

    # Create agents (no hardware needed)
    perception = PerceptionAgent()
    perception.initialize(model_path="yolo11nweed.pt")

    targeting = TargetingAgent()
    targeting.initialize()

    navigation = NavigationAgent()
    navigation.initialize()

    brain = BrainAgent()
    brain.initialize()

    # Register all agents
    await registry.register(perception)
    await registry.register(targeting)
    await registry.register(navigation)
    await registry.register(brain)

    # Create and attach dashboard
    server = DashboardServer(port=8001, cors_origins=[
        "http://localhost:5175",
        "http://localhost:3000",
    ])
    server.attach_system(event_bus, registry)

    # Start background push tasks
    await server.start_background_tasks()

    # Start demo event generator
    asyncio.create_task(generate_demo_events(event_bus))

    logger.info("=" * 50)
    logger.info("EMACF Dashboard Demo")
    logger.info("Backend:  http://localhost:8001")
    logger.info("Frontend: http://localhost:5175")
    logger.info("API docs: http://localhost:8001/docs")
    logger.info("=" * 50)

    # Run FastAPI with uvicorn
    import uvicorn
    config = uvicorn.Config(server.app, host="0.0.0.0", port=8001, log_level="info")
    uv_server = uvicorn.Server(config)
    await uv_server.serve()


if __name__ == "__main__":
    asyncio.run(main())
