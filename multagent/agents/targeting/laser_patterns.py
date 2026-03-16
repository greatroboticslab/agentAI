"""
Laser pattern generation for firing.
Extracted from lasercar.py LaserShapeGenerator class.
"""

import math
from typing import List, Tuple


class LaserPatterns:
    """Generates laser firing patterns around a center position."""

    def __init__(self, points_per_frame: int = 1000):
        self.points_per_frame = points_per_frame

    def generate(
        self,
        center_x: int,
        center_y: int,
        pattern_type: str = "zigzag",
        size: int = 80,
        density: float = 0.7,
    ) -> List[Tuple[int, int]]:
        """Generate pattern points around a center position."""
        if pattern_type == "zigzag":
            return self.generate_zigzag(center_x, center_y, size, density)
        elif pattern_type == "circle":
            return self.generate_circle(center_x, center_y, size, density)
        elif pattern_type == "cross":
            return self.generate_cross(center_x, center_y, size, density)
        else:
            return [(center_x, center_y)]

    def generate_zigzag(
        self,
        center_x: int,
        center_y: int,
        size: int = 80,
        density: float = 0.7,
    ) -> List[Tuple[int, int]]:
        """
        Generate zigzag scanning pattern.
        Back-and-forth horizontal sweeps covering a rectangular area.
        """
        points = []
        half_size = size // 2
        num_lines = max(3, int(10 * density))
        points_per_line = max(5, int(20 * density))

        for line_idx in range(num_lines):
            y_offset = -half_size + (size * line_idx) // max(1, num_lines - 1)
            y = center_y + y_offset

            for pt_idx in range(points_per_line):
                if line_idx % 2 == 0:
                    # Left to right
                    x_offset = -half_size + (size * pt_idx) // max(1, points_per_line - 1)
                else:
                    # Right to left
                    x_offset = half_size - (size * pt_idx) // max(1, points_per_line - 1)

                points.append((center_x + x_offset, y))

        return points if points else [(center_x, center_y)]

    def generate_circle(
        self,
        center_x: int,
        center_y: int,
        size: int = 80,
        density: float = 0.7,
    ) -> List[Tuple[int, int]]:
        """Generate circular pattern."""
        points = []
        num_points = max(8, int(32 * density))
        radius = size // 2

        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center_x + int(radius * math.cos(angle))
            y = center_y + int(radius * math.sin(angle))
            points.append((x, y))

        return points

    def generate_cross(
        self,
        center_x: int,
        center_y: int,
        size: int = 80,
        density: float = 0.7,
    ) -> List[Tuple[int, int]]:
        """Generate cross/plus pattern."""
        points = []
        half_size = size // 2
        num_points = max(5, int(15 * density))

        # Horizontal line
        for i in range(num_points):
            x = center_x - half_size + (size * i) // max(1, num_points - 1)
            points.append((x, center_y))

        # Vertical line
        for i in range(num_points):
            y = center_y - half_size + (size * i) // max(1, num_points - 1)
            points.append((center_x, y))

        return points
