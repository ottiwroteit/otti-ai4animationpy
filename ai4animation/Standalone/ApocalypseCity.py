# Copyright (c) Meta Platforms, Inc. and affiliates.
import math

import pyray as pr
import raylib as rl
from ai4animation.Components.Component import Component


class ApocalypseCity(Component):
    def Start(self, params):
        self.Sky = pr.Color(206, 184, 122, 255)
        self.Street = pr.Color(70, 72, 66, 255)
        self.Sidewalk = pr.Color(116, 116, 105, 255)
        self.Lane = pr.Color(190, 162, 82, 255)
        self.Foliage = pr.Color(70, 112, 55, 255)
        self.DeepFoliage = pr.Color(42, 91, 43, 255)
        self.DryGrass = pr.Color(156, 144, 82, 255)
        self.WindowDark = pr.Color(31, 35, 34, 255)
        self.Haze = pr.Color(215, 175, 84, 44)

        self.Buildings = self.CreateBuildings()
        self.Cars = self.CreateCars()
        self.Grass = self.CreateGrass()
        self.Signs = self.CreateSigns()

    def CreateBuildings(self):
        specs = []
        for side, x in [(-1, -15.5), (1, 15.5)]:
            for i, z in enumerate([-18.0, -12.5, -7.0, -1.5, 4.0, 9.5, 15.0]):
                width = 2.2 + (i % 3) * 0.35
                depth = 1.7 + ((i + 1) % 2) * 0.45
                height = 3.5 + ((i * 7) % 5) * 1.15
                color = (
                    pr.Color(96, 91, 78, 255)
                    if i % 2 == 0
                    else pr.Color(116, 102, 82, 255)
                )
                specs.append((x, height / 2, z, width, height, depth, color))

        for i, x in enumerate([-8.0, -3.0, 2.5, 7.5]):
            height = 4.6 + i * 0.8
            specs.append(
                (
                    x,
                    height / 2,
                    -21.0,
                    2.3,
                    height,
                    1.8,
                    pr.Color(86, 84, 78, 255),
                )
            )
        return specs

    def CreateCars(self):
        return [
            (-3.4, 0.28, -7.5, 1.0, 0.55, 1.8, pr.Color(74, 82, 78, 255), 8.0),
            (3.2, 0.28, -5.4, 0.95, 0.5, 1.7, pr.Color(112, 96, 53, 255), -7.0),
            (-2.6, 0.28, 4.6, 0.9, 0.5, 1.6, pr.Color(88, 73, 64, 255), -12.0),
            (4.0, 0.28, 8.4, 1.05, 0.55, 1.85, pr.Color(62, 70, 82, 255), 5.0),
        ]

    def CreateGrass(self):
        clumps = []
        for i in range(95):
            lane = -1 if i % 2 == 0 else 1
            x = lane * (1.6 + ((i * 17) % 50) / 28.0)
            z = -12.0 + ((i * 29) % 240) / 10.0
            height = 0.25 + ((i * 13) % 20) / 60.0
            lean = (((i * 19) % 11) - 5) / 40.0
            color = self.Foliage if i % 3 else self.DryGrass
            clumps.append((x, z, height, lean, color))
        return clumps

    def CreateSigns(self):
        return [
            (-13.6, 3.2, -3.5, "HOTEL"),
            (13.4, 2.8, 5.6, "EXIT"),
            (-12.8, 2.5, 9.8, "SUBWAY"),
        ]

    def Draw(self):
        self.DrawGround()
        self.DrawBuildings()
        self.DrawCars()
        self.DrawOvergrowth()
        self.DrawStreetDetails()
        self.DrawHaze()

    def DrawGround(self):
        rl.DrawPlane(pr.Vector3(0.0, -0.025, 0.0), pr.Vector2(42.0, 46.0), self.Sky)
        rl.DrawPlane(pr.Vector3(0.0, 0.0, 0.0), pr.Vector2(7.0, 34.0), self.Street)
        rl.DrawPlane(pr.Vector3(-7.2, 0.01, 0.0), pr.Vector2(5.8, 34.0), self.Sidewalk)
        rl.DrawPlane(pr.Vector3(7.2, 0.01, 0.0), pr.Vector2(5.8, 34.0), self.Sidewalk)

        for z in [-10.0, -5.0, 0.0, 5.0, 10.0]:
            rl.DrawCube(pr.Vector3(0.0, 0.025, z), 0.08, 0.03, 1.0, self.Lane)

        for x, z, width, length in [
            (-2.0, -8.0, 1.1, 2.8),
            (2.2, -2.5, 1.4, 3.6),
            (-1.6, 4.0, 1.0, 3.0),
            (2.6, 9.0, 1.2, 2.4),
        ]:
            rl.DrawCube(pr.Vector3(x, 0.045, z), width, 0.06, length, self.DeepFoliage)

    def DrawBuildings(self):
        for x, y, z, width, height, depth, color in self.Buildings:
            rl.DrawCube(pr.Vector3(x, y, z), width, height, depth, color)
            rl.DrawCubeWires(pr.Vector3(x, y, z), width, height, depth, self.WindowDark)
            floors = max(1, int(height / 1.2))
            for floor in range(1, floors):
                wy = floor * 1.05
                for col in [-0.55, 0.0, 0.55]:
                    window_x = x + col * width
                    face_z = z + (depth / 2 + 0.015)
                    rl.DrawCube(
                        pr.Vector3(window_x, wy, face_z),
                        0.28,
                        0.22,
                        0.025,
                        self.WindowDark,
                    )

        for x, y, z, text in self.Signs:
            rl.DrawCube(pr.Vector3(x, y, z), 1.8, 0.62, 0.08, pr.Color(50, 47, 40, 255))

    def DrawCars(self):
        for x, y, z, width, height, length, color, yaw in self.Cars:
            angle = math.radians(yaw)
            offset_x = math.sin(angle) * 0.18
            offset_z = math.cos(angle) * 0.18
            body = pr.Vector3(x, y, z)
            roof = pr.Vector3(x + offset_x, y + 0.42, z + offset_z)
            rl.DrawCube(body, width, height, length, color)
            rl.DrawCube(roof, width * 0.72, height * 0.55, length * 0.48, pr.Color(42, 48, 49, 255))
            rl.DrawCubeWires(body, width, height, length, pr.Color(24, 25, 23, 255))

    def DrawOvergrowth(self):
        for x, z, height, lean, color in self.Grass:
            start = pr.Vector3(x, 0.02, z)
            end = pr.Vector3(x + lean, height, z + lean * 0.4)
            rl.DrawCylinderEx(start, end, 0.018, 0.004, 5, color)

        for z in [-9.0, -3.5, 2.0, 7.5]:
            rl.DrawCube(pr.Vector3(-5.2, 0.045, z), 0.9, 0.06, 2.2, self.Foliage)
            rl.DrawCube(pr.Vector3(5.2, 0.045, z + 1.2), 0.8, 0.06, 2.0, self.Foliage)

        for x, z in [(-13.8, -6.0), (13.8, -1.0), (-13.5, 6.5), (13.3, 10.5)]:
            rl.DrawCylinderEx(
                pr.Vector3(x, 0.2, z),
                pr.Vector3(x, 2.2, z),
                0.06,
                0.02,
                6,
                self.DeepFoliage,
            )

    def DrawStreetDetails(self):
        for z in [-11.5, -2.5, 6.5]:
            rl.DrawCylinderEx(
                pr.Vector3(-5.1, 0.0, z),
                pr.Vector3(-5.1, 1.9, z),
                0.035,
                0.035,
                8,
                pr.Color(44, 46, 43, 255),
            )
            rl.DrawCube(pr.Vector3(-5.1, 2.05, z), 0.18, 0.34, 0.18, pr.Color(120, 92, 45, 255))

    def DrawHaze(self):
        rl.DrawCube(pr.Vector3(0.0, 6.8, -22.0), 42.0, 9.0, 0.08, self.Haze)
