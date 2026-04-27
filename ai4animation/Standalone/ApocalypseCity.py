# Copyright (c) Meta Platforms, Inc. and affiliates.
import math

import pyray as pr
import raylib as rl
from ai4animation.Components.Component import Component


class ApocalypseCity(Component):
    def Start(self, params):
        self.StreetLength = 132.0
        self.RoadHalfWidth = 4.2
        self.SidewalkWidth = 3.1
        self.Setback = 1.2

        self.SkyUpper = pr.Color(50, 78, 88, 255)
        self.SkyMid = pr.Color(114, 119, 103, 255)
        self.SkyHorizon = pr.Color(134, 143, 133, 255)
        self.CloudLight = pr.Color(205, 209, 196, 165)
        self.CloudShadow = pr.Color(93, 105, 101, 130)
        self.Ground = pr.Color(103, 105, 82, 255)
        self.Street = pr.Color(57, 60, 56, 255)
        self.StreetPatch = pr.Color(45, 47, 43, 255)
        self.Sidewalk = pr.Color(118, 118, 108, 255)
        self.Curb = pr.Color(145, 139, 118, 255)
        self.Lane = pr.Color(190, 162, 82, 255)
        self.Foliage = pr.Color(73, 118, 59, 255)
        self.DeepFoliage = pr.Color(42, 88, 42, 255)
        self.DryGrass = pr.Color(157, 143, 82, 255)
        self.WindowDark = pr.Color(27, 32, 33, 255)
        self.WindowDust = pr.Color(69, 83, 79, 255)
        self.Haze = pr.Color(220, 181, 86, 42)

        self.Buildings, self.CrossStreets = self.CreateBuildings()
        self.Skyline = self.CreateSkyline()
        self.Cars = self.CreateCars()
        self.Grass = self.CreateGrass()
        self.Trees = self.CreateTrees()
        self.StreetProps = self.CreateStreetProps()

    def CreateBuildings(self):
        specs = []
        cross_streets = []
        palette = [
            pr.Color(93, 91, 83, 255),
            pr.Color(111, 98, 80, 255),
            pr.Color(82, 86, 82, 255),
            pr.Color(125, 111, 91, 255),
            pr.Color(76, 78, 75, 255),
        ]

        for side in [-1, 1]:
            z = -58.0
            block = 0
            while z < 60.0:
                if block in [3, 8, 14]:
                    cross_streets.append((side, z, 4.8))
                    z += 5.5
                    block += 1
                    continue

                depth = 4.8 + ((block * 7 + side) % 4) * 0.75
                width = 4.0 + ((block * 5) % 4) * 0.7
                height = 5.0 + ((block * 11 + side) % 7) * 1.25
                height += max(0.0, (18.0 - abs(z)) * 0.06)
                x = side * (
                    self.RoadHalfWidth
                    + self.SidewalkWidth
                    + self.Setback
                    + width / 2.0
                    + ((block % 3) * 0.25)
                )
                color = palette[(block + (0 if side < 0 else 2)) % len(palette)]
                roof = pr.Color(
                    max(color.r - 20, 0),
                    max(color.g - 20, 0),
                    max(color.b - 18, 0),
                    255,
                )
                specs.append((side, x, height / 2.0, z, width, height, depth, color, roof))
                z += depth + 1.2 + ((block * 3) % 3) * 0.55
                block += 1

        return specs, cross_streets

    def CreateSkyline(self):
        specs = []
        for i, x in enumerate([-28.0, -22.5, -16.5, -10.0, -4.0, 2.0, 8.0, 14.5, 21.0, 27.0]):
            height = 8.5 + ((i * 9) % 8) * 1.1
            width = 3.2 + (i % 3) * 0.9
            color = pr.Color(68 + (i % 2) * 12, 74 + (i % 3) * 6, 70, 255)
            specs.append((x, height / 2.0, -73.0, width, height, 2.2, color))
        return specs

    def CreateCars(self):
        return [
            (-3.5, 0.28, -35.0, 1.0, 0.52, 1.85, pr.Color(73, 80, 76, 255), 7.0),
            (3.1, 0.28, -24.0, 0.95, 0.5, 1.7, pr.Color(113, 96, 54, 255), -8.0),
            (-2.7, 0.28, -9.5, 0.9, 0.5, 1.65, pr.Color(87, 72, 63, 255), -12.0),
            (3.8, 0.28, 7.5, 1.05, 0.55, 1.9, pr.Color(62, 70, 82, 255), 5.0),
            (-3.3, 0.28, 27.0, 1.0, 0.52, 1.8, pr.Color(58, 66, 63, 255), 13.0),
            (3.4, 0.28, 42.0, 1.05, 0.55, 1.95, pr.Color(88, 79, 63, 255), -5.0),
        ]

    def CreateGrass(self):
        clumps = []
        for i in range(240):
            lane = -1 if i % 2 == 0 else 1
            near_curb = 1.3 + ((i * 17) % 90) / 40.0
            x = lane * near_curb
            z = -58.0 + ((i * 29) % 1160) / 10.0
            height = 0.22 + ((i * 13) % 26) / 58.0
            lean = (((i * 19) % 13) - 6) / 42.0
            color = self.Foliage if i % 4 else self.DryGrass
            clumps.append((x, z, height, lean, color))

        for i in range(90):
            side = -1 if i % 2 == 0 else 1
            x = side * (5.2 + ((i * 11) % 38) / 16.0)
            z = -60.0 + ((i * 31) % 1230) / 10.0
            height = 0.35 + ((i * 7) % 24) / 50.0
            lean = (((i * 23) % 15) - 7) / 38.0
            clumps.append((x, z, height, lean, self.DeepFoliage))
        return clumps

    def CreateTrees(self):
        trees = []
        for i, z in enumerate([-48.0, -33.0, -18.0, -2.0, 16.0, 31.0, 48.0]):
            side = -1 if i % 2 == 0 else 1
            trees.append((side * 7.0, z, 1.6 + (i % 3) * 0.25))
            trees.append((-side * 8.4, z + 7.0, 1.3 + ((i + 1) % 3) * 0.25))
        return trees

    def CreateStreetProps(self):
        props = []
        for i, z in enumerate([-52.0, -37.0, -22.0, -7.0, 8.0, 23.0, 38.0, 53.0]):
            side = -1 if i % 2 == 0 else 1
            props.append((side * 5.55, z, side))
        return props

    def Draw(self):
        self.DrawSky()
        self.DrawGround()
        self.DrawSkyline()
        self.DrawBuildings()
        self.DrawCars()
        self.DrawOvergrowth()
        self.DrawStreetDetails()
        self.DrawAtmosphere()

    def DrawSky(self):
        for z, width in [(-44.0, 70.0), (-68.0, 98.0), (-86.0, 124.0)]:
            rl.DrawCube(pr.Vector3(0.0, 16.6, z), width, 12.8, 0.08, self.SkyUpper)

        self.DrawCloud(-24.0, 9.0, -43.7, 18.0, 0.42)
        self.DrawCloud(16.0, 10.3, -43.7, 24.0, 0.38)
        self.DrawCloud(4.0, 6.4, -43.7, 30.0, 0.32)
        self.DrawCloud(-12.0, 12.0, -67.7, 32.0, 0.36)

    def DrawCloud(self, x, y, z, width, height):
        rl.DrawCube(pr.Vector3(x, y, z), width, height, 0.05, self.CloudLight)
        rl.DrawCube(
            pr.Vector3(x + width * 0.12, y - height * 0.55, z + 0.02),
            width * 0.72,
            height * 0.45,
            0.05,
            self.CloudShadow,
        )

    def DrawGround(self):
        rl.DrawPlane(pr.Vector3(0.0, -0.035, 0.0), pr.Vector2(88.0, 148.0), self.Ground)
        rl.DrawPlane(pr.Vector3(0.0, 0.0, 0.0), pr.Vector2(8.4, self.StreetLength), self.Street)
        rl.DrawPlane(pr.Vector3(-6.2, 0.012, 0.0), pr.Vector2(3.0, self.StreetLength), self.Sidewalk)
        rl.DrawPlane(pr.Vector3(6.2, 0.012, 0.0), pr.Vector2(3.0, self.StreetLength), self.Sidewalk)

        for x in [-4.35, 4.35]:
            rl.DrawCube(pr.Vector3(x, 0.035, 0.0), 0.18, 0.06, self.StreetLength, self.Curb)

        for z in range(-58, 62, 7):
            rl.DrawCube(pr.Vector3(0.0, 0.03, float(z)), 0.08, 0.035, 1.35, self.Lane)

        for side, z, width in self.CrossStreets:
            x = side * 11.5
            rl.DrawCube(pr.Vector3(x, 0.02, z), 14.0, 0.045, width, self.StreetPatch)

        for i, z in enumerate(range(-56, 58, 9)):
            x = -1.6 + (i % 4) * 1.05
            rl.DrawCube(
                pr.Vector3(x, 0.04, float(z)),
                0.55 + (i % 3) * 0.18,
                0.035,
                2.2,
                self.StreetPatch,
            )

    def DrawSkyline(self):
        for x, y, z, width, height, depth, color in self.Skyline:
            rl.DrawCube(pr.Vector3(x, y, z), width, height, depth, color)
            rl.DrawCubeWires(pr.Vector3(x, y, z), width, height, depth, pr.Color(43, 48, 48, 255))

    def DrawBuildings(self):
        for side, x, y, z, width, height, depth, color, roof in self.Buildings:
            rl.DrawCube(pr.Vector3(x, y, z), width, height, depth, color)
            rl.DrawCube(pr.Vector3(x, height + 0.11, z), width * 1.04, 0.22, depth * 1.03, roof)
            rl.DrawCubeWires(pr.Vector3(x, y, z), width, height, depth, pr.Color(45, 46, 42, 255))

            face_x = x - side * (width / 2.0 + 0.025)
            rl.DrawCube(
                pr.Vector3(face_x, 0.85, z),
                0.05,
                1.35,
                depth * 0.82,
                pr.Color(49, 48, 43, 255),
            )
            rl.DrawCube(
                pr.Vector3(face_x - side * 0.14, 1.55, z),
                0.24,
                0.16,
                depth * 0.72,
                pr.Color(108, 80, 45, 255),
            )

            floors = max(2, int(height / 1.15))
            columns = max(2, int(depth / 0.95))
            for floor in range(2, floors):
                wy = floor * 0.92
                for col in range(columns):
                    z_offset = -depth * 0.38 + col * (depth * 0.76 / max(1, columns - 1))
                    tint = self.WindowDark if (floor + col) % 4 else self.WindowDust
                    rl.DrawCube(pr.Vector3(face_x, wy, z + z_offset), 0.055, 0.28, 0.34, tint)

            if int(abs(z) + width) % 3 == 0:
                rl.DrawCube(
                    pr.Vector3(face_x - side * 0.08, 2.8, z - depth * 0.15),
                    0.08,
                    1.2,
                    0.18,
                    self.DeepFoliage,
                )
                rl.DrawCube(
                    pr.Vector3(face_x - side * 0.08, 2.1, z + depth * 0.15),
                    0.08,
                    1.6,
                    0.18,
                    self.DeepFoliage,
                )

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

        for x, z, height in self.Trees:
            rl.DrawCylinderEx(
                pr.Vector3(x, 0.05, z),
                pr.Vector3(x + 0.08, height, z + 0.05),
                0.055,
                0.035,
                7,
                pr.Color(58, 49, 35, 255),
            )
            rl.DrawSphere(pr.Vector3(x + 0.12, height + 0.35, z), 0.62, self.DeepFoliage)
            rl.DrawSphere(pr.Vector3(x - 0.28, height + 0.18, z + 0.18), 0.45, self.Foliage)

        for z in range(-52, 58, 11):
            rl.DrawCube(pr.Vector3(-5.3, 0.045, float(z)), 0.9, 0.06, 2.4, self.Foliage)
            rl.DrawCube(pr.Vector3(5.3, 0.045, float(z + 4)), 0.8, 0.06, 2.1, self.Foliage)

    def DrawStreetDetails(self):
        for x, z, side in self.StreetProps:
            rl.DrawCylinderEx(
                pr.Vector3(x, 0.0, z),
                pr.Vector3(x, 2.25, z),
                0.035,
                0.035,
                8,
                pr.Color(44, 46, 43, 255),
            )
            rl.DrawCube(pr.Vector3(x - side * 0.22, 2.08, z), 0.45, 0.16, 0.16, pr.Color(119, 88, 45, 255))
            if int(abs(z)) % 2 == 0:
                rl.DrawCube(pr.Vector3(x - side * 0.32, 1.45, z), 0.08, 0.75, 0.62, pr.Color(53, 48, 41, 255))

    def DrawAtmosphere(self):
        rl.DrawCube(pr.Vector3(0.0, 5.7, -45.0), 76.0, 8.5, 0.08, self.Haze)
        rl.DrawCube(pr.Vector3(0.0, 6.2, -64.0), 92.0, 10.5, 0.08, pr.Color(224, 188, 94, 62))
