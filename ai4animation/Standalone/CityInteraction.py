# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import pyray as pr
import raylib as rl
from ai4animation.Components.Component import Component


class CityInteraction(Component):
    def Start(self, params):
        self.City = params[0]
        self.DoorSpecs = self.City.GetDoorSpecs()
        self.Open = [False for _ in self.DoorSpecs]
        self.Progress = [0.0 for _ in self.DoorSpecs]
        self.PushTimer = 0.0
        self.InteractDistance = 2.15

        self.RoomFloor = pr.Color(69, 68, 61, 255)
        self.RoomBack = pr.Color(42, 43, 40, 255)
        self.RoomTrim = pr.Color(95, 83, 62, 255)
        self.DoorClosed = pr.Color(53, 43, 32, 255)
        self.DoorOpen = pr.Color(83, 64, 42, 255)
        self.Glass = pr.Color(65, 83, 82, 190)
        self.Highlight = pr.Color(182, 210, 185, 180)
        self.ItemFood = pr.Color(126, 116, 68, 255)
        self.ItemMedical = pr.Color(170, 62, 58, 255)
        self.ItemTool = pr.Color(73, 90, 110, 255)

    def Update(self):
        for i, is_open in enumerate(self.Open):
            target = 1.0 if is_open else 0.0
            self.Progress[i] += (target - self.Progress[i]) * min(1.0, 8.0 * rl.GetFrameTime())
        self.PushTimer = max(0.0, self.PushTimer - rl.GetFrameTime())

    def InteractWithActor(self, actor, pressed):
        if not pressed:
            return self.PushTimer > 0.0

        root = actor.GetRootPosition()
        nearest = self.GetNearestDoor(root)
        if nearest is None:
            return self.PushTimer > 0.0

        index, distance = nearest
        if distance <= self.InteractDistance:
            self.Open[index] = True
            self.PushTimer = 0.9
        return self.PushTimer > 0.0

    def GetNearestDoor(self, position):
        best = None
        for i, spec in enumerate(self.DoorSpecs):
            target = spec["interact"]
            delta = np.array([position[0] - target[0], position[2] - target[2]])
            distance = float(np.linalg.norm(delta))
            if best is None or distance < best[1]:
                best = (i, distance)
        return best

    def Draw(self):
        for index, spec in enumerate(self.DoorSpecs):
            self.DrawStorefront(index, spec)
            if self.Progress[index] > 0.02:
                self.DrawInterior(index, spec, self.Progress[index])

    def DrawStorefront(self, index, spec):
        side = spec["side"]
        face_x = spec["face_x"]
        z = spec["z"]
        street_x = face_x - side * 0.08
        trim_color = self.RoomTrim

        rl.DrawCube(pr.Vector3(street_x, 0.92, z), 0.09, 1.82, 1.08, self.RoomBack)
        rl.DrawCube(pr.Vector3(street_x - side * 0.03, 1.88, z), 0.12, 0.16, 1.25, trim_color)
        rl.DrawCube(pr.Vector3(street_x - side * 0.03, 0.04, z), 0.12, 0.08, 1.22, trim_color)

        open_amount = self.Progress[index]
        closed_width = max(0.02, 1.0 - open_amount)
        if closed_width > 0.03:
            rl.DrawCube(
                pr.Vector3(street_x - side * 0.05, 0.86, z),
                0.08,
                1.56,
                0.86 * closed_width,
                self.DoorClosed,
            )
            rl.DrawCube(
                pr.Vector3(street_x - side * 0.095, 1.14, z + 0.13 * closed_width),
                0.04,
                0.52,
                0.28 * closed_width,
                self.Glass,
            )

        if open_amount > 0.04:
            swing = 0.16 + open_amount * 0.58
            rl.DrawCube(
                pr.Vector3(street_x + side * swing, 0.86, z - 0.37),
                0.72 * open_amount,
                1.56,
                0.08,
                self.DoorOpen,
            )

    def DrawInterior(self, index, spec, open_amount):
        side = spec["side"]
        face_x = spec["face_x"]
        z = spec["z"]
        room_x = face_x - side * 0.16
        width = max(1.8, spec["depth"] * 0.74)

        rl.DrawCube(pr.Vector3(room_x, 0.02, z), 1.95, 0.05, width, self.RoomFloor)
        rl.DrawCube(pr.Vector3(room_x + side * 0.88, 1.1, z), 0.08, 2.1, width, self.RoomBack)

        style = spec["style"]
        if style == 0:
            self.DrawGrocery(room_x, z, side, width)
        elif style == 1:
            self.DrawDiner(room_x, z, side, width)
        elif style == 2:
            self.DrawCabinets(room_x, z, side, width)
        else:
            self.DrawBathStorage(room_x, z, side, width)

        rl.DrawCube(
            pr.Vector3(room_x - side * 0.08, 1.84, z),
            0.06,
            0.18,
            width * 0.9,
            pr.Color(38, 37, 34, int(190 * open_amount)),
        )

    def DrawGrocery(self, x, z, side, width):
        for offset in [-0.38, 0.0, 0.38]:
            rl.DrawCube(pr.Vector3(x + side * 0.42, 0.72, z + offset * width), 0.38, 1.05, 0.18, self.RoomTrim)
            rl.DrawCube(pr.Vector3(x + side * 0.25, 0.95, z + offset * width), 0.18, 0.16, 0.16, self.ItemFood)

    def DrawDiner(self, x, z, side, width):
        for offset in [-0.28, 0.28]:
            rl.DrawCube(pr.Vector3(x + side * 0.28, 0.42, z + offset * width), 0.52, 0.08, 0.46, self.RoomTrim)
            rl.DrawCube(pr.Vector3(x + side * 0.28, 0.22, z + offset * width), 0.08, 0.36, 0.08, self.RoomTrim)
            rl.DrawCube(pr.Vector3(x - side * 0.15, 0.28, z + offset * width), 0.24, 0.42, 0.34, pr.Color(61, 55, 48, 255))

    def DrawCabinets(self, x, z, side, width):
        for offset in [-0.36, 0.0, 0.36]:
            rl.DrawCube(pr.Vector3(x + side * 0.45, 0.62, z + offset * width), 0.32, 1.0, 0.28, pr.Color(92, 78, 58, 255))
            rl.DrawCube(pr.Vector3(x - side * 0.05, 0.35, z + offset * width), 0.22, 0.2, 0.22, self.ItemTool)

    def DrawBathStorage(self, x, z, side, width):
        rl.DrawCube(pr.Vector3(x + side * 0.45, 0.42, z - width * 0.25), 0.42, 0.5, 0.34, pr.Color(159, 156, 139, 255))
        rl.DrawCube(pr.Vector3(x + side * 0.44, 0.25, z + width * 0.22), 0.5, 0.2, 0.42, pr.Color(121, 124, 112, 255))
        rl.DrawCube(pr.Vector3(x - side * 0.04, 0.42, z), 0.24, 0.22, 0.24, self.ItemMedical)
