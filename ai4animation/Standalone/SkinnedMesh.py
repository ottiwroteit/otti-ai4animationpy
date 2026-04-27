# Copyright (c) Meta Platforms, Inc. and affiliates.
from array import array
from io import BytesIO
from sys import platform

import cffi
import numpy as np
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Math import Tensor
from pyray import Matrix, Mesh, load_model_from_mesh
from raylib import (
    LoadImageFromMemory,
    LoadTextureFromImage,
    LIGHTGRAY,
    MATERIAL_MAP_DIFFUSE,
    MatrixIdentity,
    MemAlloc,
    RAYWHITE,
    SetMaterialTexture,
    UnloadImage,
    UpdateMeshBuffer,
    UploadMesh,
    WHITE,
)

ffi = cffi.FFI()


def _create_texture_from_image(image):
    if image is None:
        return None

    encoded = BytesIO()
    image.convert("RGBA").save(encoded, format="PNG")
    image_bytes = encoded.getvalue()

    raylib_image = LoadImageFromMemory(
        b".png",
        ffi.from_buffer("unsigned char[]", image_bytes),
        len(image_bytes),
    )
    texture = LoadTextureFromImage(raylib_image)
    UnloadImage(raylib_image)
    return texture


class SkinnedMesh:
    def __init__(self, actor, model):
        self.Actor = actor

        self.SkinnedMeshes = [mesh for mesh in model.Meshes if mesh.HasSkinning]

        self.BindMatrices = np.transpose(
            Tensor.Create(model.Skin.Inverse_bind_matrices), axes=(0, 2, 1)
        )

        self.Models = []
        self.BoneMatrixViews = []
        self.MeshBuffers = []
        self.CpuSkinningMeshes = []
        self.UseCpuSkinning = platform == "darwin"
        self.Textures = []
        self.Color = (
            LIGHTGRAY
            if all(
                getattr(mesh, "Image", None) is None
                or getattr(mesh.Image, "size", None) == (1, 1)
                for mesh in self.SkinnedMeshes
            )
            else RAYWHITE
        )

        print(
            f"Loading {len(self.SkinnedMeshes)} skinned meshes (skipping {len(model.Meshes) - len(self.SkinnedMeshes)} non-skinned meshes)"
        )

        boneCount = len(model.JointNames)
        self.BoneCount = boneCount

        MAX_BONES_SUPPORTED = 254
        if boneCount > MAX_BONES_SUPPORTED:
            raise ValueError(
                f"Character has {boneCount} bones, but shader only supports {MAX_BONES_SUPPORTED}. "
                f"Increase MAX_BONE_NUM in skinnedShadow.vs and skinnedBasic.vs"
            )

        for mesh in self.SkinnedMeshes:
            vertexCount = len(mesh.Vertices)

            # Create Raylib mesh for this mesh
            vertices = array("f", mesh.Vertices.flatten())
            normals = array("f", mesh.Normals.flatten())
            triangles = array("H", mesh.Triangles.flatten().astype(np.uint16))
            colors = array("B", [255, 255, 255, 255] * vertexCount)
            if getattr(mesh, "TexCoords", None) is not None and len(mesh.TexCoords) == vertexCount:
                texcoords = array("f", np.asarray(mesh.TexCoords, dtype=np.float32).flatten())
            else:
                texcoords = array("f", [0.5, 0.5] * vertexCount)

            # 4 bones per vertex
            boneIds = np.zeros((vertexCount, 4), dtype=np.uint8)
            currentSkinBones = min(mesh.SkinIndices.shape[1], 4)
            boneIds[:, :currentSkinBones] = mesh.SkinIndices[
                :, :currentSkinBones
            ].astype(np.uint8)
            bone_ids = array("B", boneIds.flatten())

            # Bone weights
            boneWeights = np.zeros((vertexCount, 4), dtype=np.float32)
            boneWeights[:, :currentSkinBones] = mesh.SkinWeights[:, :currentSkinBones]
            bone_weights = array("f", boneWeights.flatten())

            raylib_mesh = Mesh()
            raylib_mesh.vertexCount = vertexCount
            raylib_mesh.triangleCount = int(len(triangles) / 3)
            raylib_mesh.vertices = ffi.cast("float*", vertices.buffer_info()[0])
            raylib_mesh.texcoords = ffi.cast("float*", texcoords.buffer_info()[0])
            raylib_mesh.normals = ffi.cast("float*", normals.buffer_info()[0])
            raylib_mesh.colors = ffi.cast("unsigned char*", colors.buffer_info()[0])
            raylib_mesh.indices = ffi.cast(
                "unsigned short*", triangles.buffer_info()[0]
            )
            raylib_mesh.boneIds = ffi.cast("unsigned char*", bone_ids.buffer_info()[0])
            raylib_mesh.boneWeights = ffi.cast("float*", bone_weights.buffer_info()[0])
            raylib_mesh.boneCount = boneCount
            raylib_mesh.vaoId = 0

            # Allocate bone matrices
            raylib_mesh.boneMatrices = MemAlloc(boneCount * ffi.sizeof(Matrix()))
            for i in range(boneCount):
                raylib_mesh.boneMatrices[i] = MatrixIdentity()

            # Upload mesh with dynamic flag for bone updates
            UploadMesh(ffi.addressof(raylib_mesh), True)
            self.MeshBuffers.append(
                (vertices, texcoords, normals, colors, triangles, bone_ids, bone_weights)
            )

            # Create Model for this mesh
            raylib_model = load_model_from_mesh(raylib_mesh)
            raylib_model.materials[0].maps[MATERIAL_MAP_DIFFUSE].color = WHITE

            texture = _create_texture_from_image(getattr(mesh, "Image", None))
            if texture is not None:
                SetMaterialTexture(
                    ffi.addressof(raylib_model.materials[0]),
                    MATERIAL_MAP_DIFFUSE,
                    texture,
                )
                self.Textures.append(texture)

            self.Models.append(raylib_model)

            # Cache numpy view of bone matrices for efficient updates
            gpu_mesh = raylib_model.meshes[0]
            matView = np.frombuffer(
                ffi.buffer(gpu_mesh.boneMatrices, gpu_mesh.boneCount * ffi.sizeof(Matrix())),
                dtype=np.float32,
            ).reshape(gpu_mesh.boneCount, 4, 4)
            self.BoneMatrixViews.append(matView)
            self.CpuSkinningMeshes.append(
                {
                    "mesh": gpu_mesh,
                    "vertexPositions": np.concatenate(
                        (
                            np.asarray(mesh.Vertices, dtype=np.float32),
                            np.ones((vertexCount, 1), dtype=np.float32),
                        ),
                        axis=1,
                    ),
                    "normals": np.asarray(mesh.Normals, dtype=np.float32),
                    "boneIds": boneIds,
                    "boneWeights": boneWeights,
                    "skinnedVertices": np.empty((vertexCount, 3), dtype=np.float32),
                    "skinnedNormals": np.empty((vertexCount, 3), dtype=np.float32),
                }
            )

        print(
            f"Initialized {len(self.Models)} skinned submeshes with {boneCount} bones"
        )

        AI4Animation.Standalone.RenderPipeline.RegisterModel(
            name=self.Actor.Entity.Name,
            model=self.Models,
            skinned_mesh=self,
            color=self.Color,
        )

    def SetColor(self, color):
        self.Color = color
        self.Unregister()
        self.Register()

    def Register(self):
        if not AI4Animation.Standalone.RenderPipeline.HasModel(self.Models):
            AI4Animation.Standalone.RenderPipeline.RegisterModel(
                name=self.Actor.Entity.Name,
                model=self.Models,
                skinned_mesh=self,
                color=self.Color,
            )

    def Unregister(self):
        if AI4Animation.Standalone.RenderPipeline.HasModel(self.Models):
            AI4Animation.Standalone.RenderPipeline.UnregisterModel(self.Models)

    def Update(self):
        # GPU skinning - compute and update bone matrices
        if not self.Models:
            return

        # Update bone matrices for all meshes (GPU will use these in shaders)
        transforms = np.matmul(
            AI4Animation.Scene.GetSkinningTransforms(self.Actor.Entities),
            self.BindMatrices,
        )
        for matView in self.BoneMatrixViews:
            matView[:] = transforms

        if self.UseCpuSkinning:
            self.UpdateCpuSkinnedMeshes(transforms)

    def UpdateCpuSkinnedMeshes(self, transforms):
        for mesh_data in self.CpuSkinningMeshes:
            bone_ids = mesh_data["boneIds"]
            bone_weights = mesh_data["boneWeights"]
            vertex_positions = mesh_data["vertexPositions"]
            normals = mesh_data["normals"]

            skinned_vertices = mesh_data["skinnedVertices"]
            skinned_normals = mesh_data["skinnedNormals"]
            skinned_vertices.fill(0.0)
            skinned_normals.fill(0.0)

            for i in range(4):
                weights = bone_weights[:, i : i + 1]
                active = weights[:, 0] > 0.0
                if not np.any(active):
                    continue

                matrices = transforms[bone_ids[active, i]]
                skinned_vertices[active] += (
                    np.einsum("nij,nj->ni", matrices, vertex_positions[active])[:, :3]
                    * weights[active]
                )
                skinned_normals[active] += (
                    np.einsum("nij,nj->ni", matrices[:, :3, :3], normals[active])
                    * weights[active]
                )

            lengths = np.linalg.norm(skinned_normals, axis=1, keepdims=True)
            np.divide(
                skinned_normals,
                lengths,
                out=skinned_normals,
                where=lengths > 1e-6,
            )

            raylib_mesh = mesh_data["mesh"]
            UpdateMeshBuffer(
                raylib_mesh,
                0,
                ffi.from_buffer("float[]", skinned_vertices),
                skinned_vertices.nbytes,
                0,
            )
            UpdateMeshBuffer(
                raylib_mesh,
                2,
                ffi.from_buffer("float[]", skinned_normals),
                skinned_normals.nbytes,
                0,
            )
