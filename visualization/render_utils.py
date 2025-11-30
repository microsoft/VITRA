# Render utils for PyTorch3D
# Adapted and improved from: https://github.com/ThunderVVV/HaWoR/blob/main/lib/vis/renderer.py
import torch
import numpy as np
from typing import List, Tuple, Union

from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRenderer, 
    MeshRasterizer, 
    SoftPhongShader,
    RasterizationSettings, 
    PointLights, 
    TexturesVertex
)

from pytorch3d.structures import Meshes
from pytorch3d.renderer.camera_conversions import _cameras_from_opencv_projection

def update_intrinsics_from_bbox(
    K_org: torch.Tensor, bbox: torch.Tensor
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Update intrinsic matrix K according to the given bounding box.
    
    Args:
        K_org (torch.Tensor): Original intrinsic matrix of shape (B, 3, 3).
        bbox (torch.Tensor): Bounding boxes of shape (B, 4) in (left, top, right, bottom) format.
    
    Returns:
        K_new (torch.Tensor): Updated intrinsics with shape (B, 4, 4).
        image_sizes (List[Tuple[int, int]]): List of image sizes (height, width) for each bbox.
    """
    device, dtype = K_org.device, K_org.dtype

    # Initialize 4x4 intrinsic matrix
    K_new = torch.zeros((K_org.shape[0], 4, 4), device=device, dtype=dtype)
    K_new[:, :3, :3] = K_org.clone()
    K_new[:, 2, 2] = 0
    K_new[:, 2, -1] = 1
    K_new[:, -1, 2] = 1

    image_sizes = []
    for idx, box in enumerate(bbox):
        left, top, right, bottom = box
        cx, cy = K_new[idx, 0, 2], K_new[idx, 1, 2]

        # Adjust principal point according to bbox
        new_cx = cx - left
        new_cy = cy - top

        # Compute new width and height
        new_height = max(bottom - top, 1)
        new_width = max(right - left, 1)

        # Flip principal point coordinates if needed
        new_cx = new_width - new_cx
        new_cy = new_height - new_cy

        K_new[idx, 0, 2] = new_cx
        K_new[idx, 1, 2] = new_cy

        image_sizes.append((int(new_height), int(new_width)))

    return K_new, image_sizes

class Renderer():
    """
    Renderer class using PyTorch3D for mesh rendering with Phong shading.
    
    Attributes:
        width (int): Target image width.
        height (int): Target image height.
        focal_length (Union[float, Tuple[float, float]]): Camera focal length(s).
        device (torch.device): Device to run rendering on.
        renderer (MeshRenderer): PyTorch3D mesh renderer.
        cameras (PerspectiveCameras): Camera object.
        lights (PointLights): Lighting setup for rendering.
    """
    def __init__(
        self,
        width: int,
        height: int,
        focal_length: Union[float, Tuple[float, float]],
        device: torch.device,
        bin_size: int = 512,
        max_faces_per_bin: int = 200000,
    ):

        self.width = width
        self.height = height
        self.focal_length = focal_length
        self.device = device

        # Initialize camera parameters
        self._initialize_camera_params()

        # Set up lighting
        self.lights = PointLights(
            device=device,
            location = ((0.0, -1.5, -1.5),),
            ambient_color=((0.75, 0.75, 0.75),), 
            diffuse_color=((0.25, 0.25, 0.25),), 
            specular_color=((0.02, 0.02, 0.02),) 
                    )
        
        # Initialize renderer
        self._create_renderer(bin_size, max_faces_per_bin)

    def _create_renderer(self, bin_size: int, max_faces_per_bin: int):
        """
        Create the PyTorch3D MeshRenderer with rasterizer and shader.
        """
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=RasterizationSettings(
                    image_size=self.image_sizes[0],
                    blur_radius=1e-5,
                    bin_size=bin_size,
                    max_faces_per_bin=max_faces_per_bin,
                )
            ),
            shader=SoftPhongShader(
                device=self.device,
                lights=self.lights,
            ),
        )

    def _initialize_camera_params(self):
        """
        Initialize camera intrinsics and extrinsics.
        """
        # Extrinsics (identity rotation and zero translation)
        self.R = torch.eye(3, device=self.device).unsqueeze(0)
        self.T = torch.zeros(1, 3, device=self.device)

        # Intrinsics
        if isinstance(self.focal_length, (list, tuple)):
            fx, fy = self.focal_length
        else:
            fx = fy = self.focal_length

        self.K = torch.tensor(
            [[fx, 0, self.width / 2],
             [0, fy, self.height / 2],
             [0, 0, 1]],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)

        self.bboxes = torch.tensor([[0, 0, self.width, self.height]], dtype=torch.float32)
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)

        # Create PyTorch3D cameras
        self.cameras = self._create_camera_from_cv()

    def _create_camera_from_cv(
        self,
        R: torch.Tensor = None,
        T: torch.Tensor = None,
        K: torch.Tensor = None,
        image_size: torch.Tensor = None,
    ) -> PerspectiveCameras:
        """
        Create a PyTorch3D camera from OpenCV-style intrinsics and extrinsics.
        """
        if R is None:
            R = self.R
        if T is None:
            T = self.T
        if K is None:
            K = self.K
        if image_size is None:
            image_size = torch.tensor(self.image_sizes, device=self.device)

        cameras = _cameras_from_opencv_projection(R, T, K, image_size)
        return cameras
    
    def render(
        self,
        verts_list: List[torch.Tensor],
        faces_list: List[torch.Tensor],
        colors_list: List[torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render a batch of meshes into an RGB image and mask.

        Args:
            verts_list (List[torch.Tensor]): List of vertex tensors.
            faces_list (List[torch.Tensor]): List of face tensors.
            colors_list (List[torch.Tensor]): List of per-vertex color tensors.

        Returns:
            rend (np.ndarray): Rendered RGB image as uint8 array.
            mask (np.ndarray): Boolean mask of rendered pixels.
        """
        all_verts = []
        all_faces = []
        all_colors = []
        vertex_offset = 0

        for verts, faces, colors in zip(verts_list, faces_list, colors_list):
            all_verts.append(verts)
            all_colors.append(colors)
            all_faces.append(faces + vertex_offset)  # Offset face indices
            vertex_offset += verts.shape[0]

        # Combine all meshes into a single mesh for rendering
        all_verts = torch.cat(all_verts, dim=0)
        all_faces = torch.cat(all_faces, dim=0)
        all_colors = torch.cat(all_colors, dim=0)

        mesh = Meshes(
            verts=[all_verts],  # batch_size=1
            faces=[all_faces],
            textures=TexturesVertex(all_colors.unsqueeze(0)),
        )

        # Render the image
        images = self.renderer(mesh, cameras=self.cameras, lights=self.lights)

        rend = np.clip(images[0, ..., :3].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        mask = images[0, ..., -1].cpu().numpy() > 0

        return rend, mask