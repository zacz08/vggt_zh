import os
import time
import threading
from typing import List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
import viser
import viser.transforms as viser_tf
import cv2

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from utils.visual_util import segment_sky, download_file_from_url
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map


# Global reusable state for single-page visualization
_STATE = {
    "server": None,
    "point_cloud": None,
    "frames": [],
    "frustums": [],
    "gui": {},                  # store GUI handles
    "bg_thread_started": False,
    "data": {}                 # cached arrays for current scene
}


def _extract_scene_data(
    pred_dict: dict,
    init_conf_threshold: float,
    use_point_map: bool,
    mask_sky: bool,
    raw_images: torch.Tensor,
    extrinsics_in_ego_frame: bool,
    cfg: OmegaConf
):
    """
    Compute points/colors/conf/frame indices and camera poses from pred_dict.
    Returns a dict with all arrays needed by the renderer.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        image_folder (str): Path to the folder containing input images.
        cfg (OmegaConf): Configuration for the dataset and BEV
    """

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    points_conf = pred_dict["world_points_conf"]  # (S, H, W)

    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)

    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # Compute world points from depth if not using the precomputed point map
    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = points_conf

    # Apply sky segmentation if enabled
    if mask_sky:
        conf = apply_sky_segmentation(conf, raw_images)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    # colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    colors = images.permute(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    # colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    colors_flat = (colors.reshape(-1, 3) * 255).to(torch.uint8)
    conf_flat = conf.reshape(-1)
    # frame_indices = np.repeat(np.arange(S), H * W)
    frame_indices = torch.arange(S, device=points.device).repeat_interleave(H * W)

    ## Filter points within BEV bounds
    x_min, x_max = cfg.lift.x_bound[0:2]
    y_min, y_max = cfg.lift.y_bound[0:2]
    z_min, z_max = cfg.lift.z_bound[0:2]

    bev_mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    points = points[bev_mask]
    colors_flat = colors_flat[bev_mask]
    conf_flat = conf_flat[bev_mask]
    frame_indices = frame_indices[bev_mask]

    # For convenience, we store only (3,4) portion
    if extrinsics_in_ego_frame:
        cam_to_world = extrinsics_cam[:, :3, :]
        points_centered = points
    else:
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)[:, :3, :]
        # Compute scene center and recenter
        scene_center = np.mean(points, axis=0)
        points_centered = points - scene_center
        cam_to_world[..., -1] -= scene_center

    return {
        "S": S, "H": H, "W": W,
        "images": images,  # for frustum textures
        "points_centered": points_centered,
        "colors_flat": colors_flat,
        "conf_flat": conf_flat,
        "frame_indices": frame_indices,
        "cam_to_world": cam_to_world
    }


def _rebuild_frames_and_frustums(server, scenes_data: List) -> None:
    """Remove old frames/frustums and rebuild them from current data."""
    # Clear any existing frames or frustums
    for f in _STATE["frames"]:
        f.remove()
    for fr in _STATE["frustums"]:
        fr.remove()
    _STATE["frames"].clear()
    _STATE["frustums"].clear()

    S = scenes_data["S"]
    images = scenes_data["images"]
    cam_to_world = scenes_data["cam_to_world"]

    # Optionally attach a callback that sets the viewpoint to the chosen camera
    def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
        @frustum.on_click
        def _(_) -> None:
            for client in server.get_clients().values():
                client.camera.wxyz = frame.wxyz
                client.camera.position = frame.position

    for img_id in range(S):
        T_world_camera = viser_tf.SE3.from_matrix(cam_to_world[img_id].cpu().numpy())

        # Add a small frame axis
        frame_axis = server.scene.add_frame(
            f"frame_{img_id}",
            wxyz=T_world_camera.rotation().wxyz,
            position=T_world_camera.translation(),
            axes_length=0.05,
            axes_radius=0.002,
            origin_radius=0.002,
        )
        _STATE["frames"].append(frame_axis)

        # Convert the image for the frustum
        img = images[img_id]  # shape (3, H, W)
        img = (img.permute(1, 2, 0) * 255).contiguous().cpu().numpy().astype(np.uint8)
        h, w = img.shape[:2]

        # If you want correct FOV from intrinsics, do something like:
        # fx = intrinsics_cam[img_id, 0, 0]
        # fov = 2 * np.arctan2(h/2, fx)
        # For demonstration, we pick a simple approximate FOV:
        fy = 1.1 * h
        fov = 2 * np.arctan2(h / 2, fy)

        # Add the frustum
        frustum_cam = server.scene.add_camera_frustum(
            f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
        )
        _STATE["frustums"].append(frustum_cam)
        attach_callback(frustum_cam, frame_axis)

def _apply_gui_and_update_pointcloud():
    """Use current GUI states to filter and update the existing point cloud."""
    server = _STATE["server"]
    pc = _STATE["point_cloud"]
    gui = _STATE["gui"]
    data = _STATE["data"]

    if server is None or pc is None or not data:
        return

    conf_flat = data["conf_flat"].cpu().numpy()
    frame_indices = data["frame_indices"].cpu().numpy()
    points_centered = data["points_centered"].cpu().numpy()
    colors_flat = data["colors_flat"].cpu().numpy()

    percentage = gui["conf_slider"].value
    threshold_val = np.percentile(conf_flat, percentage)
    conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

    if gui["frame_selector"].value == "All":
        frame_mask = np.ones_like(conf_mask, dtype=bool)
    else:
        selected_idx = int(gui["frame_selector"].value)
        frame_mask = frame_indices == selected_idx

    mask = conf_mask & frame_mask
    pc.points = points_centered[mask]
    pc.colors = colors_flat[mask]


def _apply_colormap_jet(x01: np.ndarray) -> np.ndarray:
    """x01: [H,W] in [0,1] -> uint8 RGB heatmap"""
    x = np.clip(x01, 0.0, 1.0)
    # Simple jet approximation
    r = np.clip(1.5 - np.abs(4*x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4*x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4*x - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


# --------------------------------- public APIs ---------------------------------

def viser_init(pred_dict: dict,
               *,
               port: int = 8080,
               init_conf_threshold: float = 50.0,
               use_point_map: bool = False,
               background_mode: bool = False,
               mask_sky: bool = False,
               raw_images: torch.Tensor = None,
               extrinsics_in_ego_frame: bool = True,
               cfg: OmegaConf = None
               ) -> viser.ViserServer:
    """
    Initialize a single Viser server and build the scene ONCE.
    Later calls should use `viser_update(...)`.
    """
    if _STATE["server"] is not None:
        # Already initialized; just update with new data
        return viser_update(
            pred_dict,
            use_point_map=use_point_map,
            mask_sky=mask_sky,
            raw_images=raw_images,
            extrinsics_in_ego_frame=extrinsics_in_ego_frame,
            cfg=cfg,
        )

    print(f"[viser_init] Starting viser server on port {port}")
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    _STATE["server"] = server

    # compute arrays and cache
    data = _extract_scene_data(
        pred_dict,
        init_conf_threshold=init_conf_threshold,
        use_point_map=use_point_map,
        mask_sky=mask_sky,
        raw_images=raw_images,
        extrinsics_in_ego_frame=extrinsics_in_ego_frame,
        cfg=cfg
    )
    _STATE["data"] = data

    # GUI (existing)
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)
    gui_points_conf = server.gui.add_slider("Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold)
    gui_frame_selector = server.gui.add_dropdown("Show Points from Frames", options=["All"] + [str(i) for i in range(data["S"])], initial_value="All")
    _STATE["gui"] = {
        "show_frames": gui_show_frames,
        "conf_slider": gui_points_conf,
        "frame_selector": gui_frame_selector,
    }

    # initial point cloud (percentile mask)
    init_threshold_val = np.percentile(data["conf_flat"].cpu(), init_conf_threshold)
    init_conf_mask = (data["conf_flat"] >= init_threshold_val) & (data["conf_flat"] > 0.1)
    pc = server.scene.add_point_cloud(
        name="viser_pcd",
        points=data["points_centered"][init_conf_mask].cpu().numpy(),
        colors=data["colors_flat"][init_conf_mask].cpu().numpy(),
        point_size=0.02,
        point_shape="circle",
    )
    _STATE["point_cloud"] = pc

    # frames + frustums
    _rebuild_frames_and_frustums(server, data)

    # GUI callbacks (existing)
    @gui_points_conf.on_update
    def _(_evt) -> None:
        _apply_gui_and_update_pointcloud()

    @gui_frame_selector.on_update
    def _(_evt) -> None:
        _apply_gui_and_update_pointcloud()

    @gui_show_frames.on_update
    def _(_evt) -> None:
        """Toggle visibility of camera frames and frustums."""
        for f in _STATE["frames"]:
            f.visible = gui_show_frames.value
        for fr in _STATE["frustums"]:
            fr.visible = gui_show_frames.value

    # # Add the camera frames to the scene
    # visualize_frames(cam_to_world, images)

    print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode and not _STATE["bg_thread_started"]:
        def server_loop():
            while True:
                time.sleep(0.01)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
        _STATE["bg_thread_started"] = True

    return server


def viser_update(
    pred_dict: dict,
    *,
    init_conf_threshold: float = 50.0,
    use_point_map: bool = False,
    mask_sky: bool = False,
    raw_images: torch.Tensor = None,
    extrinsics_in_ego_frame: bool = True,
    cfg: OmegaConf = None,
):
    """
    Update the SAME page with new data. Assumes `viser_init` was called before.
    """
    if _STATE["server"] is None:
        raise RuntimeError("viser_update called before viser_init.")

    server = _STATE["server"]


    # recompute arrays
    data = _extract_scene_data(
        pred_dict,
        init_conf_threshold=init_conf_threshold,
        use_point_map=use_point_map,
        mask_sky=mask_sky,
        raw_images=raw_images,
        extrinsics_in_ego_frame=extrinsics_in_ego_frame,
        cfg=cfg
    )
    _STATE["data"] = data

    # refresh frame selector options if S changes
    gui = _STATE["gui"]
    new_options = ["All"] + [str(i) for i in range(data["S"])]
    if gui["frame_selector"].options != new_options:
        gui["frame_selector"].options = new_options
        gui["frame_selector"].value = "All"

    # rebuild frames/frustums and update PC using current GUI filters
    _rebuild_frames_and_frustums(server, data)
    _apply_gui_and_update_pointcloud()

    return server


# --- Backward compatible wrapper: keep old API but reuse the same page ---
def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    raw_images: torch.Tensor = None,
    extrinsics_in_ego_frame: bool = True,
    cfg: OmegaConf = None,
):
    """
    Backward-compatible: first call initializes; subsequent calls update in the same page.
    """
    if _STATE["server"] is None:
        return viser_init(
            pred_dict,
            port=port,
            init_conf_threshold=init_conf_threshold,
            use_point_map=use_point_map,
            background_mode=background_mode,
            mask_sky=mask_sky,
            raw_images=raw_images,
            extrinsics_in_ego_frame=extrinsics_in_ego_frame,
            cfg=cfg,
        )
    else:
        return viser_update(
            pred_dict,
            use_point_map=use_point_map,
            mask_sky=mask_sky,
            raw_images=raw_images,
            extrinsics_in_ego_frame=extrinsics_in_ego_frame,
            cfg=cfg,
        )


# Helper functions for sky segmentation
def apply_sky_segmentation(conf, images: torch.Tensor):
    """
    Apply sky segmentation to confidence scores.

    Args:
        conf (np.ndarray or torch.Tensor): Confidence scores with shape (S, H, W)
        images (torch.Tensor): Input RGB images [B, 3, H, W], dtype=float32, range [0, 255]

    Returns:
        np.ndarray or torch.Tensor: Updated confidence scores with sky regions masked out
    """
    if images.ndim == 5:
        # [B, 6, C, H, W] → merge first two dimensions
        B = images.shape[0] * images.shape[1]
        C, H, W = images.shape[2:]
        images = images.view(B, C, H, W)
    else:
        B, C, _, _ = images.shape
    assert C == 3, "Input images must have 3 channels (RGB)"

    _, H, W = conf.shape

    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    # image_files = sorted(glob.glob(os.path.join(image_folder, "*")))

    # print("Generating sky masks from tensors...")
    sky_mask_list = []
    for i in range(B):
        image_np = images[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # (H, W, 3)
        sky_mask = segment_sky(image_np, skyseg_session)

        # Resize mask to match H×W if needed
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))
        sky_mask_list.append(sky_mask)

    # Apply sky mask to confidence scores
    sky_mask_binary = np.array(sky_mask_list) > 0.1
    if isinstance(conf, np.ndarray):
        mask = sky_mask_binary.astype(conf.dtype, copy=False)
    elif isinstance(conf, torch.Tensor):
        mask = torch.from_numpy(sky_mask_binary).to(device=conf.device, dtype=conf.dtype)
    else:
        raise TypeError(f"Unsupported conf type: {type(conf)}")

    return conf * mask
