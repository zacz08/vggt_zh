import torch

def transform_pointmap_to_current_ego(
    pointmap,        # [B, N, H, W, 3], VGGT predicted pointmaps (per batch, per camera)
    extrinsic_pred,  # [B, N, 3, 4], predicted camera extrinsics (world -> cam)
    extrinsic_gt,    # [B, seq_len, num_cam, 4, 4], GT camera extrinsics (cam -> ego)
    ego_pose,        # [B, seq_len, 4, 4], GT ego poses (ego -> global)
    num_cam = 6      # number of cameras per frame
):
    """
    Align VGGT-predicted pointmaps into the current ego frame coordinates (batched version).

    Args:
        pointmap: [B, N, H, W, 3], VGGT-predicted pointmaps in the reference system
                  (before alignment, N = seq_len * num_cam).
        extrinsic_pred: [B, N, 3, 4], predicted camera extrinsics from VGGT (world -> cam).
        extrinsic_gt: [B, seq_len, num_cam, 4, 4], ground-truth camera extrinsics (cam -> ego).
        ego_pose: [B, seq_len, 4, 4], ground-truth ego poses (ego -> global).
        num_cam: int, number of cameras per frame.

    Returns:
        pointmap_last_ego: [B, N, H, W, 3], pointmaps transformed into the last ego frame coordinates.
    """

    # ======================= Process Transformations =======================
    B, N, H, W, _ = pointmap.shape
    seq_len = N // num_cam

    # 1. Predicted camera centers C_pred
    # extrinsic_pred: [B, N, 4, 4], world-> cam, 第1帧第1相机为单位阵，其余是相对预测）
    R_pred = extrinsic_pred[:, :, :3, :3]          # world -> cam rotation
    t_pred = extrinsic_pred[:, :, :3, 3]           # world -> cam translation
    C_pred = (-R_pred.transpose(-2, -1) @ t_pred.unsqueeze(-1)).squeeze(-1)  # [B, N, 3]

    # 2. Ground-truth camera centers C_gt
    # extrinsic_gt : cam -> ego
    # ego_pose: ego -> global
    ego_pose_exp = ego_pose[:, :, None, :, :]             # [B, seq_len, 1, 4, 4]
    ego_pose_exp = ego_pose_exp.expand(-1, -1, num_cam, -1, -1)  # [B, seq_len, num_cam, 4, 4]

    # batch matrix multiply: (ego->global) @ (cam->ego) = cam->global
    T_gt = torch.matmul(ego_pose_exp, extrinsic_gt)       # [B, seq_len, num_cam, 4, 4]

    # camera centers in global: take translation part
    C_gt = T_gt[:, :, :, :3, 3]                          # [B, seq_len, num_cam, 3]
    C_gt = C_gt.reshape(B, seq_len * num_cam, 3)         # [B,N,3]

    # 3. Sim(3) alignment for each batch
    T_sim3 = umeyama_alignment_3d(C_pred, C_gt, with_scale=True)


    # 4. Apply Sim(3) to pointmaps
    # pointmap = predictions["world_points"][0]
    pointmap_pred_reg = apply_sim3_transform(pointmap, T_sim3)

    # 5. Transform to the last ego frame
    T_world2ego = torch.linalg.inv(ego_pose[:, -1])   # 当前ego frame的变换
    pointmap_last_ego = apply_sim3_transform(pointmap_pred_reg, T_world2ego)

    return pointmap_last_ego



def transform_extrinsics_to_current_ego(
    sensor_to_ego,   # [seq_len, num_cam, 4, 4]  # 即nuscenes原始extrinsic
    ego_to_global    # [seq_len, 4, 4]           # nuscenes中的ego pose
):
    """
    把所有历史帧的所有相机的外参都变到最后一帧ego frame
    返回: [seq_len * num_cam, 4, 4]，每个相机在最后一帧ego frame下的外参
    """
    seq_len, num_cam = sensor_to_ego.shape[:2]
    device = sensor_to_ego.device

    T_last_ego_global = ego_to_global[-1]               # [4, 4]
    T_global_last_ego = torch.linalg.inv(T_last_ego_global)

    all_extrinsics = []
    for t in range(seq_len):
        T_ego_global = ego_to_global[t]
        for c in range(num_cam):
            T_cam_ego = sensor_to_ego[t, c]
            # 当前相机->global
            T_cam_global = T_ego_global @ T_cam_ego
            # global->最后一帧ego
            T_cam_last_ego = T_global_last_ego @ T_cam_global
            all_extrinsics.append(T_cam_last_ego)
    all_extrinsics = torch.stack(all_extrinsics, dim=0) # [seq_len*num_cam, 4, 4]
    return all_extrinsics



def umeyama_alignment_3d(src, dst, with_scale=True):
    """
    Umeyama alignment in 3D (Sim(3) transformation).

    Args:
        src: [B, N, 3], predicted camera centers (VGGT reference coordinates).
        dst: [B, N, 3], ground-truth camera centers (in global/ego coordinates).
        with_scale: bool, whether to estimate a uniform scaling factor.

    Returns:
        T: [B, 4, 4], batched Sim(3) transformation matrices aligning src -> dst.
    """
    assert src.shape == dst.shape and src.shape[-1] == 3

    with torch.cuda.amp.autocast(enabled=False):
        src = src.float()   # [B, N, 3]
        dst = dst.float()

        B, N, _ = src.shape

        # 1. Compute centroids
        mu_src = src.mean(dim=1, keepdim=True)  # [B, 1, 3]
        mu_dst = dst.mean(dim=1, keepdim=True)  # [B, 1, 3]

        src_centered = src - mu_src             # [B, N, 3]
        dst_centered = dst - mu_dst             # [B, N, 3]

        # 2. Covariance matrices (batched)
        cov = torch.matmul(dst_centered.transpose(1, 2), src_centered) / N   # [B, 3, 3]

        # 3. SVD for each batch
        U, S, Vt = torch.linalg.svd(cov)        # [B, 3, 3], [B, 3], [B, 3, 3]

        R = torch.matmul(U, Vt)                 # [B, 3, 3]

        # Handle reflection case: enforce det(R) > 0
        detR = torch.linalg.det(R)              # [B]
        mask = detR < 0
        if mask.any():
            Vt[mask, -1, :] *= -1
            R[mask] = torch.matmul(U[mask], Vt[mask])

        # 4. Compute scale
        if with_scale:
            var_src = (src_centered ** 2).sum(dim=(1, 2)) / N   # [B]
            scale = S.sum(dim=1) / var_src                      # [B]
        else:
            scale = torch.ones(B, device=src.device, dtype=src.dtype)

        # 5. Compute translation
        t = mu_dst.squeeze(1) - scale.unsqueeze(1) * torch.matmul(R, mu_src.squeeze(1).unsqueeze(-1)).squeeze(-1)  # [B, 3]

        # 6. Assemble Sim(3) transform
        T = torch.eye(4, device=src.device, dtype=src.dtype).unsqueeze(0).repeat(B, 1, 1)  # [B, 4, 4]
        T[:, :3, :3] = R * scale.view(B, 1, 1)
        T[:, :3, 3] = t

        return T


def apply_sim3_transform(points, T):
    """
    Apply batched Sim(3) transformation to 3D points.

    Args:
        points: [B, N, H, W, 3] or [B, N, 3], batched 3D points.
        T: [B, 4, 4], batched Sim(3) transformation matrices.

    Returns:
        points_reg: same shape as input points, transformed by T.
    """
    B = points.shape[0]
    orig_shape = points.shape

    # Flatten points: [B, M, 3], where M = product of remaining dims
    points_flat = points.reshape(B, -1, 3)                       # [B, M, 3]

    # Homogenize: [B, M, 4]
    ones = torch.ones((B, points_flat.shape[1], 1),
                      dtype=points.dtype, device=points.device)
    points_homo = torch.cat([points_flat, ones], dim=-1)         # [B, M, 4]

    # Apply Sim(3): [B, M, 4]
    points_reg = torch.bmm(points_homo, T.transpose(1, 2))       # [B, M, 4]

    # Drop homogeneous coordinate: [B, M, 3]
    points_reg = points_reg[..., :3]

    # Reshape back to original shape
    points_reg = points_reg.reshape(orig_shape)

    return points_reg