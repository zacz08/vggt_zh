import torch
import torch.nn.functional as F
from ultralytics import YOLO
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import preprocess_image_batch
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from datasets.nuScenesDataset import nuScenesDataset
from nuscenes.nuscenes import NuScenes
from omegaconf import OmegaConf
from utils.geometry import transform_extrinsics_to_current_ego, transform_pointmap_to_current_ego
from utils.viser_tools import viser_wrapper


# ---------- 参数 ----------
person_id    = 0   # COCO 类别编号
car_id       = 2
# -------------------------

    
device = "cuda" if torch.cuda.is_available() else "cpu"
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

cfg = OmegaConf.load('./configs/vggt_seg.yaml')
cfg = cfg.model.params
nusc = NuScenes(version=cfg.version, dataroot=cfg.dataroot, verbose=False)
dataset = nuScenesDataset(nusc, 'val', cfg)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

with torch.no_grad():
    for batch in dataloader:
        images = batch['image'].to(device)              # [bs, seq_length, num_cam, c, h, w]
        extrinsic_gt = batch['extrinsics'].to(device)   # [bs, seq_length, num_cam, 4, 4] cam-> ego
        intrinsic_gt = batch['intrinsics']             # [bs, seq_length, num_cam, 3, 3]
        ego_pose = batch['egopose'].to(device)         # [bs, seq_length, 4, 4] ego-> global
        bev_token = batch['bev_token']               # nuscenes sample token

        ###################### mask dynamic objects ######################
        # 1. 加载YOLO Seg
        B, S, N, C, H, W = images.shape
        images_history = images[:, :-1].reshape(-1, C, H, W)
        images_history = F.interpolate(images_history.float(), size=(896, 1600), mode="bilinear", align_corners=False)
        yoloseg = YOLO("yolov8n-seg.pt")
        seg_results = yoloseg.predict(images_history, imgsz=896, conf=0.5, verbose=False)

        # 2. 遍历历史帧，mask掉行人和车辆
        masked_images = []

        for img, result in zip(images_history, seg_results):   # img: [3, H, W], result: YOLO结果
            img = img.clone()

            if result.masks is not None:
                # result.masks.data: [num_objects, H, W], 0/1 float tensor
                masks = result.masks.data.to(img.device)   # 确保在同一个 device
                combined_mask = masks.sum(dim=0).clamp(max=1)  # 合并所有实例 mask → [H, W]
                combined_mask = combined_mask.unsqueeze(0).repeat(3, 1, 1)  # 扩展到通道维度 [3, H, W]
                img = img * (1 - combined_mask) # 在 mask 区域置 0

            masked_images.append(img)

        # 最后拼成一个 batch
        masked_images = torch.stack(masked_images)   # [B, 3, H, W]

        # 最后一组原始图像
        last_seq = images[:, -1]      # [1, 6, 3, 900, 1600]
        last_seq = last_seq.reshape(-1, 3, 900, 1600)   # [6, 3, 900, 1600]

        # resize 到 (896, 1600)
        last_seq_resized = F.interpolate(
            last_seq.float(), size=(896, 1600), mode="bilinear", align_corners=False
        )

        # 拼接回 masked_images
        all_images = torch.cat([masked_images, last_seq_resized], dim=0)   # [18, 3, 896, 1600]
        images = all_images.reshape(1, 3, 6, 3, 896, 1600)

        # Flatten in the order of [frame, camera] to ensure correspondence between extrinsics and images
        images_flat = images.reshape(B, S * N, 3, images.shape[-2], images.shape[-1])
        images_flat = preprocess_image_batch(images_flat)  # [B, seq_len * num_cam, 3, H, W]

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images_flat)

        predictions.pop('pose_enc_list', None)
        extrinsic_pred, intrinsic_pred = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

        
        # 使用GT外参, transfrom pointmap到GT外参下
        predictions["extrinsic"] = transform_extrinsics_to_current_ego(extrinsic_gt[0], ego_pose[0])
        predictions["intrinsic"] = intrinsic_gt.reshape(S * N, 3, 3)  # [N, 3, 3]

        pointmap_last_ego = transform_pointmap_to_current_ego(
            predictions["world_points"], 
            extrinsic_pred,  # [seq_len * num_cam, 4, 4]
            extrinsic_gt,
            ego_pose)
        predictions["world_points"] = pointmap_last_ego
        predictions["images"] = images_flat 


        # 保存mask
        all_masks = []
        for result in seg_results:
            if result.masks is not None:
                masks = result.masks.data.to(device)   # [num_obj, H, W]
                combined_mask = masks.sum(dim=0).clamp(max=1)  # [H, W]
            else:
                combined_mask = torch.zeros(images_history.shape[-2:], device=device)
            all_masks.append(combined_mask)
        all_masks = torch.stack(all_masks)  # 拼成 [B, H, W]

        # 和最后一组原图对齐
        last_seq_mask = torch.zeros_like(last_seq_resized[:,0])  # [6, H, W]
        all_masks = torch.cat([all_masks, last_seq_mask], dim=0) # [18, H, W]

        # reshape 回 [1, 3, 6, H, W]
        mask = all_masks.reshape(1, S, N, 896, 1600).reshape(-1, 896, 1600)
        world_points = predictions["world_points"]
        
        # 将 mask resize成VGGT输入输出大小，并应用到 world_points 上
        mask_resized = F.interpolate(
            mask.unsqueeze(1).float(),  # [18, 1, 896, 1600]
            size=(world_points.shape[2], world_points.shape[3]),  # (294, 518)
            mode="nearest"
        ).squeeze(1).bool()  # [18, 294, 518]
        mask_resized = mask_resized.unsqueeze(0).unsqueeze(-1)
        mask_resized = mask_resized.expand(-1, -1, -1, -1, 3) 
        world_points[mask_resized] = float('nan') 
        predictions["world_points"] = world_points        

        # print("Processing model outputs...")
        for key in predictions.keys():
            predictions[key] = predictions[key].squeeze(0)  # remove batch dimension

        # --- Visualize Point cloud and BEV feature (on the BEV plane) ---
        viser_wrapper(predictions,port=8080,init_conf_threshold=25,use_point_map=True,
            background_mode=True,mask_sky=True,raw_images=images[0],cfg=cfg)
        
        print(f"Processed sample {bev_token} with {S} frames and {N} cameras.")