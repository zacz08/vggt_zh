def find_best_resize_and_crop(
    orig_h, orig_w, patch_size=14, top_crop_ratio=46/227,
    min_n=32, max_n=64, min_scale=0.3, max_scale=1.0
):
    """
    自动搜索在指定resize_scale范围下，经过top-crop和14倍数裁剪后，padding最少且尽量大的输出尺寸
    """
    best = None
    min_pad = float('inf')
    bottom_keep_ratio = 1 - top_crop_ratio

    for n in range(max_n, min_n-1, -1):  # 从大到小找
        final_H = patch_size * n
        # 反推resize_scale，使保留的下部分正好能整除14
        resize_scale = final_H / (orig_h * bottom_keep_ratio)
        if not (min_scale <= resize_scale <= max_scale):
            continue
        H1 = orig_h * resize_scale
        W1 = orig_w * resize_scale
        final_W = patch_size * int(W1 // patch_size)  # 向下取最近的14倍数
        pad_W = int(W1) - final_W
        # 可适当限制宽度不低于一定patch数
        if final_W < patch_size * 32:
            continue
        total_pad = pad_W
        if (best is None
            or total_pad < min_pad
            or (total_pad == min_pad and final_W > best[2])):
            best = (resize_scale, final_H, final_W, total_pad, n)
            min_pad = total_pad
    return best

# 使用方法
patch_size = 14
orig_h, orig_w = 900, 1600
top_crop_ratio = 46/227

result = find_best_resize_and_crop(
    orig_h, orig_w, patch_size, top_crop_ratio,
    min_n=32, max_n=64, min_scale=0.3, max_scale=1.0
)

if result:
    resize_scale, final_H, final_W, pad_W, n = result
    print(f"最佳resize_scale: {resize_scale:.4f}, final_H: {final_H}, final_W: {final_W}, patch数=({n}×{final_W//patch_size}), padding宽度: {pad_W}")
else:
    print("找不到满足条件的参数，请适当放宽scale或尺寸要求")
