# -ResNet50-
平面图像的粗糙度检测主要集中在图像纹理的识别，纹理的粗细和疏密程度以及GLCM信息都是判断粗糙度等级的重要依据。本库将开源检测代码以及所用数据集。
（1）第一点为内窥图像的环形展开的主要算法部分，为坐标的逆映射并使用最近邻像素算法，具体可查看项目中的完整代码：
def unwrap_ring_3_to_1(ring_rgba, mask_center, inner_r, outer_r):
    ring_width = outer_r - inner_r
    unwrap_height = ring_width
    unwrap_width = int(unwrap_height * 3.7)
    unwrapped_rgba = np.zeros((unwrap_height, unwrap_width, 4), dtype=np.uint8)

    for y_unwrap in range(unwrap_height):
        for x_unwrap in range(unwrap_width):
            theta = (x_unwrap / unwrap_width) * 2 * np.pi
            rho = inner_r + (y_unwrap / unwrap_height) * (outer_r - inner_r)
            x_ring = mask_center[0] + rho * np.cos(theta)
            y_ring = mask_center[1] + rho * np.sin(theta)
            x_ring_clamped = np.clip(int(np.round(x_ring)), 0, ring_rgba.shape[1] - 1)
            y_ring_clamped = np.clip(int(np.round(y_ring)), 0, ring_rgba.shape[0] - 1)
            unwrapped_rgba[y_unwrap, x_unwrap] = ring_rgba[y_ring_clamped, x_ring_clamped]
    return unwrapped_rgba

# 执行展开
if cropped_ring_rgba is not None:
    mask_center = (mask_center_x, mask_center_y)
    unwrapped_ring_rgba = unwrap_ring_3_to_1(cropped_ring_rgba, mask_center, inner_r, outer_r)
    print("✅ 环形展开完成（3:1比例）")
