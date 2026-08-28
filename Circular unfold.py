import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ==================== 1. 图像读取与预处理 ====================
img_path = r"C:\Users\LLXK\Desktop\huanxingzhankai\01.jpg"
img_original_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

assert img_original_bgr is not None, f"❌ 图像读取失败：{img_path}"

folder = os.path.dirname(img_path)
filename = os.path.splitext(os.path.basename(img_path))[0]

h, w = img_original_bgr.shape[:2]
cx, cy = w // 2, h // 2
short_side = min(h, w)

center_thresh = short_side * 0.05
cross_len = int(short_side * 0.5)

gray = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2GRAY)
imgThreshold = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]
img_rgb_for_show = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB)


# ==================== 2. 二值图边缘增强 ====================
def enhance_binary_edge(binary_img):
    enhanced = cv2.convertScaleAbs(binary_img, alpha=1, beta=0)
    _, enhanced = cv2.threshold(enhanced, 50, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)
    )
    enhanced = cv2.dilate(
        enhanced, kernel, iterations=3
    )

    return enhanced


enhanced_gray = enhance_binary_edge(imgThreshold)
print("✅ 二值化+边缘增强完成")


# ==================== 3. 霍夫圆检测 ====================
minR = int(short_side * 0.02)
maxR = int(short_side * 0.8)

circles = cv2.HoughCircles(
    enhanced_gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=int(short_side * 0.01),
    param1=100,
    param2=50,
    minRadius=minR,
    maxRadius=maxR
)

vis_detection = img_rgb_for_show.copy()
optimal_center = None
valid_circles = []

if circles is not None:
    circles = np.around(circles[0]).astype(int)

    dists = np.hypot(
        circles[:, 0] - cx,
        circles[:, 1] - cy
    )

    valid_circles = circles[dists <= center_thresh]

    for x, y, r in valid_circles:
        cv2.circle(
            vis_detection,
            (x, y),
            r,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

        cv2.circle(
            vis_detection,
            (x, y),
            3,
            (0, 255, 0),
            -1,
            cv2.LINE_AA
        )

        cv2.putText(
            vis_detection,
            f"R:{r}",
            (x + r, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 100, 0),
            1
        )

    if len(valid_circles) >= 1:
        x_list = valid_circles[:, 0]
        y_list = valid_circles[:, 1]
        r_list = valid_circles[:, 2]

        # 按半径加权计算最优圆心
        opt_x = int(np.average(x_list, weights=r_list))
        opt_y = int(np.average(y_list, weights=r_list))
        optimal_center = (opt_x, opt_y)

        cv2.line(
            vis_detection,
            (opt_x - cross_len, opt_y),
            (opt_x + cross_len, opt_y),
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.line(
            vis_detection,
            (opt_x, opt_y - cross_len),
            (opt_x, opt_y + cross_len),
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            vis_detection,
            f"最优圆心({opt_x},{opt_y})",
            (opt_x + 10, opt_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            1
        )

    status = (
        f"检测到{len(circles)}个圆，"
        f"{len(valid_circles)}个在中心区\n"
        f"最优圆心：{optimal_center}"
    )

else:
    status = "未检测到任何圆，无法裁剪"
    print(f"❌ {status}")


# ==================== 4. 环形区域裁剪 ====================
cropped_ring_rgba = None
vis_crop_area = img_rgb_for_show.copy()

inner_r = 0
outer_r = 0

if optimal_center is not None:
    opt_x, opt_y = optimal_center

    avg_r = (
        int(np.mean(valid_circles[:, 2]))
        if len(valid_circles) > 0
        else int(short_side * 0.2)
    )

    default_inner = int(avg_r * 0.6)
    default_outer = avg_r

    print(
        f"\n📏 环形裁剪参数"
        f"（参考值：内圈={default_inner}，外圈={default_outer}）"
    )

    # 输入并校验内外圈半径
    while True:
        try:
            inner_r = int(
                input("请输入内圈半径（内圈以内将被裁掉）：")
            )

            outer_r = int(
                input("请输入外圈半径（外圈以外将被裁掉）：")
            )

            max_safe_outer = min(
                opt_x,
                w - opt_x,
                opt_y,
                h - opt_y
            )

            if inner_r < 0 or outer_r < 0:
                print("❌ 错误：半径不能为负数！")

            elif inner_r >= outer_r:
                print("❌ 错误：内圈半径不能≥外圈半径！")

            elif outer_r > max_safe_outer:
                print(
                    f"❌ 错误：外圈半径过大"
                    f"（最大安全值为{max_safe_outer}）！"
                )

            else:
                print(
                    f"✅ 有效参数："
                    f"内圈={inner_r}，外圈={outer_r}"
                )
                break

        except ValueError:
            print("❌ 错误：请输入整数！")

    # 截取外圈矩形区域
    x_min = max(0, opt_x - outer_r)
    y_min = max(0, opt_y - outer_r)
    x_max = min(w, opt_x + outer_r)
    y_max = min(h, opt_y + outer_r)

    img_crop_bgr = img_original_bgr[
        y_min:y_max,
        x_min:x_max
    ].copy()

    # 创建环形掩码
    mask_h = y_max - y_min
    mask_w = x_max - x_min

    mask = np.zeros(
        (mask_h, mask_w),
        dtype=np.uint8
    )

    mask_center_x = opt_x - x_min
    mask_center_y = opt_y - y_min

    cv2.circle(
        mask,
        (mask_center_x, mask_center_y),
        outer_r,
        255,
        -1
    )

    cv2.circle(
        mask,
        (mask_center_x, mask_center_y),
        inner_r,
        0,
        -1
    )

    # 生成透明背景环形图像
    img_crop_rgba = cv2.cvtColor(
        img_crop_bgr,
        cv2.COLOR_BGR2RGBA
    )

    img_crop_rgba[:, :, 3] = mask
    cropped_ring_rgba = img_crop_rgba

    # 标记裁剪区域
    cv2.rectangle(
        vis_crop_area,
        (x_min, y_min),
        (x_max, y_max),
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.circle(
        vis_crop_area,
        (opt_x, opt_y),
        outer_r,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.circle(
        vis_crop_area,
        (opt_x, opt_y),
        inner_r,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )


# ==================== 5. 环形区域展开(关键代码：涉及到坐标转换) ====================
def unwrap_ring_3_to_1(
        ring_rgba,
        mask_center,
        inner_r,
        outer_r
):
    ring_width = outer_r - inner_r

    unwrap_height = ring_width
    unwrap_width = int(unwrap_height * 3.7)

    unwrapped_rgba = np.zeros(
        (unwrap_height, unwrap_width, 4),
        dtype=np.uint8
    )

    for y_unwrap in range(unwrap_height):
        for x_unwrap in range(unwrap_width):

            theta = (
                x_unwrap / unwrap_width
            ) * 2 * np.pi

            rho = (
                inner_r
                + (y_unwrap / unwrap_height)
                * (outer_r - inner_r)
            )

            x_ring = (
                mask_center[0]
                + rho * np.cos(theta)
            )

            y_ring = (
                mask_center[1]
                + rho * np.sin(theta)
            )

            x_ring_clamped = np.clip(
                int(np.round(x_ring)),
                0,
                ring_rgba.shape[1] - 1
            )

            y_ring_clamped = np.clip(
                int(np.round(y_ring)),
                0,
                ring_rgba.shape[0] - 1
            )

            unwrapped_rgba[
                y_unwrap,
                x_unwrap
            ] = ring_rgba[
                y_ring_clamped,
                x_ring_clamped
            ]

    return unwrapped_rgba


unwrapped_ring_rgba = None

if cropped_ring_rgba is not None:
    mask_center = (
        mask_center_x,
        mask_center_y
    )

    unwrapped_ring_rgba = unwrap_ring_3_to_1(
        cropped_ring_rgba,
        mask_center,
        inner_r,
        outer_r
    )

    print("✅ 环形展开完成")


# ==================== 6. 结果可视化 ====================
plt.figure(figsize=(20, 10))

plt.subplot(2, 3, 1)
plt.imshow(img_rgb_for_show)
plt.title("1. 原始图像")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(imgThreshold, cmap='gray')
plt.title("2. 二值化结果")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(vis_detection)
plt.title("3. 圆心检测结果")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(vis_crop_area)
plt.title("4. 裁剪区域标记")
plt.axis("off")

if cropped_ring_rgba is not None:
    plt.subplot(2, 3, 5)
    plt.imshow(cropped_ring_rgba)
    plt.title("5. 裁剪环形结果")
    plt.axis("off")

if unwrapped_ring_rgba is not None:
    plt.subplot(2, 3, 6)
    plt.imshow(unwrapped_ring_rgba)
    plt.title("6. 环形展开结果")
    plt.axis("off")

plt.tight_layout()
plt.show()


# ==================== 7. 保存各阶段结果 ====================
save_dir = folder

print("\n📤 正在自动保存所有阶段图片...")

cv2.imwrite(
    os.path.join(
        save_dir,
        f"{filename}_01_original.jpg"
    ),
    img_original_bgr
)

cv2.imwrite(
    os.path.join(
        save_dir,
        f"{filename}_02_threshold.png"
    ),
    imgThreshold
)

vis_detection_bgr = cv2.cvtColor(
    vis_detection,
    cv2.COLOR_RGB2BGR
)

cv2.imwrite(
    os.path.join(
        save_dir,
        f"{filename}_03_circle_detection.png"
    ),
    vis_detection_bgr
)

vis_crop_bgr = cv2.cvtColor(
    vis_crop_area,
    cv2.COLOR_RGB2BGR
)

cv2.imwrite(
    os.path.join(
        save_dir,
        f"{filename}_04_crop_mark.png"
    ),
    vis_crop_bgr
)

if cropped_ring_rgba is not None:
    cv2.imwrite(
        os.path.join(
            save_dir,
            f"{filename}_05_ring_crop.png"
        ),
        cropped_ring_rgba
    )

if unwrapped_ring_rgba is not None:
    cv2.imwrite(
        os.path.join(
            save_dir,
            f"{filename}_06_unwrap_3to1.png"
        ),
        unwrapped_ring_rgba
    )

print("✅ 所有阶段图片已保存完毕！")