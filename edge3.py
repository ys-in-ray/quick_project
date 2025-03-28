import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import splprep, splev
import cv2
import heapq

# ====== 參數區 (Fully Annotated & Adjustable) ======

kernel_size = 3    # 🌟 Gaussian blur kernel size (must be odd)
                   # Small (3) ➔ retains detail but may leave noise
                   # Large (7, 9...) ➔ smoother but loses fine lines

sigma = 3          # 🌟 Blur strength (standard deviation)
                   # 0 = auto; higher = stronger blur

blockSize = 31     # 🌟 Adaptive threshold block size (must be odd)
                   # Small ➔ keeps fine features
                   # Large ➔ smoother thresholded shapes

C = 7              # 🌟 Constant subtracted in adaptive threshold
                   # Lower = brighter; higher = darker areas emphasized

dilate_iter = 0    # 🌟 Dilation iterations
                   # 0 = no dilation; 1~2 = thicken edges

morph_kernel = (5, 5)  # 🌟 Kernel size for closing (filling gaps)
                       # Small (3,3) = fine cracks; large (7,7) = strong fill

min_area = 100     # 🌟 Minimum area of contour to keep
                   # Lower ➔ retain more small detail (more noise)
                   # Higher ➔ cleaner result, fewer small contours

K = 100             # 🌟 Number of Fourier components to keep

frame_step = 2     # 🌟 Animation speed (lower = smoother, higher = faster but rougher)
tail_ratio = 0.6   # 🌟 Ratio of N used for tail length (0.0 ~ 1.0)

# ============================================================

# Step 1: Image preprocessing
img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
blur = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

binary = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    blockSize=blockSize,
    C=C
)

kernel = np.ones(morph_kernel, np.uint8)
dilated = cv2.dilate(binary, kernel, iterations=dilate_iter)
closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
def is_informative_contour(cnt, img, gradient_threshold=10):
    # Create a mask for the contour boundary
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=1)

    # Compute gradient magnitude of the whole image
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Mean gradient magnitude on the contour line
    edge_strength = grad_mag[mask == 255]
    mean_gradient = edge_strength.mean() if edge_strength.size > 0 else 0

    return mean_gradient > gradient_threshold and cv2.contourArea(cnt) > min_area

filtered_contours = [cnt for cnt in contours if is_informative_contour(cnt, img)]

# Visualize detected contours for debug
debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(debug_img, filtered_contours, -1, (0, 0, 255), 1)
plt.imshow(debug_img)
plt.title("Filtered Contours (params adjusted)")
plt.axis("off")
plt.show()

# Step 2: Combine contours using hierarchical closest-merge strategy and ensure each is closed
all_points = []
for cnt in filtered_contours:
    pts = cnt.squeeze()
    if pts.shape[0] >= 5:
        if not np.array_equal(pts[0], pts[-1]):
            pts = np.concatenate([pts, pts[0:1]], axis=0)
        all_points.append(pts)

# Start with each contour as a group
import itertools
next_gid = itertools.count()
groups = {}
gid_map = []
for pts in all_points:
    gid = next(next_gid)
    groups[gid] = [pts]
    gid_map.append(gid)

# Function to get distance between two contour groups

def group_distance(i, j):
    pts1 = np.concatenate(groups[i], axis=0)
    pts2 = np.concatenate(groups[j], axis=0)
    dists = np.linalg.norm(pts1[:, None, :] - pts2[None, :, :], axis=2)
    idx = np.unravel_index(np.argmin(dists), dists.shape)
    p1 = pts1[idx[0]]
    p2 = pts2[idx[1]]
    return dists[idx], p1, p2

# Use data directly (for deleted groups)
def group_distance_from_data(group1, group2):
    pts1 = np.concatenate(group1, axis=0)
    pts2 = np.concatenate(group2, axis=0)
    dists = np.linalg.norm(pts1[:, None, :] - pts2[None, :, :], axis=2)
    idx = np.unravel_index(np.argmin(dists), dists.shape)
    p1 = pts1[idx[0]]
    p2 = pts2[idx[1]]
    return dists[idx], p1, p2

# initial heap
heap = []
gid_list = list(groups.keys())
for i in range(len(gid_list)):
    for j in range(i + 1, len(gid_list)):
        gi = gid_list[i]
        gj = gid_list[j]
        d, p1, p2 = group_distance(gi, gj)
        heapq.heappush(heap, (d, gi, gj, tuple(p1), tuple(p2)))

# track merge history
merge_debug_frames = []
merge_debug_base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).copy()
merge_step = 0

while len(groups) > 1:
    while True:
        d, gi, gj, p1, p2 = heapq.heappop(heap)
        if gi in groups and gj in groups:
            break

    color = tuple(np.random.randint(50, 255, size=3).tolist())
    p1 = tuple(map(int, p1))
    p2 = tuple(map(int, p2))
    highlight_img = merge_debug_base.copy()
    cv2.drawContours(highlight_img, [np.concatenate(groups[gi], axis=0)], -1, color, 1)
    cv2.drawContours(highlight_img, [np.concatenate(groups[gj], axis=0)], -1, color, 1)
    cv2.line(highlight_img, p1, p2, color=color, thickness=1)
    cv2.putText(highlight_img, f"Merge {merge_step}", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    merge_debug_frames.append(highlight_img)

    # 不再持續疊加線條，避免畫出多餘的連線與三角形
    merge_step += 1

    # 合併 group
    group_gi = groups[gi]
    group_gj = groups[gj]

    # 找出最短距離點並切割兩輪廓從該點起始
    def rotate_to_start(pts, target):
        dists = np.linalg.norm(pts - target, axis=1)
        idx = np.argmin(dists)
        return np.concatenate([pts[idx:], pts[1:idx+1]], axis=0)

    _, pa, pb = group_distance(gi, gj)
    new_pts = [
        rotate_to_start(np.concatenate(group_gi, axis=0), pa),
        rotate_to_start(np.concatenate(group_gj, axis=0), pb)
    ]

    new_gid = next(next_gid)
    groups[new_gid] = new_pts
    del groups[gi]
    del groups[gj]

    # 重新與其他 group 建立距離（只對新合併的 group）
    for other_gid in groups:
        if other_gid == new_gid:
            continue
        # 距離定義為合併前群組之一與其他群組的最小距離
        d1, p1a, p2a = group_distance_from_data(group_gi, groups[other_gid])
        d2, p1b, p2b = group_distance_from_data(group_gj, groups[other_gid])
        if d1 < d2:
            heapq.heappush(heap, (d1, new_gid, other_gid, tuple(p1a), tuple(p2a)))
        else:
            heapq.heappush(heap, (d2, new_gid, other_gid, tuple(p1b), tuple(p2b)))

# # 動畫方式顯示合併過程
# fig_merge, ax_merge = plt.subplots()
# img_disp = ax_merge.imshow(merge_debug_frames[0][..., ::-1])
# ax_merge.axis("off")
# ax_merge.set_title("Merging Contours")

# def merge_anim(i):
#     img_disp.set_data(merge_debug_frames[i][..., ::-1])
#     return [img_disp]

# merge_ani = FuncAnimation(fig_merge, merge_anim, frames=len(merge_debug_frames), interval=300, blit=True)
# plt.show()

# Flatten final group
assert len(groups) == 1, "合併後應只剩一組群組"
final_gid = list(groups.keys())[0]
combined_points = np.concatenate(groups[final_gid], axis=0)

# Remove duplicate points (important for splprep stability)
_, unique_idx = np.unique(combined_points, axis=0, return_index=True)
combined_points = combined_points[np.sort(unique_idx)]

# Step 3: Normalize and interpolate
h, w = img.shape
fx = combined_points[:, 0] - w // 2
fy = -(combined_points[:, 1] - h // 2)

# Non-uniform spline sampling
tck, u = splprep([fx, fy], s=0, per=False)
fx_smooth, fy_smooth = splev(u, tck)
z = fx_smooth + 1j * fy_smooth
N = len(z)

# Step 4: Fourier decomposition
Z = np.fft.fft(z)
freqs = np.fft.fftfreq(N, d=1/N)
indices = np.argsort(-np.abs(Z))[:K]
components = [(freqs[n], Z[n]) for n in indices]
components.sort(key=lambda x: -np.abs(x[1]))

# Step 5: Setup plot
fig, ax = plt.subplots()
dot, = ax.plot([], [], 'ro', markersize=4)
vectors, circles = [], []
for _ in range(K):
    v, = ax.plot([], [], 'b-', lw=1, alpha=0.4)
    c, = ax.plot([], [], 'gray', lw=0.5, linestyle='--', alpha=0.3)
    vectors.append(v)
    circles.append(c)

M = int(N * tail_ratio)  # Tail length based on ratio
tail_lines = [ax.plot([], [], 'k-', lw=1.5, alpha=0.0)[0] for _ in range(M)]
trajectory_x, trajectory_y = [], []

ax.set_aspect('equal')
ax.set_xlim(np.real(z).min()-50, np.real(z).max()+50)
ax.set_ylim(np.imag(z).min()-50, np.imag(z).max()+50)
ax.set_title(f"Fourier Epicycle Drawing with Fading Tail (K={K})")

# Step 6: Animation frame update
THRESHOLD = 5  # distance threshold for fade

def update(frame):
    t = frame / N
    origin = 0 + 0j
    positions = [origin]

    for i, (freq, coef) in enumerate(components):
        vec = coef * np.exp(2j * np.pi * freq * t) / N
        next_point = origin + vec
        positions.append(next_point)
        vectors[i].set_data([origin.real, next_point.real], [origin.imag, next_point.imag])

        r = np.abs(coef) / N
        theta = np.linspace(0, 2 * np.pi, 100)
        cx = origin.real + r * np.cos(theta)
        cy = origin.imag + r * np.sin(theta)
        circles[i].set_data(cx, cy)
        origin = next_point

    # Draw tail
    current_x, current_y = origin.real, origin.imag
    trajectory_x.append(current_x)
    trajectory_y.append(current_y)

    max_tail = min(M, len(trajectory_x) - 1)
    for i in range(max_tail):
        x0, x1 = trajectory_x[-(i+2)], trajectory_x[-(i+1)]
        y0, y1 = trajectory_y[-(i+2)], trajectory_y[-(i+1)]
        tail_lines[i].set_data([x0, x1], [y0, y1])

        dist = np.hypot(x1 - x0, y1 - y0)
        if dist > THRESHOLD:
            alpha = (M - i) / M * max(0, 1 - (dist / THRESHOLD - 1) / 10)
        else:
            alpha = (M - i) / M
        tail_lines[i].set_alpha(alpha)

    dot.set_data(current_x, current_y)
    return [dot] + tail_lines + vectors + circles

# Step 7: Animate
ani = FuncAnimation(fig, update, frames=range(0, N, frame_step), interval=3, blit=True)
plt.show()
