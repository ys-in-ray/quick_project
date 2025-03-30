# gradio_app.py
import os
import cv2
import math
import time
import heapq
import tempfile
import itertools
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from scipy.interpolate import splprep, splev

global_state = {}

def preprocess_image(image):
    tmpdir = tempfile.mkdtemp()
    preprocessed_path = os.path.join(tmpdir, "preprocessed.png")

    blockSize = 31
    C = 7
    dilate_iter = 0
    morph_kernel = (5, 5)
    min_area = 100

    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=blockSize, C=C
    )
    kernel = np.ones(morph_kernel, np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=dilate_iter)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    def is_informative_contour(cnt, img, gradient_threshold=10):
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=1)
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        edge_strength = grad_mag[mask == 255]
        mean_gradient = edge_strength.mean() if edge_strength.size > 0 else 0
        return mean_gradient > gradient_threshold and cv2.contourArea(cnt) > min_area

    filtered_contours = [cnt for cnt in contours if is_informative_contour(cnt, img)]
    debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug_img, filtered_contours, -1, (0, 0, 255), 1)
    cv2.imwrite(preprocessed_path, debug_img)

    global_state["img"] = img
    global_state["contours"] = filtered_contours
    global_state["tmpdir"] = tmpdir

    return preprocessed_path

def generate_fourier_animation(K, frame_step, tail_ratio):
    if "img" not in global_state or "contours" not in global_state:
        yield None, "❗ 請先上傳圖片並預處理後再產生動畫。"
        return

    img = global_state["img"]
    filtered_contours = global_state["contours"]
    tmpdir = global_state["tmpdir"]
    result_gif_path = os.path.join(tmpdir, "result.gif")

    all_points = []
    for cnt in filtered_contours:
        pts = cnt.squeeze()
        if pts.shape[0] >= 5:
            if not np.array_equal(pts[0], pts[-1]):
                pts = np.concatenate([pts, pts[0:1]], axis=0)
            all_points.append(pts)

    next_gid = itertools.count()
    groups = {next(next_gid): [pts] for pts in all_points}

    def group_distance(i, j):
        pts1 = np.concatenate(groups[i], axis=0)
        pts2 = np.concatenate(groups[j], axis=0)
        dists = np.linalg.norm(pts1[:, None, :] - pts2[None, :, :], axis=2)
        idx = np.unravel_index(np.argmin(dists), dists.shape)
        return dists[idx], pts1[idx[0]], pts2[idx[1]]

    def group_distance_from_data(group1, group2):
        pts1 = np.concatenate(group1, axis=0)
        pts2 = np.concatenate(group2, axis=0)
        dists = np.linalg.norm(pts1[:, None, :] - pts2[None, :, :], axis=2)
        idx = np.unravel_index(np.argmin(dists), dists.shape)
        return dists[idx], pts1[idx[0]], pts2[idx[1]]

    heap = []
    gid_list = list(groups.keys())
    for i in range(len(gid_list)):
        for j in range(i + 1, len(gid_list)):
            gi, gj = gid_list[i], gid_list[j]
            d, p1, p2 = group_distance(gi, gj)
            heapq.heappush(heap, (d, gi, gj, tuple(p1), tuple(p2)))

    while len(groups) > 1:
        while True:
            d, gi, gj, p1, p2 = heapq.heappop(heap)
            if gi in groups and gj in groups:
                break

        def rotate_to_start(pts, target):
            dists = np.linalg.norm(pts - target, axis=1)
            idx = np.argmin(dists)
            return np.concatenate([pts[idx:], pts[1:idx+1]], axis=0)

        group_gi, group_gj = groups[gi], groups[gj]
        _, pa, pb = group_distance(gi, gj)
        new_pts = [
            rotate_to_start(np.concatenate(group_gi, axis=0), pa),
            rotate_to_start(np.concatenate(group_gj, axis=0), pb)
        ]

        new_gid = next(next_gid)
        groups[new_gid] = new_pts
        del groups[gi], groups[gj]

        for other_gid in groups:
            if other_gid == new_gid:
                continue
            d1, p1a, p2a = group_distance_from_data(group_gi, groups[other_gid])
            d2, p1b, p2b = group_distance_from_data(group_gj, groups[other_gid])
            closer = (d1, new_gid, other_gid, tuple(p1a), tuple(p2a)) if d1 < d2 \
                     else (d2, new_gid, other_gid, tuple(p1b), tuple(p2b))
            heapq.heappush(heap, closer)

    combined_points = np.concatenate(groups[list(groups.keys())[0]], axis=0)
    _, unique_idx = np.unique(combined_points, axis=0, return_index=True)
    combined_points = combined_points[np.sort(unique_idx)]

    h, w = img.shape
    fx = combined_points[:, 0] - w // 2
    fy = -(combined_points[:, 1] - h // 2)
    tck, u = splprep([fx, fy], s=0, per=False)
    fx_smooth, fy_smooth = splev(u, tck)
    z = fx_smooth + 1j * fy_smooth
    N = len(z)

    Z = np.fft.fft(z)
    freqs = np.fft.fftfreq(N, d=1/N)
    indices = np.argsort(-np.abs(Z))[:K]
    components = sorted([(freqs[n], Z[n]) for n in indices], key=lambda x: -np.abs(x[1]))

    fig, ax = plt.subplots()
    dot, = ax.plot([], [], 'ro', markersize=4)
    vectors, circles = [], []
    for _ in range(K):
        v, = ax.plot([], [], 'b-', lw=1, alpha=0.4)
        c, = ax.plot([], [], 'gray', lw=0.5, linestyle='--', alpha=0.3)
        vectors.append(v)
        circles.append(c)

    M = int(N * tail_ratio)
    tail_lines = [ax.plot([], [], 'k-', lw=1.5, alpha=0.0)[0] for _ in range(M)]
    trajectory_x, trajectory_y = [], []

    ax.set_aspect('equal')
    ax.set_xlim(np.real(z).min() - 50, np.real(z).max() + 50)
    ax.set_ylim(np.imag(z).min() - 50, np.imag(z).max() + 50)
    ax.axis('off')

    def update(frame):
        t = frame / N
        origin = 0 + 0j
        for i, (freq, coef) in enumerate(components):
            vec = coef * np.exp(2j * np.pi * freq * t) / N
            next_point = origin + vec
            vectors[i].set_data([origin.real, next_point.real], [origin.imag, next_point.imag])
            r = np.abs(coef) / N
            theta = np.linspace(0, 2 * np.pi, 100)
            cx = origin.real + r * np.cos(theta)
            cy = origin.imag + r * np.sin(theta)
            circles[i].set_data(cx, cy)
            origin = next_point

        current_x, current_y = origin.real, origin.imag
        trajectory_x.append(current_x)
        trajectory_y.append(current_y)
        max_tail = min(M, len(trajectory_x) - 1)
        for i in range(max_tail):
            x0, x1 = trajectory_x[-(i+2)], trajectory_x[-(i+1)]
            y0, y1 = trajectory_y[-(i+2)], trajectory_y[-(i+1)]
            tail_lines[i].set_data([x0, x1], [y0, y1])
            dist = np.hypot(x1 - x0, y1 - y0)
            alpha = (M - i) / M * max(0, 1 - (dist / 5 - 1) / 10) if dist > 5 else (M - i) / M
            tail_lines[i].set_alpha(alpha)

        dot.set_data([current_x], [current_y])
        return [dot] + tail_lines + vectors + circles

    frames = list(range(0, N, frame_step))
    total_frames = len(frames)
    start_time = time.time()

    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    canvas = FigureCanvas(fig)
    writer = PillowWriter(fps=30)
    with writer.saving(fig, result_gif_path, dpi=100):
        for i, frame in enumerate(frames):
            update(frame)
            canvas.draw()
            writer.grab_frame()
            elapsed = time.time() - start_time
            processed = i + 1
            estimated_total = elapsed / processed * total_frames if processed > 0 else 0
            remaining = estimated_total - elapsed
            progress_percent = int((processed / total_frames) * 100)
            if True:
                time_status = f"已過 {int(elapsed)} 秒 / 預估剩餘 {max(0, int(remaining))} 秒"
                yield time_status, progress_percent, None

    yield "✅ 動畫產生完成！", 100, result_gif_path
    plt.close()

with gr.Blocks() as demo:
    with gr.Row():
        image_input = gr.Image(type="numpy", label="上傳圖片")
        pre_btn = gr.Button("預處理圖像 (邊框檢測)")
    pre_output = gr.Image(type="filepath", label="預處理結果")

    with gr.Row():
        k_slider = gr.Slider(10, 200, value=70, step=1, label="K 值 (傅立葉分量數)")
        step_slider = gr.Slider(1, 10, value=3, step=1, label="動畫速度 (frame_step)")
        tail_slider = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="尾巴長度 (tail_ratio)")
        anim_btn = gr.Button("產生動畫")

    status_text = gr.Textbox(label="動畫進度說明", interactive=False)
    estimate_box = gr.Slider(minimum=0, maximum=100, step=1, label="處理進度 (%)", value=0)
    anim_output = gr.Image(type="filepath", label="傅立葉動畫 (GIF)")

    pre_btn.click(fn=preprocess_image, inputs=image_input, outputs=pre_output)
    anim_btn.click(fn=generate_fourier_animation, inputs=[k_slider, step_slider, tail_slider], outputs=[status_text, estimate_box, anim_output])

if __name__ == "__main__":
    demo.launch()
