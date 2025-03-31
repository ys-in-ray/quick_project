import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# 視窗配置
window_width = 1024
window_height = 768
window = ti.ui.Window("3D 橢球體屁股模擬", (window_width, window_height), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0.2, 0.2, 0.2))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

# 設置相機 - 調整相機位置以更好地觀察模型
camera.position(0.2, 0.3, 0.7)
camera.lookat(0.2, 0.2, 0.2)
camera.fov(45)

n = 10  # 保持原始分辨率，避免性能問題
h = 14  # 保持原始高度
spring_k = 2000.0  # 保持原始彈簧係數
mass = 0.05  # 保持原始質量
dt = 2e-3  # 保持原始時間步長
spacing = 0.015  # 保持原始間距

ground_y = 0.0
ground_stiffness = 8000.0
ground_damping = 0.5
damping_factor = 0.98

valid = ti.field(dtype=ti.i32, shape=(n, h, n))
x = ti.Vector.field(3, dtype=ti.f32, shape=(n, h, n))
x_old = ti.Vector.field(3, dtype=ti.f32, shape=(n, h, n))
v = ti.Vector.field(3, dtype=ti.f32, shape=(n, h, n))
f = ti.Vector.field(3, dtype=ti.f32, shape=(n, h, n))
slap_force = ti.Vector.field(3, dtype=ti.f32, shape=(n, h, n))

# 為了更好的渲染，我們需要準確計算有效點的數量
max_points = n * h * n  # 最大可能的點數
points = ti.Vector.field(3, dtype=ti.f32, shape=max_points)
colors = ti.Vector.field(3, dtype=ti.f32, shape=max_points)
point_count = ti.field(dtype=ti.i32, shape=())  # 跟踪實際有效點數

# 擴充彈簧連接：32-connected（包括內部對角線）
offsets = [(i, j, k) for i in range(-2, 3)
                      for j in range(-2, 3)
                      for k in range(-2, 3)
                      if not (i == 0 and j == 0 and k == 0) and i**2 + j**2 + k**2 <= 4]

@ti.func
def ellipsoid(p, c, rx, ry, rz):
    d = p - c
    return (d[0] / rx)**2 + (d[1] / ry)**2 + (d[2] / rz)**2

@ti.kernel
def init():
    # 改進的屁股形狀建模
    for i, j, k in ti.ndrange(n, h, n):
        valid[i, j, k] = 0  # 預設為無效

        # 基準位置 - 確保模型在相機視野內
        base = ti.Vector([0.2, 0.2, 0.2])
        
        # 計算相對於中心的偏移
        dx = (i - n/2) * spacing
        dy = (j - h/2) * spacing
        dz = (k - n/2) * spacing
        
        # 調整屁股建模參數
        butt_radius_x = 0.07  # 增大半徑
        butt_radius_y = 0.06  # 增大半徑
        butt_radius_z = 0.08  # 增大半徑
        center_offset = 0.05
        
        # 兩個橢球球心
        left_center = ti.Vector([-center_offset, 0.0, 0.0])
        right_center = ti.Vector([center_offset, 0.0, 0.0])
        
        # 當前網格點在局部坐標系中的位置
        local_pos = ti.Vector([dx, dy, dz])

        # 計算到兩個球心的距離比例
        dist_left = ellipsoid(local_pos, left_center, butt_radius_x, butt_radius_y, butt_radius_z)
        dist_right = ellipsoid(local_pos, right_center, butt_radius_x, butt_radius_y, butt_radius_z)
        
        # 更平滑的結合 - 使用平滑的最小值函數
        combined_dist = ti.min(dist_left, dist_right)
        
        # 中間凹槽定義 - 使用更自然的衰減
        crease_depth = 0.02
        crease_width = 0.015
        crease_factor = ti.exp(-local_pos.x**2 / (2 * crease_width**2))
        
        # 底部形狀調整 - 使底部更渾圓
        bottom_factor = 0.0
        if local_pos.y < 0:
            # 底部向下膨脹一點
            bottom_factor = 0.02 * (1.0 + local_pos.y / 0.1) * ti.exp(-(local_pos.x**2) / 0.005)
        
        # 只保留形狀內的點
        if combined_dist <= 1.0:
            # 計算最終位置
            pos = base + local_pos
            
            # 應用中間凹槽
            if abs(local_pos.x) < 0.03:
                pos.y -= crease_depth * crease_factor
            
            # 應用底部調整
            pos.y += bottom_factor
            
            x[i, j, k] = pos
            x_old[i, j, k] = pos
            v[i, j, k] = ti.Vector([0, 0, 0])
            slap_force[i, j, k] = ti.Vector([0, 0, 0])
            valid[i, j, k] = 1

@ti.kernel
def compute_force():
    for i, j, k in x:
        if valid[i, j, k] == 0:
            continue
        f[i, j, k] = ti.Vector([0, -9.8 * mass, 0])
        for dx, dy, dz in ti.static(offsets):
            ni, nj, nk = i + dx, j + dy, k + dz
            if 0 <= ni < n and 0 <= nj < h and 0 <= nk < n and valid[ni, nj, nk] == 1:
                dir = x[ni, nj, nk] - x[i, j, k]
                dist = dir.norm() + 1e-4
                rest_length = spacing * (dx**2 + dy**2 + dz**2)**0.5
                
                # 調整彈簧參數 - 底部更緊實
                local_k = spring_k
                if j < h/3:  # 底部
                    local_k *= 1.2
                
                force = local_k * (dist - rest_length) * dir.normalized()
                f[i, j, k] += force

    for i, j, k in x:
        if valid[i, j, k] == 0:
            continue
        f[i, j, k] += slap_force[i, j, k]
        slap_force[i, j, k] = ti.Vector([0, 0, 0])

@ti.kernel
def apply_ground_reaction():
    for i, j, k in x:
        if valid[i, j, k] == 0:
            continue
        if x[i, j, k].y < ground_y:
            penetration = ground_y - x[i, j, k].y
            f[i, j, k].y += ground_stiffness * penetration - ground_damping * v[i, j, k].y

@ti.kernel
def predict_position():
    for i, j, k in x:
        if valid[i, j, k] == 0:
            continue
        x_old[i, j, k] = x[i, j, k]
        v[i, j, k] += dt * f[i, j, k] / mass
        x[i, j, k] += dt * v[i, j, k]

@ti.kernel
def enforce_position_constraints_once(min_dist_scale: ti.f32, max_dist_scale: ti.f32):
    for i, j, k in x:
        if valid[i, j, k] == 0:
            continue
        for dx, dy, dz in ti.static(offsets):
            ni, nj, nk = i + dx, j + dy, k + dz
            if 0 <= ni < n and 0 <= nj < h and 0 <= nk < n and valid[ni, nj, nk] == 1:
                xi = x[i, j, k]
                xj = x[ni, nj, nk]
                dir = xi - xj
                dist = dir.norm() + 1e-6
                rest_length = spacing * (dx**2 + dy**2 + dz**2)**0.5
                min_dist = rest_length * min_dist_scale
                max_dist = rest_length * max_dist_scale
                
                # 調整約束強度 - 底部更強
                adjust_strength = 0.5
                if j < h/3 or nj < h/3:
                    adjust_strength = 0.6
                
                if dist < min_dist or dist > max_dist:
                    target = rest_length
                    correction = (dist - target) * adjust_strength * dir.normalized()
                    x[i, j, k] -= correction
                    x[ni, nj, nk] += correction

@ti.kernel
def update_velocity():
    for i, j, k in x:
        if valid[i, j, k] == 0:
            continue
        v[i, j, k] = (x[i, j, k] - x_old[i, j, k]) / dt
        
        # 位置相關阻尼
        local_damping = damping_factor
        if x[i, j, k].y < 0.15:  # 靠近地面
            local_damping = ti.min(damping_factor + 0.01, 0.99)
            
        v[i, j, k] *= local_damping

@ti.kernel
def slap(force: ti.f32, cx: ti.f32, cy: ti.f32, cz: ti.f32, radius: ti.f32, dir_x: ti.f32, dir_y: ti.f32, dir_z: ti.f32):
    dir_vec = ti.Vector([dir_x, dir_y, dir_z])
    norm_dir = dir_vec.normalized()
    for i, j, k in x:
        if valid[i, j, k] == 0:
            continue
        dx = x[i, j, k][0] - cx
        dy = x[i, j, k][1] - cy
        dz = x[i, j, k][2] - cz
        dist = ti.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < radius:
            # 平滑衰減
            scale = force * (1.0 - dist / radius)
            delta_f = norm_dir * scale
            slap_force[i, j, k] += delta_f

@ti.kernel
def update_render_data():
    # 首先重置計數器
    point_count[None] = 0
    
    # 收集所有有效點
    for i, j, k in ti.ndrange(n, h, n):
        if valid[i, j, k] == 1:
            idx = ti.atomic_add(point_count[None], 1)
            if idx < max_points:  # 防止越界
                points[idx] = x[i, j, k]
                
                # 更生動的膚色
                v_norm = v[i, j, k].norm()
                height_factor = (x[i, j, k].y - ground_y) / 0.3
                
                # 底色 - 膚色
                base_r = 0.85
                base_g = 0.6
                base_b = 0.5
                
                # 速度影響
                speed_effect = ti.min(v_norm * 2.0, 0.3)
                
                colors[idx] = ti.Vector([
                    ti.min(base_r + speed_effect, 1.0),  # 紅色
                    ti.max(base_g - speed_effect * 0.3, 0.4),  # 綠色
                    ti.max(base_b - speed_effect * 0.5, 0.3)   # 藍色
                ])

# 初始化
init()

# 添加地面
ground_positions = ti.Vector.field(3, dtype=ti.f32, shape=4)
ground_indices = ti.field(int, shape=6)
ground_colors = ti.Vector.field(3, dtype=ti.f32, shape=4)

@ti.kernel
def init_ground():
    # 地面四個角落 - 確保覆蓋模型區域
    ground_positions[0] = ti.Vector([0.0, ground_y, 0.0])
    ground_positions[1] = ti.Vector([0.5, ground_y, 0.0])
    ground_positions[2] = ti.Vector([0.5, ground_y, 0.5])
    ground_positions[3] = ti.Vector([0.0, ground_y, 0.5])
    
    # 兩個三角形構成矩形
    ground_indices[0] = 0
    ground_indices[1] = 1
    ground_indices[2] = 2
    ground_indices[3] = 0
    ground_indices[4] = 2
    ground_indices[5] = 3
    
    # 溫暖的地面顏色
    for i in ti.static(range(4)):
        ground_colors[i] = ti.Vector([0.7, 0.65, 0.6])

init_ground()

# 主循環
frame = 0
prev_mouse = None

# 初始"拍打"使物體開始移動
initial_slap_done = False

while window.running:
    # 在模擬開始時施加一個初始的拍打力
    if not initial_slap_done and frame == 10:
        slap(100.0, 0.2, 0.25, 0.25, 0.1, 0.0, -1.0, 0.2)  # 向下略帶後方的拍打
        initial_slap_done = True
    
    # 獲取滑鼠位置和按鍵狀態
    mouse_pos = window.get_cursor_pos()
    if window.is_pressed(ti.ui.LMB):
        # 如果滑鼠左鍵被按下，進行"slap"操作
        if prev_mouse is not None:
            # 計算滑鼠移動方向
            dx = mouse_pos[0] - prev_mouse[0]
            dy = mouse_pos[1] - prev_mouse[1]
            # 將 2D 滑鼠移動轉換為 3D 方向
            dir_x = dx * 2
            dir_y = -dy * 2  # Y 軸反向
            dir_z = -0.5  # 加入一些深度方向的力
            
            # 計算力度基於滑鼠移動速度
            force = 200.0 * ((dx*dx + dy*dy) ** 0.5) * 10
            
            # 應用拍打力量（位置和方向基於滑鼠位置）
            slap_force_center_x = 0.2 + (mouse_pos[0] - 0.5) * 0.3
            slap_force_center_y = 0.2 + (mouse_pos[1] - 0.5) * 0.3
            slap_force_center_z = 0.2
            slap(force, slap_force_center_x, slap_force_center_y, slap_force_center_z, 0.1, dir_x, dir_y, dir_z)
    
    prev_mouse = mouse_pos
    
    # 物理模擬步驟
    compute_force()
    apply_ground_reaction()
    predict_position()
    for _ in range(10):
        enforce_position_constraints_once(0.8, 1.2)
    update_velocity()
    
    # 更新渲染數據
    update_render_data()
    
    # 設置相機和燈光
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
    scene.point_light(pos=(0.0, 0.5, 0.5), color=(0.6, 0.6, 0.7))  # 增加第二盞燈提高可視性
    
    # 渲染物體
    scene.particles(points, radius=0.008, per_vertex_color=colors)  # 增大粒子半徑以更易於觀察
    scene.mesh(ground_positions, indices=ground_indices, per_vertex_color=ground_colors, two_sided=True)
    
    # 繪製場景並顯示
    canvas.scene(scene)
    window.show()
    
    frame += 1