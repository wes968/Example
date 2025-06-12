from flask import Flask, render_template, jsonify, request
import numpy as np

app = Flask(__name__)

# 折扣因子
gamma = 0.9
# 行動：上、下、左、右
actions = {'↑': (-1, 0), '↓': (1, 0), '←': (0, -1), '→': (0, 1)}

def value_iteration(grid_size, start, end, obstacles, threshold=0.01):
    # 初始化價值矩陣和策略矩陣
    value_matrix = np.zeros((grid_size, grid_size))
    policy_matrix = np.full((grid_size, grid_size), '', dtype=object)
    reward = np.full((grid_size, grid_size), -0.1)  # 降低每步的懲罰以增加目標吸引力

    # 設置起點和終點的獎勵
    start_x, start_y = start
    end_x, end_y = end
    reward[start_x, start_y] = 0    # 起點獎勵為 0
    reward[end_x, end_y] = 100      # 終點獎勵提高到 +100，強烈吸引Agent

    # 設置障礙物的獎勵為 -100（不可通行）
    for obs_x, obs_y in obstacles:
        reward[obs_x, obs_y] = -100

    # 價值迭代
    iteration = 0
    max_iterations = 1000  # 防止無限循環
    while True:
        delta = 0
        new_value_matrix = np.copy(value_matrix)

        for i in range(grid_size):
            for j in range(grid_size):
                # 跳過終點和障礙物（這些狀態不計算策略）
                if (i == end_x and j == end_y) or reward[i, j] == -100:
                    continue

                v = value_matrix[i, j]
                best_value = float('-inf')
                best_action = None

                # 嘗試每個行動
                for action, (di, dj) in actions.items():
                    ni, nj = i + di, j + dj
                    # 檢查是否在網格內
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        # 如果下一個狀態是障礙物，則跳過
                        if reward[ni, nj] == -100:
                            continue
                        # 計算新價值：當前獎勵 + 折扣後的未來價值
                        # 如果下一個狀態是終點，直接使用終點獎勵
                        if (ni == end_x and nj == end_y):
                            new_value = reward[i, j] + reward[ni, nj]
                        else:
                            new_value = reward[i, j] + gamma * value_matrix[ni, nj]
                        if new_value > best_value:
                            best_value = new_value
                            best_action = action

                if best_action:
                    new_value_matrix[i, j] = best_value
                    policy_matrix[i, j] = best_action
                    delta = max(delta, abs(v - best_value))

        value_matrix = new_value_matrix
        iteration += 1


        # 檢查是否收斂
        if delta < threshold or iteration >= max_iterations:
            print(f"Converged after {iteration} iterations with delta {delta}")
            break

    # 正規化價值矩陣到 0-1 範圍
    min_value = np.min(value_matrix)
    max_value = np.max(value_matrix)
    if max_value != min_value:  # 避免除以零
        value_matrix = (value_matrix - min_value) / (max_value - min_value)
    else:
        value_matrix = np.zeros_like(value_matrix)  # 如果所有值相同，設為 0

    # 確保終點的價值為 0（根據你的截圖）
    value_matrix[end_x, end_y] = 0.0

    # 打印策略和價值矩陣以供檢查
    print(f"網格大小: {grid_size}x{grid_size}")
    print("策略矩陣:")
    for row in policy_matrix:
        print(row)
    print("正規化後的價值矩陣:")
    for row in value_matrix:
        print(row)

    return policy_matrix.tolist(), value_matrix.tolist()

@app.route('/generate_policy_value/<int:grid_size>', methods=['POST'])
def generate_policy_value(grid_size):
    data = request.get_json()
    start = data['start']  # 起點座標 [x, y]
    end = data['end']      # 終點座標 [x, y]
    obstacles = data['obstacles']  # 障礙物座標列表 [[x1, y1], [x2, y2], ...]

    policy, value = value_iteration(grid_size, start, end, obstacles)
    return jsonify({"policy": policy, "value": value})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)