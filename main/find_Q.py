import numpy as np

if __name__ == "__main__":
    # 弦波參數
    A = 5.0          # 振幅
    omega = 2 * np.pi # 角頻率（1 Hz）
    dt = 0.001        # 時間步長
    total_time = 1
    t = np.arange(0, total_time*1000, 1)
    measurement_noise = 0.01

    # 讀取數據
    # true_pos_shape, true_vel_shape, true_acc_shape, true_pos_addNoise_shape
    path = 'sim_data/dataset/sim_data_all_15000.txt'
    data = np.genfromtxt(path, delimiter=' ')
    true_pos = data[:total_time*1000, 0]
    true_vel = data[:total_time*1000, 1]
    true_acc = data[:total_time*1000, 2]
    # true_pos_addNoise = data[:total_time*1000, 3]
    true_pos_addNoise = data[:total_time*1000, 3]

    v_data = [0]
    a_data = [0,0]
    for i in range(len(true_pos_addNoise)-1):
        x = true_pos_addNoise
        v = (x[i+1]-x[i])/dt
        v_data.append(v)
        # print("v_data =", v_data)
        # a = (v_data[i+1]-v_data[i])/dt
    for i in range(len(true_pos_addNoise)-2):
        a = (x[i+2] - 2 * x[i+1] + x[i]) / dt**2
        a_data.append(a)
    v_data = np.array(v_data)
    a_data = np.array(a_data)

    print("true_pos_addNoise =", true_pos_addNoise.shape)
    print("v_data =", v_data.shape)
    print("a_data =", a_data.shape)
    x_data = np.array([true_pos_addNoise,v_data,a_data])
    print("x_data =", x_data)
    sigma = np.cov(x_data)
    print("sigma =", sigma)
    A = np.array([[1, dt, 0.5*dt**2],
                  [0, 1, dt],
                  [0, 0, 1 ]])
    Q = np.dot(np.dot(A, sigma), A.T)
    print("Q =", Q)