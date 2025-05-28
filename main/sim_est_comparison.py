import numpy as np
import matplotlib.pyplot as plt
import LAE, CFD, LSF, AKF, AEKF, MY_AKF, KF_v2, zero_phase_filter

class KalmanFilter:
    def __init__(self, dt, Q, R, initial_state=None, initial_covariance=None):
        self.dt = dt
        
        # 狀態轉移矩陣
        self.A_mat = np.array([
            [1, dt, 0.5 * dt**2],
            [0, 1, dt],
            [0, 0, 1]
        ])

        # 觀測矩陣
        self.C = np.array([[1, 0, 0]])  # 測量僅觀測位置

        # 系統噪聲與測量噪聲
        self.Q = Q
        self.R = R

        # 初始條件
        # self.P = np.eye(3) * 1e-3 if initial_covariance is None else initial_covariance
        self.P = np.array([[1e-4, 1e-4, 1e-4],
                             [1e-4, 1e-4, 1e-4],
                             [1e-4, 1e-4, 1e-4]]) if initial_covariance is None else initial_covariance
        self.x = np.zeros((3, 1)) if initial_state is None else initial_state
        self.x_k1 = np.zeros((3, 1)) 
        # print("self.x_k1 222 =", self.x_k1)
        self.P_k = np.array([[1e-8, 1e-8, 1e-8],
                             [1e-8, 1e-8, 1e-8],
                             [1e-8, 1e-8, 1e-8]])

        # 儲存
        self.x_true = np.zeros((3, 1))
        self.k_y = np.zeros((3, 1))
        self.KCP = np.zeros((3, 3))
        self.z = np.zeros((1, 1))
        self.z_data = [self.z.flatten()]
        self.k_y_data = [] # self.k_y.flatten()
        self.x_true_data = []
        # self.x_true_data = [self.x_true.flatten()]
        # print("true_pos_addNoise =", true_pos_addNoise)
        self.x_true_data_noise = [true_pos_addNoise]

        self.KCP_data = [self.KCP.flatten()]
        self.x_k_predict_data = []
        self.P_k_predict_data = []
        self.x_k_update_data = [] # self.x_k1.flatten()
        self.P_k_update_data = [self.P.flatten()]
        self.P_k_data = [self.P_k.flatten()]

        # 存儲估計結果
        self.est_pos = []
        self.est_vel = []
        self.est_acc = []

    def KF(self, measurement):
    # def KF(self, measurement):
        # 預測
        # print("self.A_mat =", self.A_mat)
        # print("self.x_k1 =", self.x_k1)
        self.x = np.dot(self.A_mat, self.x_k1)
        self.x_true_data.append(self.x_k1[0].flatten())
        # print("self.x_k1 =", self.x_k1)
        self.P = np.dot(self.A_mat, np.dot(self.P, self.A_mat.T)) + self.Q
        # print("self.P =", self.P)
        # 更新
        K = np.dot(self.P, self.C.T) / (np.dot(self.C, np.dot(self.P, self.C.T)) + self.R)
        # print("K =", K)
        # print("self.C =", self.C)
        # print("measurement =", measurement)
        # z  = self.C @ measurement
        # print("z =", z)
        # self.z_data.append(z)
        y = measurement - np.dot(self.C, self.x)  # 測量殘差
        # print("y =", y)
        self.k_y = K @ y
        # print("self.k_y =", self.k_y )
        self.k_y_data.append(self.k_y.flatten())
        self.x_k = self.x + np.dot(K, y)
        # print("self.x_k =", self.x_k )
        self.x_k1 = self.x_k
        self.x_k_update_data.append(self.x_k1.flatten())
        # print("self.x =", self.x)
        self.x_k_predict_data.append(self.x.flatten())

        self.KCP = np.dot(np.dot(K, self.C), self.P)
        # print("self.KCP =", self.KCP)
        self.KCP_data.append(self.KCP.flatten())
        self.P_k = np.dot(np.eye(3) - np.dot(K, self.C), self.P)
        # print("self.P_k =", self.P_k)
        self.P = self.P_k
        self.P_k_update_data.append(self.P.flatten())
        self.P_k_predict_data.append(self.P_k.flatten())

        # 儲存估計值
        self.est_pos.append(self.x[0, 0])
        self.est_vel.append(self.x[1, 0])
        self.est_acc.append(self.x[2, 0])

    def get_estimates(self):
        
        # raw data
        est_pos = np.array(self.est_pos)
        est_vel = np.array(self.est_vel)
        est_acc = np.array(self.est_acc)
        # raw_data_all = np.concatenate((est_pos, est_vel, est_acc), axis=1)

        return est_pos, est_vel, est_acc
    
    def getKFData(self):
        # kf x data
        # print("self.x_true_data =", np.array(self.x_true_data).shape)
        # print("self.x_k_update_data =",np.array(self.x_k_update_data).shape)
        self.x_tel = self.x_true_data - np.array(self.x_k_update_data)
        # print(self.k_y_data)
        k_y_data = np.array(self.k_y_data)
        # k_y_data = [np.array(item).reshape(-1) for item in self.k_y_data]
        x_k_update_data = np.array(self.x_k_update_data)
        x_k_update_data = [np.array(item).reshape(-1) for item in x_k_update_data]
        k_y_data = np.array(self.k_y_data)
        x_tel = np.array(self.x_tel)
        # print("x_true_data =", self.x_true_data)
        x_true_data = np.array(self.x_true_data)
        
        x_true_data_noise = np.array(self.x_true_data_noise)
        d = total_time * 1000
        x_true_data_noise = x_true_data_noise.reshape(d, 1)
        # print("x_true_data_noise =", x_true_data_noise.shape)
        x_true_data_noise = [np.array(item).reshape(-1) for item in x_true_data_noise]
        z_data = np.array(self.z_data)
        x_k_predict_data = np.array(self.x_k_predict_data)
        # print("x_true_data =",x_true_data)
        # print("x_true_data_noise =",x_true_data_noise)
        x_data_all = np.concatenate((x_k_update_data, k_y_data, x_tel, x_true_data, x_true_data_noise, x_k_predict_data), axis=1)# me
        # kf P data
        P_k_data = np.array(self.P_k_data)
        P_k_update_data = np.array(self.P_k_update_data)
        P_k_predict_data = np.array(self.P_k_predict_data)
        KCP_data = np.array(self.KCP_data)
        
        P_data_all = np.concatenate((P_k_update_data, KCP_data), axis=1)# me

        return x_data_all, P_data_all
        # return x_k_update_data, x_k_predict_data, P_k_update_data, P_k_predict_data, k_y_data, KCP_data, z_data, x_true_data, x_true_data_noise, P_k_data

# 主程式
if __name__ == "__main__":
    # 弦波參數
    A = 5.0          # 振幅
    omega = 2 * np.pi # 角頻率（1 Hz）
    dt = 0.001        # 時間步長
<<<<<<< HEAD
    total_time = 2
=======
    total_time = 1
>>>>>>> 924f379 (v3)
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

    # true_pos = A * np.sin(omega * t)      # 真實位置
    # true_vel = A * omega * np.cos(omega * t) # 真實速度
    # true_acc = -A * omega**2 * np.sin(omega * t) # 真實加速度

    # # 添加測量噪聲 true_pos_addNoise
    # measurement_noise = 0.01
    # true_pos_addNoise = true_pos + np.random.normal(0, measurement_noise, len(true_pos))
    # # 不佳雜訊
    # # true_pos_addNoise = true_pos
    # x_true_data = np.array(true_pos)
    # x_true_data_noise = np.array(true_pos_addNoise)


    # 系統噪聲協方差矩陣 Q
    # acc CMD variance : 779.2727282720194
    # acc CFD variance : 614691467.3034459
    # acc LAE variance : 4019.7707107859114
    sigma_a2 = 0.1 * (A * omega**2)**2
    Q = 779.2727282720194 * np.array([[dt**4 / 4, dt**3 / 2, dt**2 / 2],
                            [dt**3 / 2, dt**2, dt],
                            [dt**2 / 2, dt, 1]]) # acc CMD variance
    Q0 = 779.2727282720194 * np.array([[dt**5/20, dt**4/8, dt**3/6],
                               [dt**4/8, dt**3/3, dt**2/2],
                               [dt**3/6, dt**2/2, dt]]) # acc CMD variance
    Q1 = sigma_a2 * np.array([[dt**4 / 4, dt**3 / 2, dt**2 / 2],
                            [dt**3 / 2, dt**2, dt],
                            [dt**2 / 2, dt, 1]])
    Q2 = sigma_a2 * np.array([[dt**5/20, dt**4/8, dt**3/6],
                                [dt**4/8, dt**3/3, dt**2/2],
                                [dt**3/6, dt**2/2, dt]])
    
    Q3 = np.array([[8.54565231e-16, 2.56369441e-12, 5.12738034e-09],
                    [2.56369441e-12, 7.69108152e-09, 1.53821460e-05],
                    [5.12738034e-09, 1.53821460e-05, 3.07642919e-02]]) # CMD + CMD
    Q4 = np.array([[8.15126934e-10, 1.63042644e-06, 9.67863139e-04],
                    [1.63042644e-06, 3.26119887e-03, 1.93573114e+00],
                    [9.67863139e-04, 1.93573114e+00, 3.86731610e+03]]) # CMD + LAE
    Q5 = np.array([[1.47037065e-04, 2.94074131e-01, 4.85786766e+02],
                    [2.94074131e-01, 5.88148263e+02, 9.71573532e+05],
                    [4.85786766e+02, 9.71573532e+05, 1.94314706e+09]]) # CMD + CFD
    
    Q6 = np.array( [[ 2.10953361e-05,  1.54816460e-02, -6.34834622e-06],
                    [ 1.54816460e-02,  3.09716112e+01,  1.28983662e-04],
                    [-6.34834622e-06,  1.28983662e-04,  3.07642919e-02]]) # LSF + CMD
    Q7 = np.array( [[ 1.99376017e-05,  1.43526441e-02, -7.56560364e-02],
                    [ 1.43526441e-02,  2.86501390e+01,  1.67488550e+01],
                    [-7.56560364e-02,  1.67488550e+01,  3.51178648e+03]]) # LSF + LAE
    Q8 = np.array( [[1.19836878e-04, 2.39084171e-01, 4.17566213e+02],
                    [2.39084171e-01, 5.27290908e+02, 8.63098538e+05],
                    [4.17566213e+02, 8.63098538e+05, 1.97316680e+09]]) # LSF + CFD
    
    Q9 = np.array( [[ 4.73435319e-05,  4.56292259e-02, -1.73032293e-06],
                    [ 4.56292259e-02,  9.12584520e+01,  1.60958140e-05],
                    [-1.73032293e-06,  1.60958140e-05,  3.07642919e-02]]) # CFD + CMD
    Q10 = np.array( [[ 5.13857904e-05,  5.25416260e-02, -2.29939423e-01],
                    [ 5.25416260e-02,  1.05295128e+02, -4.55771232e+02],
                    [-2.29939423e-01, -4.55771232e+02,  3.97185147e+03]]) # CFD + LAE
    Q11 = np.array( [[2.04064838e-04, 3.09607639e-01, 6.19476684e+02],
                    [3.09607639e-01, 5.21419926e+02, 1.04251985e+06],
                    [6.19476684e+02, 1.04251985e+06, 2.08717058e+09]]) # CFD + CFD
    
    # the best Q of AKF
    # Q12 = np.array ([[1.15753083e-01,1.44723478e-01,1.44723478e+02],
    #                 [1.44723478e-01,5.24216051e+01,4.06162767e+05],
    #                 [1.44723478e+02,4.06162767e+05,2.01654777e+06]]) # 15000筆之Q @ min error without noise
    # Q12 = np.array ([[6.88107483e-02,1.44723478e-01,1.44723478e+02],
    #                 [1.44723478e-01,1.59461490e+01,4.06162767e+05],
    #                 [1.44723478e+02,4.06162767e+05,4.79293703e+06]]) # 15000筆之Q @ min error with noise
    # Q12 = np.array ([[1.91050858e-01,1.44723478e-01,1.44723478e+02],
    #     [1.44723478e-01,6.07838628e+01,4.06162767e+05],
    #     [1.44723478e+02,4.06162767e+05,5.03171839e+06]]) # 1000筆之Q @ min error with noise
    Q12 = np.array ([[1.92459601e-01,1.44723478e-01,1.44723478e+02],
                    [1.44723478e-01,6.06975328e+01,4.06162767e+05],
                    [1.44723478e+02,4.06162767e+05,4.94636026e+06]])
    # u_a, u_v用km去調整
    Q12 = np.array ([[2.88889287e+00,1.44723478e-01,1.44723478e+02],
                    [1.44723478e-01,1.72881162e+00,4.06162767e+05],
                    [1.44723478e+02,4.06162767e+05,2.36342967e+10]]) 
    Q12 = np.array ([[1.78982609e-01,1.27516429e+00,1.23241483e+02],
                    [1.27516429e+00,1.57101535e-01,8.02527454e+05],
                    [1.23241483e+02,8.02527454e+05,5.40689960e+07]])
    Q12 = np.array ([[2.64649722e-01,3.31943150e+01,3.20773320e+03],
 [3.31943150e+01,4.00092855e+03,1.18915429e+06],
 [3.20773320e+03,1.18915429e+06,3.73614976e+07]])
#     Q12 = np.array ([[2.32140748e-02,4.04080332e+00,5.88276371e+02],
#  [4.04080332e+00,6.08746173e+02,9.12197919e+05],
#  [5.88276371e+02,9.12197919e+05,2.32209278e+07]])
#     Q12 = np.array ([[6.62491784e-02,1.27516429e+00,1.23241483e+02],
#  [1.27516429e+00,1.33502851e+03,8.02527454e+05],
#  [1.23241483e+02,8.02527454e+05,1.62297297e+05]])
#     Q12 = np.array ([[3.15400510e-02,2.35778079e+01,5.50988168e+03],
#  [2.35778079e+01,1.76890818e+04,4.14050161e+06],
#  [5.50988168e+03,4.14050161e+06,9.69880022e+08]])
#     Q12 = np.array ([[1.57371942e-01,1.27516429e+00,1.23241483e+02],
#  [1.27516429e+00,1.61060354e+02,8.02527454e+05],
#  [1.23241483e+02,8.02527454e+05,1.97085627e+04]])
#     Q12 = np.array ([[4.60400759e+01,1.44723478e-01,1.44723478e+02],
#  [1.44723478e-01,1.35749629e+02,4.06162767e+05],
#  [1.44723478e+02,4.06162767e+05,1.35351932e+08]])
                    
    
    # Q by variance Q = A@sigma@A.T
    Q13 = np.array([[1.25124714e+01, 1.27516429e+00, 1.23241483e+02],
                    [1.27516429e+00, 1.74838225e+03, 8.02527454e+05],
                    [1.23241483e+02, 8.02527454e+05, 5.35272247e+08]])
    

    # 測量噪聲方差
    # R = np.array([[measurement_noise**2]])
    R = np.array([[measurement_noise]])

    # 初始化卡爾曼濾波器
    for i in range(len(true_pos_addNoise)):
        kf = KalmanFilter(dt=dt, Q=Q, R=R) # 
        kf0 = KalmanFilter(dt=dt, Q=Q0, R=R)
        kf1 = KalmanFilter(dt=dt, Q=Q1, R=R)
        kf2 = KalmanFilter(dt=dt, Q=Q2, R=R)
        kf3 = KalmanFilter(dt=dt, Q=Q3, R=R)
        kf4 = KalmanFilter(dt=dt, Q=Q4, R=R)
        kf5 = KalmanFilter(dt=dt, Q=Q5, R=R)
        kf6 = KalmanFilter(dt=dt, Q=Q6, R=R)
        kf7 = KalmanFilter(dt=dt, Q=Q7, R=R)
        kf8 = KalmanFilter(dt=dt, Q=Q8, R=R)
        kf9 = KalmanFilter(dt=dt, Q=Q9, R=R)
        kf10 = KalmanFilter(dt=dt, Q=Q10, R=R)
        kf11 = KalmanFilter(dt=dt, Q=Q11, R=R)
        kf12 = KalmanFilter(dt=dt, Q=Q12, R=R)
        kf13 = KalmanFilter(dt=dt, Q=Q13, R=R)

    # 執行濾波
    for z in true_pos_addNoise:
        kf.KF(z)
        kf0.KF(z)
        kf1.KF(z)
        kf2.KF(z)
        kf3.KF(z)
        kf4.KF(z)
        kf5.KF(z)
        kf6.KF(z)
        kf7.KF(z)
        kf8.KF(z)
        kf9.KF(z)
        kf10.KF(z)
        kf11.KF(z)
        kf12.KF(z)
        kf13.KF(z)

    # 獲取估計結果
    est_pos, est_vel, est_acc = kf.get_estimates()
    est_pos0, est_vel0, est_acc0 = kf0.get_estimates()
    est_pos1, est_vel1, est_acc1 = kf1.get_estimates()
    est_pos2, est_vel2, est_acc2 = kf2.get_estimates()
    est_pos3, est_vel3, est_acc3 = kf3.get_estimates()
    est_pos4, est_vel4, est_acc4 = kf4.get_estimates()
    est_pos5, est_vel5, est_acc5 = kf5.get_estimates()
    est_pos6, est_vel6, est_acc6 = kf6.get_estimates()
    est_pos7, est_vel7, est_acc7 = kf7.get_estimates()
    est_pos8, est_vel8, est_acc8 = kf8.get_estimates()
    est_pos9, est_vel9, est_acc9 = kf9.get_estimates()
    est_pos10, est_vel10, est_acc10 = kf10.get_estimates()
    est_pos11, est_vel11, est_acc11 = kf11.get_estimates()
    est_pos12, est_vel12, est_acc12 = kf12.get_estimates()
    est_pos13, est_vel13, est_acc13 = kf13.get_estimates()

#---------------------------------------輸出數據.txt-------------------------------------------#
    # 決定以哪個Q值得數據進行儲存輸出
    x_data_all, P_data_all = kf12.getKFData()
    np.savetxt('./sim_data/dataset/x_data_all.txt', x_data_all, delimiter=' ')
    np.savetxt('./sim_data/dataset/P_data_all.txt', P_data_all, delimiter=' ')
    # raw data save
    # est_pos = [np.array(item).reshape(-1) for item in est_pos]
    # est_vel = [np.array(item).reshape(-1) for item in est_vel]
    # est_acc = [np.array(item).reshape(-1) for item in est_acc]
    true_pos_addNoise_shape = true_pos_addNoise.reshape(-1, 1)
    true_pos_shape = true_pos.reshape(-1, 1) 
    true_vel_shape = true_vel.reshape(-1, 1)
    true_acc_shape = true_acc.reshape(-1, 1)
    raw_data_all = np.concatenate((true_pos_shape, true_vel_shape, true_acc_shape, true_pos_addNoise_shape), axis=1)
    np.savetxt('./sim_data/dataset/raw_data_all.txt', raw_data_all, delimiter=' ')
    # print("true_pos =", true_pos_addNoise[len(true_pos_addNoise) -30 :len(true_pos_addNoise)])

#---------------------------------------計算covariance-------------------------------------------#
    # 計算covariance
    Pos_CFD_est, Vel_CFD_est, Acc_CFD_est = CFD.CFD(true_pos_addNoise) 
    Vel_LSF_est = LSF.LSF14(true_pos_addNoise)
    Acc_LAE_est = LAE.LAE(true_pos_addNoise, 0.001)

    v_data = [0]
    a_data = [0,0]
    for i in range(len(true_pos_addNoise)-2):
        x = true_pos_addNoise
        v = (x[i+1]-x[i])/dt
        v_data.append(v)
        # print("v_data =", v_data)
        # a = (v_data[i+1]-v_data[i])/dt
        a = (x[i+2] - 2 * x[i+1] + x[i]) / dt**2
        a_data.append(a)

    zk = []
    err = []
    zk_data = []
    for i in range(len(true_pos_addNoise)-2):
        dt = 0.001
        # zk = np.array([[true_pos[i]], 
        #                 [Vel_CFD_est[i]],
        #                 [Acc_CFD_est[i]]]) # vel為LSF1/4
        # zk_1 = np.array([[true_pos[i+1]], 
        #                 [Vel_CFD_est[i+1]],
        #                 [Acc_CFD_est[i+1]]]) # vel為LSF1/4
        zk = np.array([[true_pos[i]], 
                        [v_data[i]],
                        [a_data[i]]]) # vel為LSF1/4
        zk_1 = np.array([[true_pos[i+1]], 
                        [v_data[i+1]],
                        [a_data[i+1]]]) # vel為LSF1/4
        A = np.array([[1, dt, 0.5*dt**2],
                    [0, 1, dt],
                    [0, 0, 1 ]])
        error = zk_1 - A @ zk
        err.append(error)
        zk_data.append(zk)
    # zk_data = np.array([[true_pos[:998].squeeze()],
    #                     [np.array(v_data)[:998].squeeze()],
    #                     [np.array(a_data)[:998].squeeze()]])
    # print("zk_data =", np.array(zk_data).shape[-1].shape)
    err = np.array(err).squeeze()
    zk_data = np.array(zk_data).squeeze()
    # zk_data = np.array(zk_data).reshape(-1, np.array(zk_data).shape[-1])
    cov_new = np.cov(zk_data.T)
    # zk_data = np.array(zk_data).reshape(-1, np.array(zk_data).shape[-1])
    # cov_new = np.corrcoef(zk_data)
    # print("cov_new =", cov_new)

    #--------------------------離散時間的VEL, ACC-------------------------#
    # plt.figure(figsize=(8, 6))
    # plt.subplot(2, 1, 1)
    # plt.plot(v_data, "black", label="True vel", linewidth=1)
    # plt.xlabel("t")
    # plt.ylabel("vel")
    # plt.legend(loc='upper right')
    # plt.subplot(2, 1, 2)
    # plt.title("ture_data", loc="center")
    # plt.plot(a_data, "blue", label="True acc", linewidth=1)
    # plt.xlabel("t")
    # plt.ylabel("acc")

    print('-----------------------------------------------------------------')
    cov = np.cov(err.T)
    formatted_Q = np.array2string(cov, separator=',', formatter={'float_kind': lambda x: f'{x:.8e}'})
    formatted_Q = formatted_Q.replace('[ ', '[').replace(' ]', ']').replace('\n', '\n').replace('  ', ' ')
    formatted_Q = f"({formatted_Q})"
    print("cov(err) = np.array", formatted_Q) 
    # variance_p = np.var(err[:, 0])
    # print("variance_p = ", variance_p)
    # variance_v = np.var(err[:, 1])
    # print("variance_v = ", variance_v)
    # variance_a = np.var(err[:, 2])
    # print("variance_a = ", variance_a)
    print('-----------------------------------------------------------------')
    print("acc estimate")
    print("acc CMD variance :", np.var(true_acc))
    print("acc CFD variance :", np.var(Acc_CFD_est))
    print("acc LAE variance :", np.var(Acc_LAE_est))
    print('-----------------------------------------------------------------')
#---------------------------------------AKF-------------------------------------------#  
    pose, vele, acce, Q_pos, Q_acc, Q_vel, u_p_values, u_v_values, u_a_values, Q_save, v_cov_data, a_cov_data = AKF.AKF_2(dt, true_pos_addNoise, true_pos, true_vel, true_acc)
    # 找出位置誤差最小的Q值
    print('-------------------------------AKF----------------------------------')
    print("u_p_values=", min(np.abs(u_p_values[1:])))
    # print("Q_save=", Q_save)
    
    # for i in range(1,len(u_p_values)):
    #     if u_p_values[i] == min(np.abs(u_p_values[1:])):
    i = np.argmin(np.abs(u_p_values[1:])) + 1
    # print("u_p_values =", u_p_values[i])
    formatted_Q = np.array2string(Q_save[i], separator=',', formatter={'float_kind': lambda x: f'{x:.8e}'})
    formatted_Q = formatted_Q.replace('[ ', '[').replace(' ]', ']').replace('\n', '\n').replace('  ', ' ')
    formatted_Q = f"({formatted_Q})"
    print("i =", i)
    print("Q12 = np.array", formatted_Q)
    # print("u_p_values=", min(np.abs(u_p_values[1:])))
    print('-----------------------------------------------------------------')

#---------------------------------------AEKF-------------------------------------------#  
    AEKF_pose, AEKF_vele, AEKF_acce, AEKF_Q_pos, AEKF_Q_acc, AEKF_Q_vel, AEKF_u_p_values, AEKF_u_v_values, AEKF_u_a_values, AEKF_Q_save = MY_AKF.MY_AKF(dt, true_pos_addNoise, true_pos, true_vel, true_acc)
    # 找出位置誤差最小的Q值
    print('---------------------------------AEKF--------------------------------')
    print("u_p_values=", min(np.abs(AEKF_u_p_values[1:])))
    # print("Q_save=", Q_save)
    # for i in range(1,len(u_p_values)):
    #     if u_p_values[i] == min(np.abs(u_p_values[1:])):
    i = np.argmin(np.abs(AEKF_u_p_values[1:])) + 1
    # print("u_p_values =", u_p_values[i])
    formatted_Q = np.array2string(AEKF_Q_save[i], separator=',', formatter={'float_kind': lambda x: f'{x:.8e}'})
    formatted_Q = formatted_Q.replace('[ ', '[').replace(' ]', ']').replace('\n', '\n').replace('  ', ' ')
    formatted_Q = f"({formatted_Q})"
    print("i =", i)
    print("Q12 = np.array", formatted_Q)
    # print("u_p_values=", min(np.abs(u_p_values[1:])))
    print('-----------------------------------------------------------------')

#---------------------------------------可視化結果-------------------------------------------#
    # # 可視化結果
    # #-------------------------------------不同模型假設比較-------------------------------------------#
    # # pos
    
    # plt.figure(figsize=(8, 6))
    # plt.subplot(3, 1, 1)
    # plt.plot(true_pos, "black", label="True Pos", linewidth=1)
    # plt.xlabel("t")
    # plt.ylabel("pos")
    # plt.legend(loc='upper right')
    # plt.title("ture_data", loc="center")
    # plt.subplot(3, 1, 2)
    # plt.plot(true_vel, "blue", label="True Vel", linewidth=1)
    # plt.xlabel("t")
    # plt.ylabel("vel")
    # plt.legend(loc='upper right')
    # plt.subplot(3, 1, 3)
    # plt.plot(true_acc, "red", label="True Acc", linewidth=1)  
    # plt.xlabel("t")
    # plt.ylabel("acc")
    # plt.legend(loc='upper right')
    # plt.tight_layout()
    # # plt.title("ground truth", loc="center")

    # # print("est_pos =", est_pos.shape)
    # plt.figure(figsize=(8, 6))
    # # plt.figure(1)
    # plt.subplot(3, 1, 1)
    # plt.plot(t, true_pos, "black", label="True Pos", linewidth=1)
    # plt.plot(t, true_pos_addNoise, "blue", label="True Pos addNoise", linestyle="dotted")
    # plt.plot(t, est_pos, label="KF Est Pos by Q", linewidth=1)
    # plt.plot(t, est_pos0, label="KF Est Pos by Q0", linewidth=1)
    # plt.plot(t, est_pos1, label="KF Est Pos by Q1", linewidth=1)
    # plt.plot(t, est_pos2, label="KF Est Pos by Q2", linewidth=1)
    # plt.plot(t, est_pos13, label="KF Est Pos by Q13", linewidth=1)
    # plt.xlabel("t")
    # plt.ylabel("pos")
    # plt.legend(loc='upper right')
    # plt.title("KF with different model assumption", loc="center")

    # # vel
    # # plt.figure(figsize=(10, 6))
    # plt.subplot(3, 1, 2)
    # plt.plot(t, true_vel, "black", label="True Velocity", linewidth=1)
    # plt.plot(t, est_vel, label="KF Est Vel by with Q", linewidth=1)
    # plt.plot(t, est_vel0, label="KF Est Vel by with Q0", linewidth=1)
    # plt.plot(t, est_vel1, label="KF Est Vel by with Q1", linewidth=1)
    # plt.plot(t, est_vel2, label="KF Est Vel by with Q2", linewidth=1)
    # plt.plot(t, est_vel13, label="KF Est Vel by with Q13", linewidth=1)
    # plt.xlabel("t")
    # plt.ylabel("vel")
    # # plt.legend()

    # # acc
    # # plt.figure(figsize=(10, 6))
    # plt.subplot(3, 1, 3)
    # plt.plot(t, true_acc, "black", label="True Acceleration", linewidth=1)
    # plt.plot(t, est_acc, label="KF Est Acc by with Q", linewidth=1)
    # plt.plot(t, est_acc0, label="KF Est Acc by with Q0", linewidth=1)
    # plt.plot(t, est_acc1, label="KF Est Acc by with Q1", linewidth=1)
    # plt.plot(t, est_acc2, label="KF Est Acc by with Q2", linewidth=1)
    # plt.plot(t, est_acc13, label="KF Est Acc by with Q13", linewidth=1)
    # plt.xlabel("t")
    # plt.ylabel("acc")
    # # plt.legend()
    # plt.tight_layout()

    # #-------------------------------------不同估測方法計算Q比較 vel:CMD-------------------------------------------#
    # # pos
    # plt.figure(figsize=(8, 6))
    # plt.subplot(3, 1, 1)
    # plt.plot(t, true_pos, "black", label="True Pos", linewidth=1)
    # plt.plot(t, true_pos_addNoise, "blue", label="True Pos addNoise", linestyle="dotted")
    # plt.plot(t, est_pos3, label="KF Est Pos by Q3 ", linewidth=1)
    # plt.plot(t, est_pos4, label="KF Est Pos by Q4", linewidth=1)
    # plt.plot(t, est_pos5, label="KF Est Pos by Q5", linewidth=1, linestyle="--")
    # plt.legend(loc='upper right')
    # plt.xlabel("t")
    # plt.ylabel("pos")
    # plt.title("KF with different estimator calculate Q -vel:CMD")

    # # vel
    # # plt.figure(figsize=(10, 6))
    # plt.subplot(3, 1, 2)
    # plt.plot(t, true_vel, "black", label="True Velocity", linewidth=1)
    # plt.plot(t, est_vel3, label="KF Est Vel by with Q3", linewidth=1)
    # plt.plot(t, est_vel4, label="KF Est Vel by with Q4", linewidth=1)
    # plt.plot(t, est_vel5, label="KF Est Vel by with Q5", linewidth=1, linestyle="--")
    # plt.xlabel("t")
    # plt.ylabel("vel")
    # # plt.legend()

    # # acc
    # # plt.figure(figsize=(10, 6))
    # plt.subplot(3, 1, 3)
    # plt.plot(t, true_acc, "black", label="True Acceleration", linewidth=1)
    # plt.plot(t, est_acc3, label="KF Est Acctby ion with Q3", linewidth=1)
    # plt.plot(t, est_acc4, label="KF Est Acctby ion with Q4", linewidth=1)
    # plt.plot(t, est_acc5, label="KF Est Acctby ion with Q5", linewidth=1, linestyle="--")
    # plt.xlabel("t")
    # plt.ylabel("acc")
    # # plt.legend()
    # plt.tight_layout()

    # #-------------------------------------不同估測方法計算Q比較 vel:LSF-------------------------------------------#
    # # pos
    # plt.figure(figsize=(8, 6))
    # plt.subplot(3, 1, 1)
    # plt.plot(t, true_pos, "black", label="True Pos", linewidth=1)
    # plt.plot(t, true_pos_addNoise, "blue", label="True Pos addNoise", linestyle="dotted")
    # plt.plot(t, est_pos6, label="KF Est Pos by Q6", linewidth=1)
    # plt.plot(t, est_pos7, label="KF Est Pos by Q7", linewidth=1)
    # plt.plot(t, est_pos8, label="KF Est Pos by Q8", linewidth=1, linestyle="--")
    # plt.legend(loc='upper right')
    # plt.xlabel("t")
    # plt.ylabel("pos")
    # plt.title("KF with different estimator calculate Q -vel:LSF")

    # # vel
    # # plt.figure(figsize=(10, 6))
    # plt.subplot(3, 1, 2)
    # plt.plot(t, true_vel, "black", label="True Velocity", linewidth=1)
    # plt.plot(t, est_vel6, label="KF Est Vel by with Q6", linewidth=1)
    # plt.plot(t, est_vel7, label="KF Est Vel by with Q7", linewidth=1)
    # plt.plot(t, est_vel8, label="KF Est Vel by with Q8", linewidth=1, linestyle="--")
    # plt.xlabel("t")
    # plt.ylabel("vel")
    # # plt.legend()

    # # acc
    # # plt.figure(figsize=(10, 6))
    # plt.subplot(3, 1, 3)
    # plt.plot(t, true_acc, "black", label="True Acceleration", linewidth=1)
    # plt.plot(t, est_acc6, label="KF Est Acctby ion with Q6", linewidth=1)
    # plt.plot(t, est_acc7, label="KF Est Acctby ion with Q7", linewidth=1)
    # plt.plot(t, est_acc8, label="KF Est Acctby ion with Q8", linewidth=1, linestyle="--")
    # plt.xlabel("t")
    # plt.ylabel("acc")
    # # plt.legend()
    # plt.tight_layout()

    # #-------------------------------------不同估測方法計算Q比較 vel:CFD-------------------------------------------#
    # # pos
    # plt.figure(figsize=(8, 6))
    # plt.subplot(3, 1, 1)
    # plt.plot(t, true_pos, "black", label="True Pos", linewidth=1)
    # plt.plot(t, true_pos_addNoise, "blue", label="True Pos addNoise", linestyle="dotted")
    # # plt.plot(t, est_pos1, label="KF Est Pos by Q1", linewidth=1)
    # # plt.plot(t, est_pos2, label="KF Est Pos by Q2", linewidth=1)
    # # plt.plot(t, est_pos3, label="KF Est Pos by Q3 ", linewidth=1)
    # # plt.plot(t, est_pos4, label="KF Est Pos by Q4", linewidth=1)
    # # plt.plot(t, est_pos5, label="KF Est Pos by Q5", linewidth=1)
    # # plt.plot(t, est_pos6, label="KF Est Pos by Q6", linewidth=1)
    # # plt.plot(t, est_pos7, label="KF Est Pos by Q7", linewidth=1)
    # # plt.plot(t, est_pos8, label="KF Est Pos by Q8", linewidth=1)
    # plt.plot(t, est_pos9, label="KF Est Pos by Q9", linewidth=1)
    # plt.plot(t, est_pos10, label="KF Est Pos by Q10", linewidth=1)
    # plt.plot(t, est_pos11, label="KF Est Pos by Q11", linewidth=1, linestyle="--")
    # plt.legend(loc='upper right')
    # plt.xlabel("t")
    # plt.ylabel("pos")
    # plt.title("KF with different estimator calculate Q -vel:CFD")

    # # vel
    # # plt.figure(figsize=(10, 6))
    # plt.subplot(3, 1, 2)
    # plt.plot(t, true_vel, "black", label="True Velocity", linewidth=1)
    # # plt.plot(t, est_vel1, label="KF Est Vel by with Q1", linewidth=1)
    # # plt.plot(t, est_vel2, label="KF Est Vel by with Q2", linewidth=1)
    # # plt.plot(t, est_vel3, label="KF Est Vel by with Q3", linewidth=1)
    # # plt.plot(t, est_vel4, label="KF Est Vel by with Q4", linewidth=1)
    # # plt.plot(t, est_vel5, label="KF Est Vel by with Q5", linewidth=1)
    # # plt.plot(t, est_vel6, label="KF Est Vel by with Q6", linewidth=1)
    # # plt.plot(t, est_vel7, label="KF Est Vel by with Q7", linewidth=1)
    # # plt.plot(t, est_vel8, label="KF Est Vel by with Q8", linewidth=1)
    # plt.plot(t, est_vel9, label="KF Est Vel by with Q9", linewidth=1)
    # plt.plot(t, est_vel10, label="KF Est Vel by with Q10", linewidth=1)
    # plt.plot(t, est_vel11, label="KF Est Vel by with Q11", linewidth=1, linestyle="--")
    # plt.xlabel("t")
    # plt.ylabel("vel")
    # # plt.legend()

    # # acc
    # # plt.figure(figsize=(10, 6))
    # plt.subplot(3, 1, 3)
    # plt.plot(t, true_acc, "black", label="True Acceleration", linewidth=1)
    # # plt.plot(t, est_acc1, label="KF Est Acctby ion with Q1", linewidth=1)
    # # plt.plot(t, est_acc2, label="KF Est Acctby ion with Q2", linewidth=1)
    # # plt.plot(t, est_acc3, label="KF Est Acctby ion with Q3", linewidth=1)
    # # plt.plot(t, est_acc4, label="KF Est Acctby ion with Q4", linewidth=1)
    # # plt.plot(t, est_acc5, label="KF Est Acctby ion with Q5", linewidth=1)
    # # plt.plot(t, est_acc6, label="KF Est Acctby ion with Q6", linewidth=1)
    # # plt.plot(t, est_acc7, label="KF Est Acctby ion with Q7", linewidth=1)
    # # plt.plot(t, est_acc8, label="KF Est Acctby ion with Q8", linewidth=1)
    # plt.plot(t, est_acc9, label="KF Est Acctby ion with Q9", linewidth=1)
    # plt.plot(t, est_acc10, label="KF Est Acctby ion with Q10", linewidth=1)
    # plt.plot(t, est_acc11, label="KF Est Acctby ion with Q11", linewidth=1, linestyle="--")
    # plt.xlabel("t")
    # plt.ylabel("acc")
    # # plt.legend()
    # plt.tight_layout()

#---------------------------------------AKF結果-------------------------------------------#
    # Plot Q
    # print("est_pos =", est_pos.shape)
    plt.figure(figsize=(8, 6))
    # plt.figure(1)
    plt.subplot(3, 1, 1)
    t = np.arange(0, total_time*1000 - 2, 1)
    plt.plot(t, true_pos[: len(true_pos)-2], "black", label="True Pos", linewidth=1)
    plt.plot(t, true_pos_addNoise[: len(true_pos_addNoise)-2], "blue", label="True Pos addNoise", linestyle="dotted")
    plt.plot(t, pose[: len(pose)-2], "red", label="AKF Est Pos", linewidth=1)
    # plt.plot(t, est_pos, label="KF Est Pos by Q", linewidth=1)
    # plt.plot(t, est_pos0, label="KF Est Pos by Q0", linewidth=1)
    # plt.plot(t, est_pos1, label="KF Est Pos by Q1", linewidth=1)
    # plt.plot(t, est_pos2, label="KF Est Pos by Q2", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("pos")
    plt.title("AKF", loc="center")

    # vel
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 2)
    plt.plot(t, true_vel[: len(true_vel)-2], "black", label="True Velocity", linewidth=1)
    plt.plot(t, vele[: len(vele)-2], "red", label="AKF Est Vel", linewidth=1)
    # plt.plot(t, est_vel, label="KF Est Vel by with Q", linewidth=1)
    # plt.plot(t, est_vel0, label="KF Est Vel by with Q0", linewidth=1)
    # plt.plot(t, est_vel1, label="KF Est Vel by with Q1", linewidth=1)
    # plt.plot(t, est_vel2, label="KF Est Vel by with Q2", linewidth=1)
    plt.xlabel("t")
    plt.ylabel("vel")
    # plt.legend()

    # acc
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 3)
    plt.plot(t, true_acc[: len(true_acc)-2], "black", label="True Acceleration", linewidth=1)
    plt.plot(t, acce[: len(acce)-2], "red", label="AKF Est Acc", linewidth=1)
    # plt.plot(t, est_acc, label="KF Est Acctby ion with Q", linewidth=1)
    # plt.plot(t, est_acc0, label="KF Est Acctby ion with Q0", linewidth=1)
    # plt.plot(t, est_acc1, label="KF Est Acctby ion with Q1", linewidth=1)
    # plt.plot(t, est_acc2, label="KF Est Acctby ion with Q2", linewidth=1)
    plt.xlabel("t")
    plt.ylabel("acc")
    # plt.legend()
    plt.tight_layout()

    #---------------------------------取穩定後的數據--------------------------------------#
    # print("est_pos =", est_pos.shape)
    plt.figure(figsize=(8, 6))
    # plt.figure(1)
    plt.subplot(3, 1, 1)
    t = np.arange(0, total_time*1000 - 100-2, 1)
    plt.plot(t, true_pos[100: len(true_pos)-2], "black", label="True Pos", linewidth=1)
    plt.plot(t, true_pos_addNoise[100: len(true_pos_addNoise)-2], "blue", label="True Pos addNoise", linestyle="dotted")
    plt.plot(t, pose[100: len(pose)-2], "red", label="AKF Est Pos", linewidth=1)
    # plt.plot(t, est_pos, label="KF Est Pos by Q", linewidth=1)
    # plt.plot(t, est_pos0, label="KF Est Pos by Q0", linewidth=1)
    # plt.plot(t, est_pos1, label="KF Est Pos by Q1", linewidth=1)
    # plt.plot(t, est_pos2, label="KF Est Pos by Q2", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("pos")
    plt.title("AKF after stability", loc="center")

    # vel
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 2)
    plt.plot(t, true_vel[100: len(true_vel)-2], "black", label="True Velocity", linewidth=1)
    plt.plot(t, vele[100: len(vele)-2], "red", label="AKF Est Vel", linewidth=1)
    # plt.plot(t, est_vel, label="KF Est Vel by with Q", linewidth=1)
    # plt.plot(t, est_vel0, label="KF Est Vel by with Q0", linewidth=1)
    # plt.plot(t, est_vel1, label="KF Est Vel by with Q1", linewidth=1)
    # plt.plot(t, est_vel2, label="KF Est Vel by with Q2", linewidth=1)
    plt.xlabel("t")
    plt.ylabel("vel")
    # plt.legend()

    # acc
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 3)
    plt.plot(t, true_acc[100: len(true_acc)-2], "black", label="True Acceleration", linewidth=1)
    plt.plot(t, acce[100: len(acce)-2], "red", label="AKF Est Acc", linewidth=1)
    # plt.plot(t, est_acc, label="KF Est Acctby ion with Q", linewidth=1)
    # plt.plot(t, est_acc0, label="KF Est Acctby ion with Q0", linewidth=1)
    # plt.plot(t, est_acc1, label="KF Est Acctby ion with Q1", linewidth=1)
    # plt.plot(t, est_acc2, label="KF Est Acctby ion with Q2", linewidth=1)
    # plt.legend()
    plt.xlabel("t")
    plt.ylabel("acc")
    plt.tight_layout()

    #---------------------------------畫出Q值更新過程--------------------------------------#
    plt.figure(figsize=(8, 6))
    # plt.scatter(range(1, len(Q_vel) + 1), Q_vel, color='b', label="VEL_Q", s=4)
    plt.plot(range(1, len(Q_vel) + 1), Q_vel, color='b', linestyle='-', marker='x', label="VEL_Q")
    plt.xlabel('Iteration')
    plt.ylabel('Q Values')
    plt.title('Q_vel Values over Iterations')
    plt.legend(loc="upper right")

    plt.figure(figsize=(8, 6))
    # plt.scatter(range(1, len(Q_acc) + 1), Q_acc, color='r', label="ACC_Q", s=4)
    plt.plot(range(1, len(Q_acc) + 1), Q_acc, color='r', linestyle='-', marker='o', label="ACC_Q")
    plt.xlabel('Iteration')
    plt.ylabel('Q_acc Values')
    plt.title('Q_acc Values over Iterations')
    plt.legend(loc="upper right")

    print('--------------------------------AKF---------------------------------')
    print("mean Q_pos Values = ", np.mean(Q_pos))
    print("mean Q_vel Values = ", np.mean(Q_vel))
    # print(len(Q_vel))
    print("mean Q_acc Values = ", np.mean(Q_acc))
    # print("Q@0err: ", Q_0err)
    print('-----------------------------------------------------------------')

    #---------------------------------------劃出位置、速度、加速度誤差-----------------------------------------#
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    # t = np.arange(0, total_time*1000 - 100-2, 1)
    plt.plot(u_p_values, "black", label="Pos error", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("pos")
    plt.title("Error of POS, VEL, ACC", loc="center")

    # vel
    plt.subplot(3, 1, 2)
    plt.plot(u_v_values, "blue", label="Vel error", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("vel")

    # acc
    plt.subplot(3, 1, 3)
    plt.plot(u_a_values, "red", label="Acc error", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("acc")
    plt.tight_layout()

    #---------------------------------------劃出位置、速度、加速度誤差-----------------------------------------#
    plt.figure(figsize=(8, 6))
    # plt.subplot(3, 1, 1)
    # t = np.arange(0, total_time*1000 - 100-2, 1)
    # plt.plot(u_p_values, "black", label="Pos error", linewidth=1)
    # plt.legend(loc='upper right')
    # plt.xlabel("t")
    # plt.ylabel("pos")
    # plt.title("Error of POS, VEL, ACC", loc="center")

    # vel
    plt.subplot(2, 1, 1)
    plt.plot(v_cov_data, "blue", label="Vel by cov", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("vel")
    plt.title("VEL, ACC by cov", loc="center")

    # acc
    plt.subplot(2, 1, 2)
    plt.plot(a_cov_data, "red", label="Acc by cov", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("acc")
    plt.tight_layout()
    
#-------------------------------------AKF取得error最小之Q值-------------------------------------------#
    # pos
    t = np.arange(0, total_time*1000 , 1)
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, true_pos, "black", label="True Pos", linewidth=1)
    plt.plot(t, true_pos_addNoise, "blue", label="True Pos addNoise", linestyle="dotted")
    plt.plot(t, pose, label="AKF Est Pos", linewidth=1)
    plt.plot(t, est_pos12, "#FF69B4", label="KF Est Pos by Q12", linewidth=1)
    plt.plot(t, est_pos13, label="KF Est Pos by Q13", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("pos")
    plt.title("KF with AKF min error Q")

    # vel
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 2)
    plt.plot(t, true_vel, "black", label="True Velocity", linewidth=1)
    plt.plot(t, vele, label="AKF Est Vel", linewidth=1)
    plt.plot(t, est_vel12, "#FF69B4", label="KF Est Vel by with Q12", linewidth=1)
    plt.plot(t, est_vel13, label="KF Est Vel by with Q13", linewidth=1)
    # plt.legend()
    plt.xlabel("t")
    plt.ylabel("vel")

    # acc
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 3)
    plt.plot(t, true_acc, "black", label="True Acceleration", linewidth=1)
    plt.plot(t, acce, label="AKF Est Acc", linewidth=1)
    plt.plot(t, est_acc12, "#FF69B4", label="KF Est Acctby ion with Q12", linewidth=1)
    plt.plot(t, est_acc13, label="KF Est Acctby ion with Q13", linewidth=1)
    # plt.legend()
    plt.xlabel("t")
    plt.ylabel("acc")
    plt.tight_layout()

#---------------------------------------AEKF結果-------------------------------------------#
    # Plot Q
    # print("est_pos =", est_pos.shape)
    plt.figure(figsize=(8, 6))
    # plt.figure(1)
    plt.subplot(3, 1, 1)
    t = np.arange(0, total_time*1000 - 2, 1)
    plt.plot(t, true_pos[: len(true_pos)-2], "black", label="True Pos", linewidth=1)
    plt.plot(t, true_pos_addNoise[: len(true_pos_addNoise)-2], "blue", label="True Pos addNoise", linestyle="dotted")
    plt.plot(t, AEKF_pose[: len(pose)-2], "red", label="AEKF Est Pos", linewidth=1)
    # plt.plot(t, est_pos, label="KF Est Pos by Q", linewidth=1)
    # plt.plot(t, est_pos0, label="KF Est Pos by Q0", linewidth=1)
    # plt.plot(t, est_pos1, label="KF Est Pos by Q1", linewidth=1)
    # plt.plot(t, est_pos2, label="KF Est Pos by Q2", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("pos")
    plt.title("AEKF", loc="center")

    # vel
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 2)
    plt.plot(t, true_vel[: len(true_vel)-2], "black", label="True Velocity", linewidth=1)
    plt.plot(t, AEKF_vele[: len(vele)-2], "red", label="AEKF Est Vel", linewidth=1)
    # plt.plot(t, est_vel, label="KF Est Vel by with Q", linewidth=1)
    # plt.plot(t, est_vel0, label="KF Est Vel by with Q0", linewidth=1)
    # plt.plot(t, est_vel1, label="KF Est Vel by with Q1", linewidth=1)
    # plt.plot(t, est_vel2, label="KF Est Vel by with Q2", linewidth=1)
    plt.xlabel("t")
    plt.ylabel("vel")
    # plt.legend()

    # acc
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 3)
    plt.plot(t, true_acc[: len(true_acc)-2], "black", label="True Acceleration", linewidth=1)
    plt.plot(t, AEKF_acce[: len(acce)-2], "red", label="AEKF Est Acc", linewidth=1)
    # plt.plot(t, est_acc, label="KF Est Acctby ion with Q", linewidth=1)
    # plt.plot(t, est_acc0, label="KF Est Acctby ion with Q0", linewidth=1)
    # plt.plot(t, est_acc1, label="KF Est Acctby ion with Q1", linewidth=1)
    # plt.plot(t, est_acc2, label="KF Est Acctby ion with Q2", linewidth=1)
    plt.xlabel("t")
    plt.ylabel("acc")
    # plt.legend()
    plt.tight_layout()

    #---------------------------------取穩定後的數據--------------------------------------#
    # print("est_pos =", est_pos.shape)
    plt.figure(figsize=(8, 6))
    # plt.figure(1)
    plt.subplot(3, 1, 1)
    t = np.arange(0, total_time*1000 - 100-2, 1)
    plt.plot(t, true_pos[100: len(true_pos)-2], "black", label="True Pos", linewidth=1)
    plt.plot(t, true_pos_addNoise[100: len(true_pos_addNoise)-2], "blue", label="True Pos addNoise", linestyle="dotted")
    plt.plot(t, AEKF_pose[100: len(pose)-2], "red", label="AEKF Est Pos", linewidth=1)
    # plt.plot(t, est_pos, label="KF Est Pos by Q", linewidth=1)
    # plt.plot(t, est_pos0, label="KF Est Pos by Q0", linewidth=1)
    # plt.plot(t, est_pos1, label="KF Est Pos by Q1", linewidth=1)
    # plt.plot(t, est_pos2, label="KF Est Pos by Q2", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("pos")
    plt.title("AEKF after stability", loc="center")

    # vel
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 2)
    plt.plot(t, true_vel[100: len(true_vel)-2], "black", label="True Velocity", linewidth=1)
    plt.plot(t, AEKF_vele[100: len(vele)-2], "red", label="AEKF Est Vel", linewidth=1)
    # plt.plot(t, est_vel, label="KF Est Vel by with Q", linewidth=1)
    # plt.plot(t, est_vel0, label="KF Est Vel by with Q0", linewidth=1)
    # plt.plot(t, est_vel1, label="KF Est Vel by with Q1", linewidth=1)
    # plt.plot(t, est_vel2, label="KF Est Vel by with Q2", linewidth=1)
    plt.xlabel("t")
    plt.ylabel("vel")
    # plt.legend()

    # acc
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 3)
    plt.plot(t, true_acc[100: len(true_acc)-2], "black", label="True Acceleration", linewidth=1)
    plt.plot(t, AEKF_acce[100: len(acce)-2], "red", label="AEKF Est Acc", linewidth=1)
    # plt.plot(t, est_acc, label="KF Est Acctby ion with Q", linewidth=1)
    # plt.plot(t, est_acc0, label="KF Est Acctby ion with Q0", linewidth=1)
    # plt.plot(t, est_acc1, label="KF Est Acctby ion with Q1", linewidth=1)
    # plt.plot(t, est_acc2, label="KF Est Acctby ion with Q2", linewidth=1)
    # plt.legend()
    plt.xlabel("t")
    plt.ylabel("acc")
    plt.tight_layout()

    #---------------------------------畫出Q值更新過程--------------------------------------#
    plt.figure(figsize=(8, 6))
    # plt.scatter(range(1, len(Q_vel) + 1), Q_vel, color='b', label="VEL_Q", s=4)
    plt.plot(range(1, len(AEKF_Q_vel) + 1), AEKF_Q_vel, color='b', linestyle='-', marker='x', label="VEL_Q")
    plt.xlabel('Iteration')
    plt.ylabel('Q Values')
    plt.title('AEKF_Q_vel Values over Iterations')
    plt.legend(loc="upper right")

    plt.figure(figsize=(8, 6))
    # plt.scatter(range(1, len(Q_acc) + 1), Q_acc, color='r', label="ACC_Q", s=4)
    plt.plot(range(1, len(AEKF_Q_acc) + 1), AEKF_Q_acc, color='r', linestyle='-', marker='o', label="ACC_Q")
    plt.xlabel('Iteration')
    plt.ylabel('Q_acc Values')
    plt.title('AEKF_Q_acc Values over Iterations')
    plt.legend(loc="upper right")

    print('---------------------------------AEKF--------------------------------')
    print("mean AEKF_Q_pos Values = ", np.mean(Q_pos))
    print("mean AEKF_Q_vel Values = ", np.mean(Q_vel))
    # print(len(Q_vel))
    print("mean AEKF_Q_acc Values = ", np.mean(Q_acc))
    # print("Q@0err: ", Q_0err)
    print('-----------------------------------------------------------------')
    
#-------------------------------------AEKF取得error最小之Q值-------------------------------------------#
    # pos
    t = np.arange(0, total_time*1000 , 1)
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, true_pos, "black", label="True Pos", linewidth=1)
    plt.plot(t, true_pos_addNoise, "blue", label="True Pos addNoise", linestyle="dotted")
    plt.plot(t, AEKF_pose, label="AEKF Est Pos", linewidth=1)
    plt.plot(t, est_pos12, "#FF69B4", label="KF Est Pos by Q12", linewidth=1)
    plt.plot(t, est_pos13, label="KF Est Pos by Q13", linewidth=1)
    plt.legend(loc='upper right', ncol=2)
    plt.xlabel("t")
    plt.ylabel("pos")
    plt.title("KF with AEKF min error Q")

    # vel
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 2)
    plt.plot(t, true_vel, "black", label="True Velocity", linewidth=1)
    plt.plot(t, AEKF_vele, label="AEKF Est Vel", linewidth=1)
    plt.plot(t, est_vel12, "#FF69B4", label="KF Est Vel by with Q12", linewidth=1)
    plt.plot(t, est_vel13, label="KF Est Vel by with Q13", linewidth=1)
    # plt.legend()
    plt.xlabel("t")
    plt.ylabel("vel")

    # acc
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 3)
    plt.plot(t, true_acc, "black", label="True Acceleration", linewidth=1)
    plt.plot(t, AEKF_acce, label="AEKF Est Acc", linewidth=1)
    plt.plot(t, est_acc12, "#FF69B4", label="KF Est Acctby ion with Q12", linewidth=1)
    plt.plot(t, est_acc13, label="KF Est Acctby ion with Q13", linewidth=1)
    # plt.legend()
    plt.xlabel("t")
    plt.ylabel("acc")
    plt.tight_layout()

#-------------------------------------AKF、AEKF比較-------------------------------------------#
    # pos
    # t = np.arange(0, total_time - 100 *dt, dt)
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, true_pos, "black", label="True Pos", linewidth=1)
    plt.plot(t, true_pos_addNoise, "blue", label="True Pos addNoise", linestyle="dotted")
    plt.plot(t, pose, label="AKF Est Pos", linewidth=1)
    plt.plot(t, AEKF_pose, label="AEKF Est Pos", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("pos")
    plt.title("AKF&AEKF estimate result comparison")

    # vel
    plt.subplot(3, 1, 2)
    plt.plot(t, true_vel, "black", label="True Velocity", linewidth=1)
    plt.plot(t, vele, label="AKF Est Vel", linewidth=1)
    plt.plot(t, AEKF_vele, label="AEKF Est Vel", linewidth=1) 
    plt.xlabel("t")
    plt.ylabel("vel")

    # acc
    plt.subplot(3, 1, 3)
    plt.plot(t, true_acc, "black", label="True Acceleration", linewidth=1)
    plt.plot(t, acce, label="AKF Est Acc", linewidth=1)
    plt.plot(t, AEKF_acce, label="AEKF Est Acc", linewidth=1)
    plt.xlabel("t")
    plt.ylabel("acc")
    plt.tight_layout()

#-------------------------------------不同Q估測結果比較 AKF、0、2、4、7、10-------------------------------------------#
    # pos
    # t = np.arange(0, total_time - 100 *dt, dt)
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, true_pos, "black", label="True Pos", linewidth=1)
    plt.plot(t, true_pos_addNoise, "blue", label="True Pos addNoise", linestyle="dotted")
    plt.plot(t, pose, label="AKF Est Pos", linewidth=1)
    plt.plot(t, AEKF_pose, label="AEKF Est Pos", linewidth=1)
    # plt.plot(t, est_pos, label="KF Est Pos by Q", linewidth=1)
    plt.plot(t, est_pos0, label="KF Est Pos by Q0", linewidth=1)
    # plt.plot(t, est_pos1, label="KF Est Pos by Q1", linewidth=1)
    plt.plot(t, est_pos2, label="KF Est Pos by Q2", linewidth=1)
    # plt.plot(t, est_pos3, label="KF Est Pos by Q3 ", linewidth=1)
    plt.plot(t, est_pos4, label="KF Est Pos by Q4", linewidth=1)
    # plt.plot(t, est_pos5, label="KF Est Pos by Q5", linewidth=1)
    # plt.plot(t, est_pos6, label="KF Est Pos by Q6", linewidth=1)
    plt.plot(t, est_pos7, label="KF Est Pos by Q7", linewidth=1)
    # plt.plot(t, est_pos8, label="KF Est Pos by Q8", linewidth=1)
    # plt.plot(t, est_pos9, label="KF Est Pos by Q9", linewidth=1)
    plt.plot(t, est_pos10, label="KF Est Pos by Q10", linewidth=1)
    # plt.plot(t, est_pos11, label="KF Est Pos by Q11", linewidth=1)
    plt.plot(t, est_pos12, label="KF Est Pos by Q12", linewidth=1)
    plt.plot(t, est_pos13, label="KF Est Pos by Q13", linewidth=1)
    plt.legend(loc='upper right', ncol=2)
    plt.xlabel("t")
    plt.ylabel("pos")
    plt.title("KF with AKF & different Q estimate result comparison")

    # vel
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 2)
    plt.plot(t, true_vel, "black", label="True Velocity", linewidth=1)
    plt.plot(t, vele, label="AKF Est Vel", linewidth=1)
    plt.plot(t, AEKF_vele, label="AEKF Est Vel", linewidth=1) 
    # plt.plot(t, est_vel, label="KF Est Vel by with Q", linewidth=1)
    plt.plot(t, est_vel0, label="KF Est Vel by with Q0", linewidth=1)
    # plt.plot(t, est_vel1, label="KF Est Vel by with Q1", linewidth=1)
    plt.plot(t, est_vel2, label="KF Est Vel by with Q2", linewidth=1)
    # plt.plot(t, est_vel3, label="KF Est Vel by with Q3", linewidth=1)
    plt.plot(t, est_vel4, label="KF Est Vel by with Q4", linewidth=1)
    # plt.plot(t, est_vel5, label="KF Est Vel by with Q5", linewidth=1)
    # plt.plot(t, est_vel6, label="KF Est Vel by with Q6", linewidth=1)
    plt.plot(t, est_vel7, label="KF Est Vel by with Q7", linewidth=1)
    # plt.plot(t, est_vel8, label="KF Est Vel by with Q8", linewidth=1)
    # plt.plot(t, est_vel9, label="KF Est Vel by with Q9", linewidth=1)
    plt.plot(t, est_vel10, label="KF Est Vel by with Q10", linewidth=1)
    # plt.plot(t, est_vel11, label="KF Est Vel by with Q11", linewidth=1)
    plt.plot(t, est_vel12, label="KF Est Vel by with Q12", linewidth=1)
    plt.plot(t, est_vel13, label="KF Est Vel by with Q13", linewidth=1)
    # plt.legend()
    plt.xlabel("t")
    plt.ylabel("vel")

    # acc
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 3)
    plt.plot(t, true_acc, "black", label="True Acceleration", linewidth=1)
    plt.plot(t, acce, label="AKF Est Acc", linewidth=1)
    plt.plot(t, AEKF_acce, label="AEKF Est Acc", linewidth=1)
    # plt.plot(t, est_acc, label="KF Est Acctby ion with Q", linewidth=1)
    plt.plot(t, est_acc0, label="KF Est Acctby ion with Q0", linewidth=1)
    # plt.plot(t, est_acc1, label="KF Est Acctby ion with Q1", linewidth=1)
    plt.plot(t, est_acc2, label="KF Est Acctby ion with Q2", linewidth=1)
    # plt.plot(t, est_acc3, label="KF Est Acctby ion with Q3", linewidth=1)
    plt.plot(t, est_acc4, label="KF Est Acctby ion with Q4", linewidth=1)
    # plt.plot(t, est_acc5, label="KF Est Acctby ion with Q5", linewidth=1)
    # plt.plot(t, est_acc6, label="KF Est Acctby ion with Q6", linewidth=1)
    plt.plot(t, est_acc7, label="KF Est Acctby ion with Q7", linewidth=1)
    # plt.plot(t, est_acc8, label="KF Est Acctby ion with Q8", linewidth=1)
    # plt.plot(t, est_acc9, label="KF Est Acctby ion with Q9", linewidth=1)
    plt.plot(t, est_acc10, label="KF Est Acctby ion with Q10", linewidth=1)
    # plt.plot(t, est_acc11, label="KF Est Acctby ion with Q11", linewidth=1)
    plt.plot(t, est_acc12, label="KF Est Acctby ion with Q12", linewidth=1)
    plt.plot(t, est_acc13, label="KF Est Acctby ion with Q13", linewidth=1)
    # plt.legend()
    plt.xlabel("t")
    plt.ylabel("acc")
    plt.tight_layout()

#-------------------------------------不同Q估測結果比較 AKF、0、2、4、7、10，取穩定後的-------------------------------------------#
    # pos
    t = np.arange(0, total_time*1000 - 200 *1, 1)
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, true_pos[200:], "black", label="True Pos", linewidth=1)
    plt.plot(t, true_pos_addNoise[200:], "blue", label="True Pos addNoise", linestyle="dotted")
    plt.plot(t, pose[200:], label="AKF Est Pos", linewidth=1)
    plt.plot(t, AEKF_pose[200:], label="AEKF Est Pos", linewidth=1)
    # plt.plot(t, est_pos,[200:] label="KF Est Pos by Q", linewidth=1)
    plt.plot(t, est_pos0[200:], label="KF Est Pos by Q0", linewidth=1)
    # plt.plot(t, est_pos1[200:], label="KF Est Pos by Q1", linewidth=1)
    plt.plot(t, est_pos2[200:], label="KF Est Pos by Q2", linewidth=1)
    # plt.plot(t, est_pos3[200:], label="KF Est Pos by Q3 ", linewidth=1)
    plt.plot(t, est_pos4[200:], label="KF Est Pos by Q4", linewidth=1)
    # plt.plot(t, est_pos5[200:], label="KF Est Pos by Q5", linewidth=1)
    # plt.plot(t, est_pos6[200:], label="KF Est Pos by Q6", linewidth=1)
    plt.plot(t, est_pos7[200:], label="KF Est Pos by Q7", linewidth=1)
    # plt.plot(t, est_pos8[200:], label="KF Est Pos by Q8", linewidth=1)
    # plt.plot(t, est_pos9[200:], label="KF Est Pos by Q9", linewidth=1)
    plt.plot(t, est_pos10[200:], label="KF Est Pos by Q10", linewidth=1)
    # plt.plot(t, est_pos11[200:], label="KF Est Pos by Q11", linewidth=1)
    plt.plot(t, est_pos12[200:], label="KF Est Pos by Q12", linewidth=1)
    plt.plot(t, est_pos13[200:], label="KF Est Pos by Q13", linewidth=1)
    plt.legend(loc='upper right', ncol=2)
    plt.xlabel("t")
    plt.ylabel("pos")
    plt.title("KF with AKF & different Q estimate result comparison")

    # vel
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 2)
    plt.plot(t, true_vel[200:], "black", label="True Velocity", linewidth=1)
    plt.plot(t, vele[200:], label="AKF Est Vel", linewidth=1)
    plt.plot(t, AEKF_vele[200:], label="AEKF Est Vel", linewidth=1)
    # plt.plot(t, est_vel,[200:] label="KF Est Vel by with Q", linewidth=1)
    plt.plot(t, est_vel0[200:], label="KF Est Vel by with Q0", linewidth=1)
    # plt.plot(t, est_vel1[200:], label="KF Est Vel by with Q1", linewidth=1)
    plt.plot(t, est_vel2[200:], label="KF Est Vel by with Q2", linewidth=1)
    # plt.plot(t, est_vel3[200:], label="KF Est Vel by with Q3", linewidth=1)
    plt.plot(t, est_vel4[200:], label="KF Est Vel by with Q4", linewidth=1)
    # plt.plot(t, est_vel5[200:], label="KF Est Vel by with Q5", linewidth=1)
    # plt.plot(t, est_vel6[200:], label="KF Est Vel by with Q6", linewidth=1)
    plt.plot(t, est_vel7[200:], label="KF Est Vel by with Q7", linewidth=1)
    # plt.plot(t, est_vel8[200:], label="KF Est Vel by with Q8", linewidth=1)
    # plt.plot(t, est_vel9[200:], label="KF Est Vel by with Q9", linewidth=1)
    plt.plot(t, est_vel10[200:], label="KF Est Vel by with Q10", linewidth=1)
    # plt.plot(t, est_vel11[200:], label="KF Est Vel by with Q11", linewidth=1)
    plt.plot(t, est_vel12[200:], label="KF Est Vel by with Q12", linewidth=1)
    plt.plot(t, est_vel13[200:], label="KF Est Vel by with Q13", linewidth=1)
    # plt.legend()
    plt.xlabel("t")
    plt.ylabel("vel")

    # acc
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 3)
    plt.plot(t, true_acc[200:], "black", label="True Acceleration", linewidth=1)
    plt.plot(t, acce[200:], label="AKF Est Acc", linewidth=1)
    plt.plot(t, AEKF_acce[200:], label="AEKF Est Acc", linewidth=1)
    # plt.plot(t, est_acc,[200:] label="KF Est Acctby ion with Q", linewidth=1)
    plt.plot(t, est_acc0[200:], label="KF Est Acctby ion with Q0", linewidth=1)
    # plt.plot(t, est_acc1[200:], label="KF Est Acctby ion with Q1", linewidth=1)
    plt.plot(t, est_acc2[200:], label="KF Est Acctby ion with Q2", linewidth=1)
    # plt.plot(t, est_acc3[200:], label="KF Est Acctby ion with Q3", linewidth=1)
    plt.plot(t, est_acc4[200:], label="KF Est Acctby ion with Q4", linewidth=1)
    # plt.plot(t, est_acc5[200:], label="KF Est Acctby ion with Q5", linewidth=1)
    # plt.plot(t, est_acc6[200:], label="KF Est Acctby ion with Q6", linewidth=1)
    plt.plot(t, est_acc7[200:], label="KF Est Acctby ion with Q7", linewidth=1)
    # plt.plot(t, est_acc8[200:], label="KF Est Acctby ion with Q8", linewidth=1)
    # plt.plot(t, est_acc9[200:], label="KF Est Acctby ion with Q9", linewidth=1)
    plt.plot(t, est_acc10[200:], label="KF Est Acctby ion with Q10", linewidth=1)
    # plt.plot(t, est_acc11[200:], label="KF Est Acctby ion with Q11", linewidth=1)
    plt.plot(t, est_acc12[200:], label="KF Est Acctby ion with Q12", linewidth=1)
    plt.plot(t, est_acc13[200:], label="KF Est Acctby ion with Q13", linewidth=1)
    # plt.legend()
    plt.xlabel("t")
    plt.ylabel("acc")
    plt.tight_layout()

    print("cov_new =", cov_new)

    plt.show()
