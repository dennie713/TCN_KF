import numpy as np

def EKF(dt, pos, SIGMA):
    def f(x, u):
        """非線性狀態轉移模型"""
        # 這裡用的是等加速度模型
        return np.array([x[0] + x[1] * dt + 0.5 * x[2] * dt**2,
                         x[1] + x[2] * dt,
                         x[2]])

    def h(x):
        """非線性觀測模型，這裡假設我們只觀測位置"""
        return np.array([x[0]])

    def F(x, u):
        """狀態轉移模型的雅可比矩陣"""
        return np.array([[1, dt, 0.5 * dt**2],
                         [0, 1, dt],
                         [0, 0, 1]])

    def H(x):
        """觀測模型的雅可比矩陣"""
        return np.array([[1, 0, 0]])

    # 初始狀態與協方差矩陣
    P = np.array([[1e-4, 0, 0],
                  [0, 1e-4, 0],
                  [0, 0, 1e-4]])  # 初始協方差
    # P = np.array([[1e-4, 1e-4, 1e-4],
    #                 [1e-4, 1e-4, 1e-4],
    #                 [1e-4, 1e-4, 1e-4]])

    # 噪聲協方差矩陣
    Q = SIGMA * np.array([[dt**4/4, dt**3/2, dt**2/2],
                          [dt**3/2, dt**2, dt],
                          [dt**2/2, dt, 1]])  # 系統過程噪聲
    # Q = np.array([[1.64521896e-06, 1.35746221e-05, 1.00125342e-04],
    #             [1.35746221e-05, 1.17514489e-04, 2.37265766e-03],
    #             [1.00125342e-04, 2.37265766e-03, 1.02581682e+00]]) # scara
    Q = np.array([[1.25124714e+01, 1.27516429e+00, 1.23241483e+02],
                    [1.27516429e+00, 1.74838225e+03, 8.02527454e+05],
                    [1.23241483e+02, 8.02527454e+05, 5.35272247e+08]]) # sin
    # Q = np.array([[0.01, 1.27516429e+00, 1.23241483e+02],
    #                 [1.27516429e+00, 0.01, 8.02527454e+05],
    #                 [1.23241483e+02, 8.02527454e+05, 0.01]]) # sin
    # Q = np.array ([[3.47644979e+00,1.27516429e+00,1.23241483e+02],
    #                [1.27516429e+00,6.37682644e+02,8.02527454e+05],
    #                [1.23241483e+02,8.02527454e+05,7.83710376e+05]]) # AKF min error Q

    R = 0.01  # 觀測噪聲協方差

    # 初始狀態估計
    xm = np.zeros((3, 1))  # 初始狀態 (位置、速度、加速度)
    Pm = P  # 初始協方差

    # 儲存結果
    pose = np.zeros(len(pos))
    vele = np.zeros(len(pos))
    acce = np.zeros(len(pos))

    for i in range(len(pos)):
        # 預測步驟
        Pp = np.dot(np.dot(F(xm, 0), Pm), F(xm, 0).T) + Q  # 預測協方差
        xp = f(xm, 0)  # 預測狀態

        # 更新步驟
        Km = np.dot(Pp, H(xm).T) / (np.dot(np.dot(H(xm), Pp), H(xm).T) + R)  # 卡爾曼增益
        y = (pos[i] - h(xp))  # 創新

        # 更新狀態估計
        xm = xp + np.dot(Km, y)
        Pm = np.dot((np.eye(3) - np.dot(Km, H(xm))), Pp)  # 更新協方差

        # 儲存結果
        pose[i] = xm[0, 0]
        vele[i] = xm[1, 0]
        acce[i] = xm[2, 0]

    return pose, vele, acce