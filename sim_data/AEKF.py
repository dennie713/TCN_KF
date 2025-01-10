import numpy as np
import matplotlib.pyplot as plt

## 速度\加速度Q值分開調整
def AEKF(dt, Pos, true_pos, true_vel, true_acc):
    A = np.array([[1, dt, 0.5*dt**2],
                  [0, 1, dt],
                  [0, 0, 1 ]])
    B = np.array([[0.5*dt**2],
                  [dt],
                  [1]])
    # u = AccCmd
    C = np.array([[1, 0, 0]]) 
    
    # Q = np.array([[0.0, 0, 0],
    #               [0, 0.0, 0],
    #               [0, 0, 0.0]]) 
    # 弦波Q值
    c = 5.0          # 振幅
    # 可調參數
    b = 0.01
    omega = 2 * np.pi # 角頻率（1 Hz）
    sigma_a2 = b * (c * omega**2)**2
    Q = sigma_a2 * np.array([[dt**5/20, dt**4/8, dt**3/6],
                                [dt**4/8, dt**3/3, dt**2/2],
                                [dt**3/6, dt**2/2, dt]])
    # 不用事先計算任何相關數值
    # Q = np.array([[0.01, 0, 0],
    #               [0, 1, 0],
    #               [0, 0, 100]]) 
    Q = np.array([[0.01, 0, 0],
                  [0, 0.01, 0],
                  [0, 0, 0.01]]) 
    # Q = np.array ([[1.37318165e+04,8.23669486e-03,8.23669486e+00],
    #                 [8.23669486e-03,3.52812352e+04,2.56892899e+04],
    #                 [8.23669486e+00,2.56892899e+04,1.30035431e+06]])
    # Q = np.array ([[1.15753083e-01,1.44723478e-01,1.44723478e+02],
    #                 [1.44723478e-01,5.24216051e+01,4.06162767e+05],
    #                 [1.44723478e+02,4.06162767e+05,2.01654777e+06]]) # 15000筆之最小Q without noise
    # Q = np.array ([[6.88107483e-02,1.44723478e-01,1.44723478e+02],
    #                 [1.44723478e-01,1.59461490e+01,4.06162767e+05],
    #                 [1.44723478e+02,4.06162767e+05,4.79293703e+06]]) # 15000筆之最小Q with noise
    # Q = np.array ([[2.29063303e-01,1.44723478e-01,1.44723478e+02],
    #                 [1.44723478e-01,8.74873112e+01,4.06162767e+05],
    #                 [1.44723478e+02,4.06162767e+05,3.99632011e+06]]) # 1000筆之最小Q without noise
    # Q = np.array ([[1.92789882e-01,1.44723478e-01,1.44723478e+02],
    #                 [1.44723478e-01,6.13500828e+01,4.06162767e+05],
    #                 [1.44723478e+02,4.06162767e+05,5.06160834e+06]]) # 1000筆之最小Q with noise
    # Q = np.array ([[1.95385210e-01,1.44723478e-01,1.44723478e+02],
    #                  [1.44723478e-01,6.27003545e+01,4.06162767e+05],
    #                  [1.44723478e+02,4.06162767e+05,5.19795402e+06]])
    
    # Q by variance Q = A@sigma@A.T
    # Q = np.array([[1.25124714e+01, 1.27516429e+00, 1.23241483e+02],
    #                 [1.27516429e+00, 1.74838225e+03, 8.02527454e+05],
    #                 [1.23241483e+02, 8.02527454e+05, 5.35272247e+08]])
    Q = np.array([[0.01, 1.27516429e+00, 1.23241483e+02],
                    [1.27516429e+00, 0.01, 8.02527454e+05],
                    [1.23241483e+02, 8.02527454e+05, 0.01]])
    
    # Q = np.eye(3) * 0.01
    # Q = np.array([[1e-6, 0, 0],
    #               [0, 0.05501, 0],
    #               [0, 0, 5501*289*10**-3]]) 
    
    # R = 0.00126*2 # 3*10e-4 # 3000 # 5000 # 5000 #150 #500 #1000 #100 #10 #1.5 #1 #與誤差有關 -> 影響平滑度
    # R = 0.0022151714012120923
    R = 0.01
    # R = 1.1966744310659976e-07
    # P = np.array([[1e-4, 0, 0],
    #               [0, 1e-4, 0],
    #               [0, 0, 1e-4]])
    P = np.array([[1e-4, 1e-4, 1e-4],
                    [1e-4, 1e-4, 1e-4],
                    [1e-4, 1e-4, 1e-4]])
    # P = np.array([[1e-8, 1e-8, 1e-8],
    #                 [1e-8, 1e-8, 1e-8],
    #                 [1e-8, 1e-8, 1e-8]])
    Wt = 0
    pose = np.zeros(len(Pos))
    vele = np.zeros(len(Pos))
    acce = np.zeros(len(Pos))
    xm = np.zeros((3, 1))  
    Pm = P
    u_p_values = []
    # u_v_values = []
    # u_a_values = []
    # y_values = []
    # delta_x_values = []
    Q_acc = []
    Q_vel = []
    Q_pos = []
    Q_save = []
    v_data = [0]
    a_data = [0, 0]

    
    for i in range(len(Pos)-2): # m = measurement;p = predict len(pos)
        Pp = np.dot(np.dot(A, Pm), A.T) + Q
        xp = np.dot(A, xm) + Wt
        # print("R_in =", R)
        Sk = np.dot(np.dot(C, Pp), C.T) + R
        Km = np.dot(Pp, C.T) / (np.dot(np.dot(C, Pp), C.T) + R)
        dk = (Pos[i] - np.dot(C, xp)) # dk = zk - hk*xk
        xm = xp + np.dot(Km, dk) # =x_hat
        ek = (Pos[i] - np.dot(C, xm)) # ek = zk - hk*xk+1
        Pm = np.dot((np.eye(3) - np.dot(Km, C)), Pp)
        # 儲存dk
        u_p_values.append(dk)

        u_p = (Pos[i] - xm[0])

        alpha = 0.9
        # ------------------調整R------------------ #
        R = alpha * R + (1 - alpha) * (ek * ek + np.dot(np.dot(C, Pp), C.T))
        print("R =", R)
        # ------------------調整Q------------------ #
        # Q = alpha * Q + (1 - alpha) * (Km @ Km.T * dk * dk) # Km @ Km.T
        # ------------------位置調整Q------------------ #
        # Q[0, 0] = alpha * Q[0, 0] + (1 - alpha) * (Km[0] * Km[0] * dk * dk) # Km @ Km.T
        Q[0, 0] = alpha * Q[0, 0] + (1 - alpha) * (Km.T @ Km * dk * dk) # Km @ Km.T
        # Q[0, 0] = alpha * Q[0, 0] + (1 - alpha) * (Km[0] *  Km[0] * dk * dk) # Km @ Km.T
        Q_pos.append(Q[0, 0])
        # ------------------速度調整Q------------------ #
        # Q[1, 1] = alpha * Q[1, 1] + (1 - alpha) * ((9.73741418e-01/5.87421163e-04) * dk * dk * (9.73741418e-01/5.87421163e-04)) # Km @ Km.T
        dk1 = Km[1] * u_p - xp[1] 
        Q[1, 1] = alpha * Q[1, 1] + (1 - alpha) * (Km.T @ Km * dk1 * dk1)
        # Q[1, 1] = alpha * Q[1, 1] + (1 - alpha) * (Km[1] * Km[1] * dk1 * dk1)
        Q_vel.append(Q[1, 1])
        # ------------------加速度調整Q------------------ #
        # Q[2, 2] = alpha * Q[2, 2] + (1 - alpha) * ((9.72314016e+02/5.87421163e-04) * dk * dk * ((9.72314016e+02/5.87421163e-04))) # Km @ Km.T
        dk2 = Km[2] * u_p - xp[2] 
        Q[2, 2] = alpha * Q[2, 2] + (1 - alpha) * (Km.T @ Km * dk2 * dk2)
        # Q[2, 2] = alpha * Q[2, 2] + (1 - alpha) * (Km[2] * Km[2] * dk2 * dk2)
        Q_acc.append(Q[2, 2])
        # 儲存Q值
        Q_save.append(Q)
        # print("AEKF_Q =", Q)

        pose[i] = xm[0]
        vele[i] = xm[1]
        acce[i] = xm[2]
    
    return pose, vele, acce , Q_pos, Q_acc, Q_vel, u_p_values, Q_save