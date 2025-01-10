import numpy as np
import matplotlib.pyplot as plt

## 速度\加速度Q值分開調整
def AKF_2(dt, Pos, true_pos, true_vel, true_acc):
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
    Q = np.array([[0.01, 0.01, 0.01],
                  [0.01, 0.01, 0.01],
                  [0.01, 0.01, 0.01]]) 
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
    # u_a, u_v用km去調整
    # Q = np.array ([[2.88889287e+00,1.44723478e-01,1.44723478e+02],
    #                  [1.44723478e-01,1.72881162e+00,4.06162767e+05],
    #                  [1.44723478e+02,4.06162767e+05,2.36342967e+10]]) 
    
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
    # P = np.array([[ 1.25376079e+01,  2.52943116e-01, -4.96150568e+02],
    #             [ 2.52943116e-01,  6.78389579e+02,  2.67310130e+05],
    #             [-4.96150568e+02,  2.67310130e+05,  5.35777931e+08]])
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
    u_v_values = []
    u_a_values = []
    y_values = []
    delta_x_values = []
    Q_acc = []
    Q_vel = []
    Q_pos = []
    Q_0err = []
    Q_save = []
    v_data = [0]
    a_data = [0, 0]
    v_cov_data = []
    a_cov_data = []

    
    for i in range(len(Pos)-2): # m = measurement;p = predict len(pos)
        # print("Q_in =", Q)
        Pp = np.dot(np.dot(A, Pm), A.T) + Q
        xp = np.dot(A, xm) + Wt
        # print("xm =", xm)
        # print("xp =", xp)
        Km = np.dot(Pp, C.T) / (np.dot(np.dot(C, Pp), C.T) + R)
        dk = (Pos[i] - np.dot(C, xp)) # dk = zk - hk*xk
        # print("Km =", Km)
        y = (Pos[i] - np.dot(C, xp))
        # print("y.shape =", np.array(y).shape)
        xm = xp + np.dot(Km, y) # =x_hat
        ek = (Pos[i] - np.dot(C, xm)) # ek = zk - hk*xk+1
        Pm = np.dot((np.eye(3) - np.dot(Km, C)), Pp)
        # print('Pm =', Pm)
        # print('Pm[0][0] =', Pm[0][0])
    #--------------------------------------------R值自適應--------------------------------------------#
        alpha = 0.2
        # R = alpha * R + (1 - alpha) * (ek * ek + np.dot(np.dot(C, Pp), C.T))
        # print("R =", R)
    #--------------------------------------------Q值自適應--------------------------------------------#
        ##-------位置求Q-------##
        # print("-------位置求Q-------")
        ## 求M
        u_p = (Pos[i]- xm[0]) # 實際值-卡爾曼估測值
        err = Pos[i]- xm[0]
        err_p = true_pos[i] - xm[0] # 命令值-卡爾曼估測值
        ratio_p = u_p / err_p # u_p = ratio_p * err_p
        u_p_values.append(u_p)
        n = i+1
        u_sqr = [val**2 for val in u_p_values[:n]]
        M = sum(u_sqr[:n])/n
        ## 求G_telda
        Y = np.dot(np.dot(C, Pp), C.T) + R
        G_tel = Km[0]**2 * Y
        ## 求G_hat
        delta_x = xm[0] - xp[0] # x - x-
        delta_x_values.append(delta_x)
        m = i+1
        delta_x_values_sqr = [val**2 for val in delta_x_values[:m]]
        G_hat = sum(delta_x_values_sqr[:m]) / m
        ## 求S
        G = G_hat/G_tel
        S = np.maximum(1, G_hat/G_tel)
        ## 求Q_hat
        Q_hat = S * M
        # 加入alpha
        # Q_hat = (1 - alpha) * S * M + alpha * Q
        # a = Q_hat
        Q_hat_matrix = Q
        Q_hat_matrix[0, 0] = Q_hat[0, 0]
        ## 更新Q值
        Q_new = Q_hat_matrix
        Q = Q_new
        Q_pos.append(Q_new[0, 0])
        # MIN_Q_VALUE[1, 1] = np.mean(Q_vel)
        # print("--------------------------------------------------")

        ##-------速度求Q-------##
        # print("-------速度求Q-------")
        ## 求M
        u_p = (Pos[i] - xm[0]) # 實際值-卡爾曼估測值
        # err_v = true_vel[i] - xm[1] # 命令值-卡爾曼估測值
        # u_v = ratio_p * err_v # u_p = ratio_p * err_p
        # u_v = 1 * err_v # u_p = ratio_p * err_p
        # u_v = (3.40598632e-01/2.18588407e-04) * u_p # sin data
        # u_v = (1.03810478e+00/6.19670228e-04) * u_p # without noise
        # u_v = (9.73741418e-01/5.87421163e-04) * u_p # with noise
        # u_v = (10**2) * u_p
        # u_v = (1.27516429e+00/1.25124714e+01) * u_p # Q by variance Q = A@sigma@A.T
        # u_v = (3.39695618e-01/1.25111239e+01) * u_p # Q by variance sigma
        # y = (Pos[i] - np.dot(C, xp))
        # u_v = (Km[1]) * u_p + np.random.normal(0, np.sqrt(Pm[1][1]))
        # print("Pm[1][1] =", Pm[1][1])
        # print("u_v =", u_v)
        # print("np.sqrt(Pm[1][1]) =", np.sqrt(Pm[1][1]))
        # u_v = (Pm[0][1] / Pm[0][0]) * u_p 
        u_v = Km[1] * u_p - xm[1]
        # u_v = u_p
        # u_v = (Pos[i] * Pm[0][1] / Pm[0][0] - xm[1])
        # v_cov = Pos[i] * Pm[0][1] / Pm[0][0]
        v_cov = Km[1] * u_p
        v_cov_data.append(v_cov)
        # u_v = (Pos[i] * Km[1]  - xm[1])
        # u_v = (Pos[i] * 2.52943116e-01 / 1.25376079e+01  - xm[1])
        # u_v = (2.52943116e-01 / 1.25376079e+01) * u_p
        # u_v = (Pos[i] * 9.73741418e-01/5.87421163e-04  - xm[1])
        # print("u_v = ", u_v)

        # if i == 0:
        #     u_v = (9.73741418e-01/5.87421163e-04) * u_p # with noise
            # print("Km[1] =", Km[1])
            # u_v = (Km[1]) * u_p
            # u_v = (Pm[0][1] / Pm[0][0]) * u_p
        # else:
        #     u_v = (9.73741418e-01/5.87421163e-04) * u_p # with noise
            # print("Km[1] =", Km[1])
            # u_v = (Km[1]) * u_p
            # u_v = (Pm[0][1] / Pm[0][0]) * u_p

        # u_v = (dt**4/8 / dt**5/20)* u_p
        # u_v = true_vel[i] - xm[1]
        # u_v = (2.04532016e-06 / 1.19728937e-07) * u_p # CMD
        # u_v = (-3.83997884e-04 / 4.03005280e-06) * u_p # CFD
        # u_v = (3.07764544e-03 /1.14494042e-05) * u_p # LAE
        # u_v = (-6.71932504e-04 /1.07560069e-04) * u_p # LAE IPS300G30
        # u_v = (2.56369441e-12/8.54565231e-16) * u_p # sim data
        # u_v = 2*1e3 * u_p # sim data
        u_v_values.append(u_v)
        n = i+1
        u_sqr = [val**2 for val in u_v_values[:n]]
        M = sum(u_sqr[:n])/n
        ## 求G_tel
        G_tel = Km[1]**2 * Y
        ## 求G_hat
        delta_x = xm[1] - xp[1] # x - x-
        delta_x_values.append(delta_x)
        m = i+1
        delta_x_values_sqr = [val**2 for val in delta_x_values[:m]]
        G_hat = sum(delta_x_values_sqr[:m]) / m
        ## 求S
        G = G_hat/G_tel
        S = np.maximum(1, G_hat/G_tel)
        ## 求Q_hat
        Q_hat = S * M 
        # Q_hat = (1 - alpha) * S * M + alpha * Q

        # #---------------AEKF--------------#
        # alpha = 0.3
        # dk1 = Km[1] * u_p - xp[1] 
        # Q[1, 1] = alpha * Q[1, 1] + (1 - alpha) * (Km.T @ Km * dk1 * dk1)
        # Q_hat[0, 0] = Q[1, 1]
        # #---------------------------------#

        Q_hat_matrix = Q
        Q_hat_matrix[1, 1] = Q_hat[0, 0]
        ## 更新Q值
        Q_new = Q_hat_matrix
        Q = Q_new
        Q_vel.append(Q_new[1, 1])
        # MIN_Q_VALUE[1, 1] = np.mean(Q_vel)
        # print("--------------------------------------------------")

        ##-------加速度求Q-------##
        # print("-------加速度求Q-------")
        ## 求M
        u_p = (Pos[i] - xm[0]) # y(k)-x_hat(k)  實際值-卡爾曼估測值
        # err_a = true_acc[i] - xm[2] # 命令值-卡爾曼估測值
        # u_a = 5*ratio_p * err_a # u_p = ratio_p * err_p
        # u_a = 5 * err_a # u_p = ratio_p * err_p
        # u_a = (6.78628723e+02/2.18588407e-04) * u_p # sin data 
        # u_a = (9.91154429e+01/2.49907392e-04) * u_p
        # u_a = (1.03739229e+03/6.19670228e-04) * u_p # without noise
        # u_a = (9.72314016e+02/5.87421163e-04) * u_p # with noise
        # u_a = (10**6) * u_p
        # u_a = (1.23241483e+02/1.25124714e+01) * u_p # Q by variance Q = A@sigma@A.T
        # u_a = (-4.11649847e+02/1.25111239e+01) * u_p # Q by variance sigma
        # y = (Pos[i] - np.dot(C, xp))
        # u_a = (Km[2]) * u_p + np.random.normal(0, np.sqrt(Pm[2][2]))
        # u_a = (Pm[0][2] / Pm[0][0]) * u_p 
        u_a = Km[2] * u_p - xm[2] 
        # u_a = u_p
        # u_a = (Pos[i] * Pm[0][2] / Pm[0][0] - xm[2])
        # a_cov = Pos[i] * Pm[0][2] / Pm[0][0]
        a_cov = Km[2] * u_p
        a_cov_data.append(a_cov)
        # u_a = (Pos[i] * Km[2]  - xm[2])
        # u_a = (Pos[i] * 4.96150568e+02 / 1.25376079e+01  - xm[2])
        # u_a = (Pos[i] * 9.72314016e+02/5.87421163e-04  - xm[2])
        # u_a = (4.96150568e+02 / 1.25376079e+01) * u_p
        # print("u_a = ", u_a)

        # if i == 0:
        #     u_a = (9.72314016e+02/5.87421163e-04) * u_p # with noise
            # print("Km[2] =", Km[2])
            # u_a = (Km[2]) * u_p # u_v = (Km[1]) * u_p # u_a = (Km[2]) * (Km[1]) * (Km[0]) * u_p
            # u_a = (Pm[0][2] / Pm[0][0]) * u_p
        # else:
        #     u_a = (9.72314016e+02/5.87421163e-04) * u_p # with noise
            # print("Km[2] =", Km[2])
            # u_a = (Km[2]) * u_p # u_v = (Km[1]) * u_p # u_a = (Km[2]) * (Km[1]) * (Km[0]) * u_p
            # u_a = (Pm[0][2] / Pm[0][0]) * u_p

        # u_a = (dt**3/6 / dt**5/20) * u_p
        # u_a = true_acc[i] - xm[2]
        # u_a = (2.04531994e-03 / 1.19728937e-07) * u_p # cmd
        # u_a = (-5.90703882e+00 / 4.03005280e-06) * u_p # CFD
        # u_a = (1.62770948e-02 /1.14494042e-05) * u_p # LAE
        # u_a = (-1.62890570e+02 / 1.07560069e-04) * u_p # LAE IPS300G30
        # u_a = (5.12738034e-09/8.54565231e-16) * u_p # sim data 
        # u_a = 5*1e5 * u_p # sim data
        u_a_values.append(u_a)
        n = i+1
        u_sqr = [val**2 for val in u_a_values[:n]]
        M = sum(u_sqr[:n])/n
        ## 求G_telda
        Y = np.dot(np.dot(C, Pp), C.T) + R
        G_tel = Km[2]**2 * Y
        ## 求G_hat
        delta_x = xm[2] - xp[2] # x - x-
        delta_x_values.append(delta_x)
        m = i+1
        delta_x_values_sqr = [val**2 for val in delta_x_values[:m]]
        G_hat = sum(delta_x_values_sqr[:m]) / m
        ## 求S
        G = G_hat/G_tel
        S = np.maximum(1, G_hat/G_tel)
        ## 求Q_hat
        Q_hat = S * M 
        # Q_hat = (1 - alpha) * S * M + alpha * Q

        Q_hat_matrix = Q
        Q_hat_matrix[2, 2] = Q_hat[0, 0]
        ## 更新Q值
        Q_new = Q_hat_matrix
        Q = Q_new
        Q_acc.append(Q_new[2, 2])
        Q_save.append(Q)
        # MIN_Q_VALUE[2, 2] = np.mean(Q_acc)
        # print("--------------------------------------------------")
        cov_xx = Q_new[0, 0] # cov(x,x)
        cov_xv = Q_new[0, 0] / dt # cov(x,v)
        cov_xa = Q_new[0, 0] / dt**2 # cov(x,v)
        cov_vv = Q_new[1, 1] # cov(v,v)
        cov_va = Q_new[1, 1] / dt # cov(v,a)
        cov_aa = Q_new[2, 2] # cov(a,a)

        # u_p = (Km[0]) * u_p
        # u_v = (Km[1]) * u_p
        # u_a = (Km[2]) * u_p

        # print("cov_xx =", cov_xx)
        # print("cov_xv =", cov_xv)
        # print("cov_xa =", cov_xa)

        # u_v = (cov_xv/cov_xx) * u_p
        # u_a = (cov_xa/cov_xx) * u_p

        # u_v = (cov_xv/cov_xx) * u_p
        # u_a = (cov_xa/cov_xx) * u_p
        # Q = np.array([[cov_xx, cov_vv/cov_xx, cov_aa/cov_xx],
        #               [cov_vv/cov_xx, cov_vv, cov_aa/cov_vv],
        #               [cov_aa/cov_xx, cov_aa/cov_vv, cov_aa]])
        # print("Q =", Q)
        # print("u_v =", (cov_xv/cov_xx))
        # print("u_a =", (cov_xa/cov_xx))
        # print("--------------------------------------------------")

        pose[i] = xm[0]
        vele[i] = xm[1]
        acce[i] = xm[2]
    # print("x =", x[:15])
    # print("v_data =", v_data[:15])
    # print("a_data =", a_data[:15])

    return pose, vele, acce, Q_pos, Q_acc, Q_vel, u_p_values, u_v_values, u_a_values, Q_save, v_cov_data, a_cov_data