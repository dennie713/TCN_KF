import numpy as np
import matplotlib.pyplot as plt

## 速度\加速度Q值分開調整
def MY_AKF(dt, Pos, true_pos, true_vel, true_acc):
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
    Q = np.array([[1.25124714e+01, 1.27516429e+00, 1.23241483e+02],
                    [1.27516429e+00, 1.74838225e+03, 8.02527454e+05],
                    [1.23241483e+02, 8.02527454e+05, 5.35272247e+08]])
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
    count = 0
    
    for i in range(len(Pos)-2): # m = measurement;p = predict len(pos)
        # print("Q_in =", Q)
        Pp = np.dot(np.dot(A, Pm), A.T) + Q
        xp = np.dot(A, xm) + Wt
        # print("xm =", xm)
        # print("xp =", xp)
        Km = np.dot(Pp, C.T) / (np.dot(np.dot(C, Pp), C.T) + R)
        dk = (Pos[i] - np.dot(C, xp)) # dk = zk - hk*xk
        # print("dk =", dk)
        # print("Km =", Km)
        y = (Pos[i] - np.dot(C, xp))
        # print("y.shape =", np.array(y).shape)
        xm = xp + np.dot(Km, y) # =x_hat
        ek = (Pos[i] - np.dot(C, xm)) # ek = zk - hk*xk+1
        Pm = np.dot((np.eye(3) - np.dot(Km, C)), Pp)

    #--------------------------------------------自適應--------------------------------------------#
        u_p = (Pos[i]- xm[0]) # 實際值-卡爾曼估測值
        u_v = Km[1] * u_p - xm[1]
        u_a = Km[2] * u_p - xm[2] 

        # u_v = (Pm[0][1] / Pm[0][0]) * u_p
        # u_a = (Pm[0][2] / Pm[0][0]) * u_p

        # u_v = (Pm[0][1] / Pm[0][0]) * Pos[i] - xm[1]
        # u_a = (Pm[0][2] / Pm[0][0]) * Pos[i] - xm[2]

        # u_v = u_p
        # u_a = u_p

        n = i+1
        u_p_sqr = [val**2 for val in u_p_values[:n]]
        M_u_p = sum(u_p_sqr[:n])/n # MSE

        u_v_sqr = [val**2 for val in u_v_values[:n]]
        M_u_v = sum(u_v_sqr[:n])/n # MSE

        u_a_sqr = [val**2 for val in u_a_values[:n]]
        M_u_a = sum(u_a_sqr[:n])/n # MSE
        # print("MSE =", M)

        delta_x = xm[0] - xp[0] # x - x-
        delta_x_values.append(delta_x)
        m = i+1
        delta_x_values_sqr = [val**2 for val in delta_x_values[:m]]
        G_hat = sum(delta_x_values_sqr[:m]) / m

        # print("G_hat / M =", G_hat / M)

        u_p_values.append(u_p)
        u_v_values.append(u_v)
        u_a_values.append(u_a)
        err = np.array([u_p_values, u_v_values, u_a_values])
        err = err.reshape(-1, 3)
        print("err =", err)
        cov_P = np.dot(err.T, err)
        cov_P = cov_P.squeeze()
        print("cov_P =", cov_P)

        ek = np.array([u_p, u_v, u_a]) - xm
        ek_2 = np.dot(ek.T, ek)
        
        #--------------------------------------------R值自適應--------------------------------------------#
        # alpha = 0.7
        # R = alpha * R + (1 - alpha) * (ek_2 * ek_2 + np.dot(np.dot(C, Pp), C.T))
        # print("R =", R)

        #--------------------------------------------P值自適應--------------------------------------------#
        gamma = np.trace(P) / (np.trace(P) + np.trace(cov_P))
        # gamma = np.linalg.norm(P) / (np.linalg.norm(P) + np.linalg.norm(cov_P))
        # gamma = 0.4
        P_new = (1-gamma) * P + (gamma) * cov_P
        # P_new = (gamma) * P + (1-gamma) * cov_P
        # P_new = P + cov_P
        Pm = P_new
        print("Pm =", Pm)

        # P_norm = np.sqrt(np.sum(cov_P**2))
        # Pm_norm = np.sqrt(np.sum(Pm**2))
        # if min(P_norm, Pm_norm) == P_norm:
        #     P_new = cov_P
        # else:
        #     P_new = Pm
        # Pp = P_new

        # P_new 與 Pm 比較
        # P_new = Pm
        # for j in range(3):
        #     if cov_P[j][j] < Pm[j][j]:
        #         P_new[j][j] = cov_P[j][j]
        # Pp = P_new

        #--------------------------------------------Q值自適應--------------------------------------------#
        del_Q = (np.dot(Km, y)) @ (np.dot(Km, y).T)
        print("del_Q =", del_Q)
        # print("np.dot(Km, y) =", np.dot(Km, y))
        # beta = np.linalg.norm(Q, 'fro') / (np.linalg.norm(Q, 'fro') + np.linalg.norm(del_Q, 'fro'))
        # beta = np.linalg.norm(del_Q, 'fro') / (np.linalg.norm(Q, 'fro'))
        
        q = xp - np.dot(A, xm)
        q_2 = np.dot(q.T, q)
        beta = np.trace(Q) / (np.trace(Q) + np.trace(q_2))
        print("q_2 =", q_2)
        # Q = beta * q_2 + (1-beta) * Q 

        # beta = 0.8
        beta = np.trace(Q) / (np.trace(Q) + np.trace(del_Q))
        # print("beta =", beta)
        # if beta < 0.4:
        #     beta = 0.4
        #     count = count + 1
        #     print("count =", count)
            # print("beta < 0.5")
        Q_new = (beta) * Q + (1-beta) * del_Q 
        # adj = np.array([[M_u_p, 0, 0],
        #                 [0, M_u_v, 0],
        #                 [0, 0, M_u_a]])
        # print("adj =", adj)
        # Q[0][0] = Q[0][0]*M_u_p
        # Q[1][1] = Q[1][1]*M_u_v
        # Q[2][2] = Q[2][2]*M_u_a
        # Q_new = Q + del_Q
        # Q_new = np.dot(np.dot(A, cov_P), A.T) + del_Q

        # q = np.dot(np.dot(A, cov_P), A.T)
        # gamma = np.linalg.norm(Q, 'fro') / (np.linalg.norm(Q, 'fro') + np.linalg.norm(q, 'fro'))
        # Q_new = gamma * Q + (1 - gamma) * np.dot(np.dot(A, cov_P), A.T) + del_Q

        # if np.abs(dk) > 0.04:
        #     Q_new = np.dot(np.dot(A, Pm), A.T) + del_Q 
        #     count = count + 1
        # else:
        #     # Q_new = Q + del_Q
        #     Q_new = beta * Q + (1 - beta) * del_Q 
        
        # print("count =", count)
        # Q_new = Q + del_Q
        # Q_new = del_Q
        # Q_new = np.exp(-beta) * del_Q + (1 - np.exp(-beta)) * Q
        Q = Q_new 
        print("Q = ", Q)
        Q_save.append(Q)
        Q_pos.append(Q[0, 0])
        Q_vel.append(Q[1, 1])
        Q_acc.append(Q[2, 2])
        #--------------------------------------------------#

        pose[i] = xm[0]
        vele[i] = xm[1]
        acce[i] = xm[2]
        # if i > 0:
        #     P = Pm
        #     Q = Q_save[0]
        

    return pose, vele, acce, Q_pos, Q_acc, Q_vel, u_p_values, u_v_values, u_a_values, Q_save