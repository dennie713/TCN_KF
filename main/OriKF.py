import numpy as np

def KalmanFilter(dt, pos):
    print("##################### Generating Data . . . #####################")
    A = np.array([[1]])
    B = np.array([[0],
                  [dt]])
    C = np.array([[1]])
    Q = np.array([[1e-4]]) #5501.379519887754為Pos的變異數
    R = 0.001 # 與誤差有關
    P = np.array([[1e-10]])
    pose = np.zeros(len(pos))
    xm = np.array([[pose[0]]])
    Pm = P

    x_k_update_data = []
    k_y_data = []
    x_tel = []
    x_true_data = [] 
    x_true_data_noise = []
    x_k_predict_data = []

    P_k_update_data = []
    KCP_data = []

    for i in range(len(pos)):
        Pp = np.dot(np.dot(A, Pm), A.T) + Q
        xp = np.dot(A, xm)
        Km = np.dot(Pp, C.T) / (np.dot(np.dot(C, Pp), C.T) + R)
        y = (pos[i] - np.dot(C, xp))  # 創新
        k_y = Km @ y
        xm = xp + np.dot(Km, (pos[i] - np.dot(C, xp)))
        Pm = np.dot((np.eye(1) - np.dot(Km, C)), Pp)
        KCP = np.dot((np.dot(Km, C)), Pp)
        pose[i] = xm[0]

        # x_data_all
        x_k_update_data.append(xm.flatten())
        k_y_data.append(k_y.flatten())
        x_true_data.append(xm[0].flatten())
        x_tel = np.array(x_true_data) - np.array(x_k_update_data)
        # x_tel.append(xm.flatten())
        x_true_data_noise = np.append(x_true_data_noise, pos[i])
        x_true_data_noise = np.expand_dims(x_true_data_noise, axis=1)
        # print("x_true_data_noise.shape =", x_true_data_noise.shape)
        x_k_predict_data.append(xp.flatten())
        # P_data_all
        P_k_update_data.append(Pm.flatten())
        KCP_data.append(KCP.flatten())
    x_true_noise = pos    
    x_data_all = np.concatenate((x_k_update_data, k_y_data, x_tel, x_true_data, x_true_data_noise, x_k_predict_data), axis=1)# me
    P_data_all = np.concatenate((P_k_update_data, KCP_data), axis=1)# me
    x_input_data_all = np.concatenate((x_true_noise, x_k_update_data, k_y_data, x_tel), axis=1) 
    # np.savetxt('sim_data/dataset/x_input_data_all_KF_15000_exp2.txt', x_input_data_all, delimiter=' ')

    return x_data_all, P_data_all, x_input_data_all