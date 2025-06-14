import numpy as np

# 讀取 txt 文件
def loadSimData(path_x, path_p, path_raw, path_Q, path_est_err):

    # raw data順序true_pos, true_vel, true_acc
    raw_data = np.loadtxt(path_raw, delimiter=' ')
    # true_pos = np.array(raw_data[:, 0])
    # true_vel = np.array(raw_data[:, 1])
    # true_acc = np.array(raw_data[:, 2])
    raw_data_all = raw_data 
    # true_pos = [np.array(item).reshape(-1) for item in true_pos]
    # true_vel = [np.array(item).reshape(-1) for item in true_vel]
    # true_acc = [np.array(item).reshape(-1) for item in true_acc]
    # print("true_pos =", true_pos.reshape(-1).shape)

    # x
    x_data = np.loadtxt(path_x, delimiter=' ') #path = 'x_input_data_all.txt'
    # print(x_data.shape)
    # 順序x_k_update_data, k_y_data, x_tel, x_true_data, x_true_data_noise
    x_k_update_data = x_data[:, 0:3]
    k_y_data = x_data[:, 3:6]
    # x_tel = x_data[:, 6:9]
    # prediction_errors_data = x_data[:, 6:8]
    x_true = x_data[:, 9:10] # x_true_data
    x_true_noise = x_data[:, 10:11] # x_true_data_noise
    x_k_predict_data = x_data[:, 11:14]
    """"原本的x_tel算錯了，因此重新計算"""
    x_tel = x_true_noise - x_k_update_data
    # x_obsve = x_data[:, 10]# z_data
    # x_k_predict_data = x_data[:, 11:13]


    Q_data = np.loadtxt(path_Q, delimiter=' ')
    # Q_data_all = Q_data
    Q_data_all = np.concatenate((Q_data[:, 0].reshape(-1, 1), Q_data[:, 4].reshape(-1, 1), Q_data[:, 8].reshape(-1, 1)), axis=1)
    

    #只取對角元素當特徵
    # x_input_data_all = np.concatenate((x_true_noise, Q_data[:, 0].reshape(-1, 1), Q_data[:, 4].reshape(-1, 1), Q_data[:, 8].reshape(-1, 1)), axis=1) # training 
    # x_input_data_all = np.concatenate((Q_data[:, 0].reshape(-1, 1), Q_data[:, 4].reshape(-1, 1), Q_data[:, 8].reshape(-1, 1)), axis=1) # training 
    # print("x_input_data_all =", x_input_data_all)
    # 推論對角元素即可
    # Q = np.array([0.01, 0.01, 0.01]).reshape(-1, 3)
    # Q = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]).reshape(-1, 9)
    # Q_expanded = np.tile(Q, (9998, 1))
    # print(Q_expanded.shape)  # (10000, 9)
    # x_input_data_all = np.concatenate((x_true_noise, Q_expanded), axis=1) # loading
    # x_input_data_all = Q_expanded # loading

    # x_input_data_all = raw_data
    x_input_data_all = x_true_noise # dataset 8
    # x_input_data_all = np.concatenate((x_true_noise,x_true_noise), axis=1)
    # x_input_data_all = np.concatenate((x_true_noise, x_k_update_data[:, 0:1].reshape(-1, 1)), axis=1) # dataset
    # x_input_data_all = np.concatenate((x_true_noise, x_true_noise, x_true_noise), axis=1)
    # x_input_data_all = np.concatenate((x_true, true_pos, true_vel, true_acc), axis=1)
    # print("x_input_data_all =", np.array(x_input_data_all).shape)
    # x_input_data_all = np.concatenate((x_true_noise, k_y_data), axis=1)
    # x_input_data_all = np.concatenate((x_true_noise, true_pos[:14998]), axis=1)
    x_input_data_all = np.concatenate((x_true_noise, x_true_noise, x_true_noise), axis=1)
    # x_input_data_all = np.concatenate((x_true_noise, k_y_data, x_tel), axis=1) # paper dataset
    # x_input_data_all = np.concatenate((k_y_data[:, 1].reshape(-1, 1), x_tel[:, 1].reshape(-1, 1)), axis=1) # paper dataset
    x_input_data_all = np.concatenate((x_true_noise, k_y_data[:, 1].reshape(-1, 1), x_tel[:, 1].reshape(-1, 1)), axis=1) # paper dataset
    x_input_data_all = np.concatenate((x_true_noise, x_k_update_data[:, 1].reshape(-1, 1), k_y_data[:, 1].reshape(-1, 1), x_tel[:, 1].reshape(-1, 1)), axis=1)
    

    # x_input_data_all = np.concatenate((x_true_noise, x_k_update_data), axis=1) # dataset 6
    # x_input_data_all = np.concatenate((x_true_noise, x_k_update_data, k_y_data), axis=1) # dataset 7
    # x_input_data_all = np.concatenate((x_true_noise, x_k_update_data, k_y_data, x_tel), axis=1) # dataset 9
    # x_input_data_all = np.concatenate((x_true_noise, x_true_noise, x_true_noise, x_true_noise), axis=1) # 

    # x_input_data_all = np.concatenate((true_pos, true_vel, true_acc), axis=1) # dataset 2
    # x_input_data_all = np.concatenate((x_true_noise, k_y_data, x_tel, true_pos, true_vel, true_acc), axis=1) # dataset 1
    # x_input_data_all = np.concatenate((x_true_noise, true_pos, true_vel, true_acc), axis=1) # dataset 2
    # x_input_data_all = np.concatenate((x_true_noise, x_true_noise, x_true_noise, raw_data[:, 0:3]), axis=1)
    # print(x_input_data_all)

    # p
    P_data = np.loadtxt(path_p, delimiter=' ') # 'P_data_10000.txt'
    # data排列順序P_k_update_data, KCP_data
    P_k_update_data = P_data[:, 0:9]
    KCP_data = P_data[:, 9:18]
    P_input_data_all = np.concatenate((P_k_update_data, KCP_data), axis=1)
    # print("P_input_data_all =", P_input_data_all)

    # # RTS
    # x_RTS_data = np.loadtxt(path_x_RTS, delimiter=' ')
    # x_RTS_data = x_RTS_data

    # K_RTS_data = np.loadtxt(path_K_RTS, delimiter=' ')
    # K_RTS_data = K_RTS_data

    # G_tel
    # data = np.loadtxt(path_G_tel, delimiter=' ')
    # G_tel_data = data

    # estimate error
    est_err_data = np.loadtxt(path_est_err, delimiter=' ')
    est_err_data = est_err_data


    # x_input_data_all
    # x_input_data_all = np.concatenate((x_k_update_data), axis=1)
    # x_input_data_all = x_k_update_data
    # x_input_data_all = np.concatenate((x_k_update_data[:, 1].reshape(-1, 1), P_k_update_data, Q_data[:, 0].reshape(-1, 1), Q_data[:, 4].reshape(-1, 1), Q_data[:, 8].reshape(-1, 1)), axis=1)
    # x_input_data_all = np.concatenate((x_k_update_data, P_k_update_data, Q_data[:, 0].reshape(-1, 1), Q_data[:, 4].reshape(-1, 1), Q_data[:, 8].reshape(-1, 1)), axis=1)
    # x_input_data_all = np.concatenate((x_true_noise, x_k_update_data[:, 1].reshape(-1, 1), P_k_update_data), axis=1) 
    # x_input_data_all = np.concatenate((x_k_update_data[:, 1].reshape(-1, 1), x_tel[:, 1].reshape(-1, 1)), axis=1) 
    x_input_data_all = np.concatenate((x_true_noise, k_y_data[:, 1].reshape(-1, 1), x_tel[:, 1].reshape(-1, 1)), axis=1)
    # x_input_data_all = np.concatenate((x_true_noise, x_tel[:, 1].reshape(-1, 1)), axis=1)
    # x_input_data_all = np.concatenate((x_k_update_data[:, 1].reshape(-1, 1), x_tel[:, 1].reshape(-1, 1)), axis=1)
    # x_input_data_all = np.concatenate((x_k_update_data[:, 2].reshape(-1, 1), k_y_data[:, 2].reshape(-1, 1), x_tel[:, 2].reshape(-1, 1)), axis=1) # paper dataset
    # x_input_data_all = np.concatenate((x_k_update_data[:, 1].reshape(-1, 1), k_y_data[:, 1].reshape(-1, 1)), axis=1) # paper dataset 
    # x_input_data_all = np.concatenate((x_k_update_data[:, 0:2].reshape(-1, 2), k_y_data[:, 0:2].reshape(-1, 2), x_tel[:, 0:2].reshape(-1, 2)), axis=1)
    # x_input_data_all = np.concatenate((x_k_update_data, k_y_data, x_tel), axis=1)
    # x_input_data_all = np.concatenate((k_y_data, est_err_data), axis=1)
    # x_input_data_all = np.concatenate((k_y_data, est_err_data, G_tel_data), axis=1)
    # x_input_data_all = np.concatenate((k_y_data[:, 0].reshape(-1, 1), est_err_data[:, 0].reshape(-1, 1), G_tel_data[:, 0].reshape(-1, 1)), axis=1)
    # x_input_data_all = np.concatenate((x_k_update_data, k_y_data, x_tel), axis=1)
    x_input_data_all = np.concatenate((x_true_noise, x_k_update_data), axis=1)
    x_input_data_all = np.concatenate((x_true_noise, k_y_data), axis=1)
    x_input_data_all = k_y_data
    # x_input_data_all = np.concatenate((x_k_update_data[:, 2].reshape(-1, 1), k_y_data[:, 2].reshape(-1, 1)), axis=1)
    # x_input_data_all = k_y_data[:, 0].reshape(-1, 1)
    # x_input_data_all = np.concatenate((k_y_data[:, 2].reshape(-1, 1), est_err_data[:, 2].reshape(-1, 1), G_tel_data[:, 2].reshape(-1, 1)), axis=1)
    # x_input_data_all = np.concatenate((k_y_data[:, 2].reshape(-1, 1), est_err_data[:, 2].reshape(-1, 1)), axis=1)
    # x_input_data_all = np.concatenate((k_y_data[:, 0].reshape(-1, 1), x_tel[:, 0].reshape(-1, 1)), axis=1)
    # x_input_data_all = np.concatenate((k_y_data, x_tel[:, 0].reshape(-1, 1)), axis=1)
    # x_input_data_all = np.concatenate((k_y_data, x_tel), axis=1)
    # x_input_data_all = x_k_update_data[:, 1].reshape(-1, 1)
    # x_input_data_all = x_k_update_data[:, 0].reshape(-1, 1)
    # x_input_data_all = np.concatenate((P_k_update_data), axis=1)
    # x_input_data_all = np.concatenate((x_true_noise[::-1][:], k_y_data[::-1, 1][:].reshape(-1, 1), x_tel[::-1, 1][:].reshape(-1, 1)), axis=1)

    # x_input_data_all = np.concatenate((x_k_update_data[::-1, 1][2:].reshape(-1, 1), k_y_data[::-1, 1][2:].reshape(-1, 1), x_tel[::-1, 1][2:].reshape(-1, 1)), axis=1)
    # x_input_data_all = np.concatenate((x_k_update_data[::-1, 1][:].reshape(-1, 1), k_y_data[::-1, 1][:].reshape(-1, 1), x_tel[::-1, 1][:].reshape(-1, 1)), axis=1)

    return x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_input_data_all, P_data, P_k_update_data, KCP_data, P_input_data_all, raw_data_all, x_k_predict_data, Q_data_all
