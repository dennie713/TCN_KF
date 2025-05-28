import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import pylab
import os
import TCN, dataset_arrange, LogCoshLoss, OriKF
import CFD, calculateError_RTS

if __name__ == "__main__":

    scara = 1
    start_size = 0
    data_size = 12000 
    #-----------------------------------Ground Truth-------------------------------------------#
    # 目前的Ground Truth是把數據先經過"zero-phase-filter濾波"後再進行"中央差分法"計算的
    # 讀取經過zero_phase_filter的資料
    if scara == 1:
        path = 'main/real_dataset/filtered_scara1_120000.txt'
    elif scara == 2:
        path = 'main/real_dataset/filtered_scara2_120000.txt'
    # path = 'real_dataset/filtered_scara2_120000.txt'
    data = np.genfromtxt(path, delimiter='\t')
    data = np.array(data)
    filtered_pos_exp = np.array(data[start_size:data_size]) 
    filtered_CFD_pose, filtered_CFD_vele, filtered_CFD_acce = CFD.CFD(filtered_pos_exp)

    if scara == 1:
        path1 = 'main/dataset/Real_AKF_OLS_scara1_n=10_12000/x_data_all_AKF.txt'
        path2 = 'main/dataset/Real_AKF_OLS_scara1_n=10_12000/P_data_all_AKF.txt'
        path3 = 'main/dataset/Real_AKF_OLS_scara1_n=10_12000/raw_data_all_AKF.txt'
        path4 = 'main/dataset/Real_AKF_OLS_scara1_n=10_12000/Q_data_all_AKF.txt'
        path5 = 'main/dataset/Real_AKF_OLS_scara1_n=10_12000/x_RTS_AKF.txt'
        path6 = 'main/dataset/Real_AKF_OLS_scara1_n=10_12000/K_RTS_AKF.txt'
    elif scara == 2:
        path1 = 'main/dataset/Real_AKF_OLS_scara2_n=10_12000/x_data_all_AKF.txt'
        path2 = 'main/dataset/Real_AKF_OLS_scara2_n=10_12000/P_data_all_AKF.txt'
        path3 = 'main/dataset/Real_AKF_OLS_scara2_n=10_12000/raw_data_all_AKF.txt'
        path4 = 'main/dataset/Real_AKF_OLS_scara2_n=10_12000/Q_data_all_AKF.txt'
        path5 = 'main/dataset/Real_AKF_OLS_scara2_n=10_12000/x_RTS_AKF.txt'
        path6 = 'main/dataset/Real_AKF_OLS_scara2_n=10_12000/K_RTS_AKF.txt'
    # Q_data = np.loadtxt(path4, delimiter=' ')
    x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_input_data_all, P_data, P_k_update_data, KCP_data, P_input_data_all, raw_data_all, x_k_predict_data, Q_data_all, x_RTS_data, K_RTS_data  = dataset_arrange.loadSimData(path1, path2, path3, path4, path5, path6)

    # 讀取TCN_output結果
    path = 'main/result/x_tcn_output.txt'
    K = np.loadtxt(path, delimiter=' ')

    dt = 0.001
    A = np.array([[1, dt, 0.5*dt**2],
                  [0, 1, dt],
                  [0, 0, 1 ]])
    state = []
    # for i in range(1, len(K), 1):
    # # for i in range(2):
    #     # print("x_k_update_data[i].reshape(3, 1) =", x_k_update_data[i].reshape(3, 1))
    #     # print("K[i].reshape(3, 1) =", K[i].reshape(3, 1))
    #     # 用TCN_output解RTS狀態
    #     # result = x_k_update_data[::-1][i].reshape(3, 1) + K[i][::-1].reshape(3, 1).T @ (x_k_update_data[::-1][i+1].reshape(3, 1) - A @ x_k_update_data[::-1][i].reshape(3, 1))
    #     result = x_k_update_data[i-1].reshape(3, 1) + K[i].reshape(3, 1).T @ (x_k_update_data[i].reshape(3, 1) - A @ x_k_update_data[i-1].reshape(3, 1))
    #     # result = x[k].reshape(3, 1) + K @ (x[k+1].reshape(3, 1) - (A @ x[k].reshape(3, 1)))  
    #     state.append(result)
    # state = np.array(state).squeeze()

    for k in range(len(x_k_update_data)-2,-1,-1):
    # for k in range(10-2,-1,-1):

        result = x_k_update_data[k].reshape(3, 1) + K[::-1][k+1].reshape(3, 1).T @ (x_k_update_data[k+1].reshape(3, 1) - (A @ x_k_update_data[k].reshape(3, 1)))  

        # 把最後一筆的資料放進去，因為不用平滑
        if k == 0:
            state.append(x_k_update_data[-1].flatten())
        state.append(result)
    state_array = np.array([np.array(s).flatten() for s in state])
    # print("state =\n", state)
    # state = np.array(state)
    # print("state =", state)


    
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(x_k_update_data[:, 2], label='x_k_update', color='orange', linewidth=1)
    plt.plot(state_array[::-1][:, 2], label='TCN_output', color='red', linewidth=1) # 
    plt.plot(x_RTS_data[:, 2], label='RTS', color='blue', linestyle='--', linewidth=1) 
    plt.plot(filtered_CFD_acce, label='Truth', color='black', linewidth=1)
    plt.xlabel('data')
    plt.ylabel('Pos')
    plt.legend(loc='upper right', ncol=2)
    plt.title('velocity')

    # 與RTS比較誤差
    vele = state_array[::-1][:, 1]
    acce = state_array[::-1][:, 2]
    x_RTS = x_RTS_data
    result_RTS = calculateError_RTS.calError(filtered_CFD_vele, filtered_CFD_acce, vele, acce, x_RTS[:, 1], x_RTS[:, 2])
    result2_RTS = calculateError_RTS.calError2(filtered_CFD_vele[600:], filtered_CFD_acce[600:], vele[600:], acce[600:], x_RTS[:, 1][600:], x_RTS[:, 2][600:])

    plt.show()






