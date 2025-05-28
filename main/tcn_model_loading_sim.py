import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import TCN, dataset_arrange
import setTCNConfig
import UKF, EKF, OriKF, CFD, calculateError_RTS
from TCN import TemporalConvNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())  # 是否可用 GPU
# print(torch.cuda.current_device())  # 當前使用的 GPU

if __name__ == "__main__":
    # 參數設置
    start_size = 0
    validation_size = 12000

    # 輸入模擬資料
    # 選擇輸入資料
    scara = 2
    if scara == 1:
        path1 = 'main/dataset/Real_AKF_OLS_6axis3_n=10_n1n2=20_12000/x_data_all_AKF.txt'
        path2 = 'main/dataset/Real_AKF_OLS_6axis3_n=10_n1n2=20_12000/P_data_all_AKF.txt'
        path3 = 'main/dataset/Real_AKF_OLS_6axis3_n=10_n1n2=20_12000/raw_data_all_AKF.txt'
        path4 = 'main/dataset/Real_AKF_OLS_6axis3_n=10_n1n2=20_12000/Q_data_all_AKF.txt'
        # path5 = 'main/dataset/Real_AKF_OLS_6axis3_n=10_n1n2=20_12000/x_RTS_AKF.txt'
        # path6 = 'main/dataset/Real_AKF_OLS_6axis3_n=10_n1n2=20_12000/K_RTS_AKF.txt'
        path7 = 'main/dataset/Real_AKF_OLS_6axis3_n=10_n1n2=20_12000/est_err_data_all_AKF.txt'
        path8 = 'main/dataset/Real_AKF_OLS_6axis3_n=10_n1n2=20_12000/G_tel_data_all_AKF.txt'
    elif scara == 2:
        path1 = 'main/dataset/Real_AKF_OLS_6axis2_n=10_n1n2=20_12000/x_data_all_AKF.txt'
        path2 = 'main/dataset/Real_AKF_OLS_6axis2_n=10_n1n2=20_12000/P_data_all_AKF.txt'
        path3 = 'main/dataset/Real_AKF_OLS_6axis2_n=10_n1n2=20_12000/raw_data_all_AKF.txt'
        path4 = 'main/dataset/Real_AKF_OLS_6axis2_n=10_n1n2=20_12000/Q_data_all_AKF.txt'
        # path5 = 'main/dataset/Real_AKF_OLS_6axis2_n=10_n1n2=20_12000/x_RTS_AKF.txt'
        # path6 = 'main/dataset/Real_AKF_OLS_6axis2_n=10_n1n2=20_12000/K_RTS_AKF.txt'
        path7 = 'main/dataset/Real_AKF_OLS_6axis2_n=10_n1n2=20_12000/est_err_data_all_AKF.txt'
        path8 = 'main/dataset/Real_AKF_OLS_6axis2_n=10_n1n2=20_12000/G_tel_data_all_AKF.txt'
    # Q_data = np.loadtxt(path4, delimiter=' ')
    x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_input_data_all, P_data, P_k_update_data, KCP_data, P_input_data_all, raw_data_all, x_k_predict_data, Q_data_all = dataset_arrange.loadSimData(path1, path2, path3, path4, path7, path8)
    
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
    filtered_pos_exp = np.array(data[start_size:validation_size]) 
    filtered_CFD_pose, filtered_CFD_vele, filtered_CFD_acce = CFD.CFD(filtered_pos_exp)
    # data_set_size = start_size + validation_size

    # x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_input_data_all, P_data, P_k_update_data, KCP_data, P_input_data_all, raw_data_all, x_k_predict_data  = dataset_arrange.loadSimData(path1, path2, path3, path4)
    # x_k_update_data = np.concatenate((Q_data[:, 0].reshape(-1, 1), Q_data[:, 4].reshape(-1, 1), Q_data[:, 8].reshape(-1, 1)), axis=1)
    # print("x_input_data_all.shape =", x_input_data_all.shape)
    # print("x_input_data_all =", x_input_data_all)

    # 一維KF數據輸出
    # x_data_all, P_data_all, x_input_data_all = OriKF.KalmanFilter(0.001, x_true_noise)
    # x_input_data_all = np.loadtxt(path5, delimiter=' ')

    # 模型參數
    setConfig = setTCNConfig.TCNConfig()
    input_size, output_size, kernel_size,  stride, dropout, num_channels = setConfig.getTCNConfig()

    # 解標準化
    # 讀取
    data = np.load('tcn_normalization/normalizer.npz')
    x_mean = data['x_mean']
    x_std = data['x_std']
    y_mean = data['y_mean']
    y_std = data['y_std']
    print("y_mean =", y_mean)
    print("y_std =", y_std)
    # train_y_data = Q_data_all[:, 2]
    # y_mean = train_y_data.mean(axis=0, keepdims=True)
    # y_std = train_y_data.std(axis=0, keepdims=True) + 1e-8

    # 加載模型
    path = 'main/tcn_model/TCN_fea3_ker6_num[256]_epo243.pth'
    x_tcn_model_loaded = TCN.TemporalConvNet(num_inputs=input_size, num_channels=num_channels, num_classes=output_size, kernel_size=kernel_size,  stride=stride, dropout=dropout)
    # x_tcn_model_loaded.load_state_dict(torch.load(path, weights_only=True))  # 加載權重
    x_tcn_model_loaded.load_state_dict(torch.load(path))  # 加載權重
    x_tcn_model_loaded.eval()  # 將模型設置為評估模式  
    x_tcn_model = x_tcn_model_loaded.to(device) 

    # 輸入標準化
    standardization = 0
    if standardization == 1:
        x_input_data_all_mean = x_input_data_all.mean(axis=0, keepdims=True)
        x_input_data_all_std = x_input_data_all.std(axis=0, keepdims=True) + 1e-8
        x_input_data_all_norm = (x_input_data_all - x_input_data_all_mean) / x_input_data_all_std

    start_time = time.time()
    x_tcn_output_data = []

    # x_input_data = x_tcn_output
    # x_input_data = x_input_data_all[0]
    for k in range(start_size, start_size+validation_size):
        # print("k =", k)
        # x_tel = np.array(x_true) - np.array(x_k_update_data)
        # print("x_input_data_all[k] =", x_input_data_all[k])
        if standardization == 1:
            x_input_data = x_input_data_all_norm[k]
        else:
            x_input_data = x_input_data_all[k]
        # x_input_data = x_tcn_output_data[k]
        x_input_data = torch.tensor(np.hstack(x_input_data), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        # x_input_tensor = x_input_data.clone().detach().to(device)
        x_input_tensor = x_input_data.permute(0, 2, 1)

        # 使用模型進行推斷
        with torch.no_grad():  # 禁用梯度計算以提高推斷效率
            x_tcn_output = x_tcn_model_loaded(x_input_tensor)  # 獲取模型的輸出
            # 解標準化
            if standardization == 1:
                x_tcn_output = x_tcn_output.detach().cpu() * y_std + y_mean
            elif standardization == 0:
                x_tcn_output = x_tcn_output.detach().cpu()
        x_tcn_output_data.append(x_tcn_output.detach().cpu().numpy().flatten())
        # x_tcn_output_data.append(x_tcn_output.detach().cpu().numpy().flatten())
        # x_true_noise[k] = torch.tensor(np.hstack(x_true_noise[k]), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device).permute(0, 2, 1)
        
        # print("x_input_data.shape=",x_input_data[1:4].shape)  # 假设是 (batch_size, channels, sequence_length)
        # print("x_tcn_output.shape=",x_tcn_output.shape)  # 假设是 (batch_size, channels)
        # x_tcn_output = x_tcn_output.unsqueeze(2).permute(0, 2, 1)
        # x_input_data = torch.cat((x_input_data, x_tcn_output.detach()), dim=2)
        
    end_time = time.time()
    x_tcn_output_data_np = x_tcn_output_data
    # np.savetxt('sim_data/result/x_tcn_output_data_sim.txt', x_tcn_output_data_np, delimiter=' ')
    np.savetxt('main/tcn_result/x_tcn_output.txt', x_tcn_output_data_np, delimiter=' ')

    #-------------------------------------AKF計算----------------------------------------#

    # dt = 0.001
    # true_pos = raw_data_all[10000:14999, 0]
    # true_vel = raw_data_all[10000:14999, 1]
    # true_acc = raw_data_all[10000:14999, 2]
    # x_true_data_noise = raw_data_all[10000:14999, 3]
    # print("true_pos =", true_pos)
    # true_pos = true_pos.reshape(-1, 1) 
    # true_vel = true_vel.reshape(-1, 1)
    # true_acc = true_acc.reshape(-1, 1)
    # x_true_data_noise = x_true_data_noise.reshape(-1, 1)
    # pose, vele, acce, Q_pos, Q_acc, Q_vel, u_p_values, Q_save = AKF.AKF_2(dt, x_true_data_noise, true_pos, true_vel, true_acc)
    dt = 0.001
#---------------------------------------UKF-------------------------------------------#  
    UKF_pose, UKF_vele, UKF_acce = UKF.UKF(dt, x_true_noise, 779)

#---------------------------------------EKF-------------------------------------------#  
    EKF_pose, EKF_vele, EKF_acce = EKF.EKF(dt, x_true_noise, 779)

#---------------------------------------KF--------------------------------------------#  
    # KF_pose, KF_vele, KF_acce = KF_v2.KF(dt, x_true_noise, 779)

    print("-------------------------------------------------------------------")
    print(path)
    print("-------------------------------------------------------------------")
    print("TCN 僅估計X的時間: ", end_time - start_time)
    print("TCN 僅估計X平均一筆的時間: ", (end_time - start_time)/validation_size)
    
#---------------------------------------------------------------所有結果比較---------------------------------------------------#
    # plt.figure()
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    x_true_noise = x_true
    # plt.plot(x_true_noise[start_size:start_size + validation_size], label='True_x1_add_noise', color='black', linewidth=3)
    # plt.plot(raw_data_all[start_size:start_size + validation_size, 0], label='true_pos', color='black', linewidth=1)
    plt.plot(np.array(x_k_update_data)[start_size:start_size + validation_size, 0], label='AKF_pos', color='blue', linewidth=1, linestyle="--")
    # plt.plot(np.array(x_RTS_data)[start_size:start_size + validation_size, 0], label='RTS_vel', color='black', linewidth=1, linestyle="-.")
    plt.plot(np.array(x_tcn_output_data)[:, 0], label='DKF_pos', color='red', linewidth=1)
    # plt.plot(UKF_pose[start_size:start_size + validation_size], label='UKF_pos', color='purple', linewidth=1)
    # plt.plot(EKF_pose[start_size:start_size + validation_size], label='EKF_pos', color='orange', linewidth=1)
    # plt.plot(np.array(pose), label='AKF_pos', color='green', linewidth=1)
    plt.xlabel('data')
    plt.ylabel('Pos')
    plt.legend(loc='upper right', ncol=2)
    plt.title('TCN Result')

    # plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 2)
    x_true_noise = x_true
    plt.plot(np.array(x_k_update_data)[start_size:start_size + validation_size, 1], label='AKF_vel', color='blue', linewidth=1, linestyle="--")
    # plt.plot(np.array(x_RTS_data)[start_size:start_size + validation_size, 1], label='RTS_vel', color='black', linewidth=1, linestyle="-.")
    # plt.plot(np.array(x_tcn_output_data)[start_size:start_size + validation_size, 1], label='DKF_vel', color='red', linewidth=1)
    # plt.plot(np.array(x_tcn_output_data)[::-1][:, 0], label='DKF_vel', color='red', linewidth=1)
    # plt.plot(raw_data_all[start_size:start_size + validation_size, 1], label='true_vel', color='black', linewidth=1)
    # plt.plot(UKF_vele[start_size:start_size + validation_size], label='UKF_vel', color='purple', linewidth=1)
    # plt.plot(EKF_vele[start_size:start_size + validation_size], label='EKF_vel', color='orange', linewidth=1)
    # plt.plot(np.array(vele), label='AKF_vel', color='green', linewidth=1)
    plt.xlabel('data')
    plt.ylabel('vel')
    plt.legend(loc='upper right', ncol=2)
    # plt.title('TCN Result')
    plt.title('Vel of TCN Result')

    # plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 3)
    x_true_noise = x_true
    plt.plot(np.array(x_k_update_data)[start_size:start_size + validation_size, 2], label='AKF_acc', color='blue', linewidth=1, linestyle="--")
    # plt.plot(np.array(x_RTS_data)[start_size:start_size + validation_size, 2], label='RTS_vel', color='black', linewidth=1, linestyle="-.")
    # plt.plot(np.array(x_tcn_output_data)[:, 1], label='DKF_vel', color='red', linewidth=1)
    # plt.plot(np.array(x_tcn_output_data)[start_size:start_size + validation_size, 2], label='DKF_acc', color='red', linewidth=1)
    # plt.plot(raw_data_all[start_size:start_size + validation_size, 2], label='true_acc', color='black', linewidth=1)
    # # plt.plot(UKF_acce[start_size:start_size + validation_size], label='UKF_acc', color='purple', linewidth=1)
    # # plt.plot(EKF_acce[start_size:start_size + validation_size], label='EKF_acc', color='orange', linewidth=1)
    # # plt.plot(np.array(acce), label='AKF_acc', color='green', linewidth=1)
    plt.xlabel('data')
    plt.ylabel('acc')
    plt.legend(loc='upper right', ncol=2)
    plt.title('Acc of TCN Result')
    plt.tight_layout()
#-----------------------------------------------------取200筆後穩定的資料-----------------------------------------------------------#
    # # plt.figure()
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    x_true_noise = x_true
    # plt.plot(x_true_noise[start_size:start_size + validation_size], label='True_x1_add_noise', color='black', linewidth=3)
    # plt.plot(raw_data_all[start_size+1000:start_size + validation_size, 0], label='true_pos', color='black', linewidth=1)
    plt.plot(np.array(x_k_update_data)[start_size+200:start_size + validation_size, 0], label='AKF_pos', color='blue', linewidth=1, linestyle="--")
    # plt.plot(np.array(x_RTS_data)[start_size+200:start_size + validation_size, 0], label='RTS_vel', color='black', linewidth=1, linestyle="-.")
    plt.plot(np.array(x_tcn_output_data)[start_size+200:start_size + validation_size, 0], label='DKF_pos', color='red', linewidth=1)
    # plt.plot(UKF_pose[start_size:start_size + validation_size], label='UKF_pos', color='purple', linewidth=1)
    # plt.plot(EKF_pose[start_size:start_size + validation_size], label='EKF_pos', color='orange', linewidth=1)
    # plt.plot(np.array(pose), label='AKF_pos', color='green', linewidth=1)
    plt.xlabel('data')
    plt.ylabel('Pos')
    plt.legend(loc='upper right', ncol=2)
    plt.title('TCN Result after 1000 datas')

    # plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 2)
    x_true_noise = x_true
    plt.plot(np.array(x_k_update_data)[start_size+200:start_size + validation_size, 1], label='AKF_vel', color='blue', linewidth=1, linestyle="--")
    # plt.plot(np.array(x_RTS_data)[start_size+200:start_size + validation_size, 1], label='RTS_vel', color='black', linewidth=1, linestyle="-.")
    # plt.plot(np.array(x_tcn_output_data)[start_size+200:start_size + validation_size, 1], label='DKF_vel', color='red', linewidth=1)
    # plt.plot(np.array(x_tcn_output_data)[::-1][200:, 0], label='DKF_vel', color='red', linewidth=1)
    # plt.plot(raw_data_all[start_size+1000:start_size + validation_size, 1], label='true_vel', color='black', linewidth=1)
    # plt.plot(UKF_vele[start_size:start_size + validation_size], label='UKF_vel', color='purple', linewidth=1)
    # plt.plot(EKF_vele[start_size:start_size + validation_size], label='EKF_vel', color='orange', linewidth=1)
    # plt.plot(np.array(vele), label='AKF_vel', color='green', linewidth=1)
    plt.xlabel('data')
    plt.ylabel('vel')
    plt.legend(loc='upper right', ncol=2)
    plt.title('Vel of TCN Result after 1000 datas')

    # plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 3)
    x_true_noise = x_true
    plt.plot(np.array(x_k_update_data)[start_size+200:start_size + validation_size, 2], label='AKF_acc', color='blue', linewidth=1, linestyle="--")
    # plt.plot(np.array(x_RTS_data)[start_size+200:start_size + validation_size, 2], label='RTS_vel', color='black', linewidth=1, linestyle="-.")
    # plt.plot(np.array(x_tcn_output_data)[200:, 1], label='DKF_vel', color='red', linewidth=1)
    # plt.plot(np.array(x_tcn_output_data)[start_size+200:start_size + validation_size, 2], label='DKF_acc', color='red', linewidth=1)
    # plt.plot(raw_data_all[start_size+1000:start_size + validation_size, 2], label='true_acc', color='black', linewidth=1)
    # plt.plot(UKF_acce[start_size:start_size + validation_size], label='UKF_acc', color='purple', linewidth=1)
    # plt.plot(EKF_acce[start_size:start_size + validation_size], label='EKF_acc', color='orange', linewidth=1)
    # plt.plot(np.array(acce), label='AKF_acc', color='green', linewidth=1)
    plt.xlabel('data')
    plt.ylabel('acc')
    plt.legend(loc='upper right', ncol=2)
    plt.title('Acc of TCN Result after 1000 datas')
    plt.tight_layout()

    # 計算誤差
    # vele = np.array(x_tcn_output_data)[:, 0]
    # acce = np.array(x_tcn_output_data)[:, 0]
    # x_RTS = x_RTS_data
    # result_RTS = calculateError_RTS.calError(filtered_CFD_vele, filtered_CFD_acce, vele, acce, x_RTS[:, 1], x_RTS[:, 2])
    # result2_RTS = calculateError_RTS.calError2(filtered_CFD_vele[600:], filtered_CFD_acce[600:], vele[600:], acce[600:], x_RTS[:, 1][600:], x_RTS[:, 2][600:])

    # Q值結果
    plt.figure()
    plt.plot(np.array(x_tcn_output_data)[:, 0], label='DKF_Q_pos', color='red', linewidth=1)
    plt.xlabel('iteration')
    plt.ylabel('Q values')
    plt.legend(loc='upper right', ncol=1)
    plt.title('Q_pos Result')

    plt.figure()
    plt.plot(np.array(x_tcn_output_data)[:, 0], label='DKF_Q_vel', color='red', linewidth=1)
    plt.xlabel('iteration')
    plt.ylabel('Q values')
    plt.legend(loc='upper right', ncol=1)
    plt.title('Q_vel Result')

    plt.figure()
    plt.plot(np.array(x_tcn_output_data)[:, 0], label='DKF_Q_acc', color='red', linewidth=1)
    plt.xlabel('iteration')
    plt.ylabel('Q values')
    plt.legend(loc='upper right', ncol=1)
    plt.title('Q_acc Result')
    #---------------------------------------------------------------所有結果比較---------------------------------------------------#
    # # plt.figure()
    # plt.figure(figsize=(8, 6))
    # plt.subplot(3, 1, 1)
    # x_true_noise = x_true
    # # plt.plot(x_true_noise[start_size:start_size + validation_size], label='True_x1_add_noise', color='black', linewidth=3)
    # # plt.plot(raw_data_all[start_size:start_size + validation_size, 0], label='true_pos', color='black', linewidth=1)
    # plt.plot(np.array(x_k_update_data)[start_size:start_size + validation_size, 0], label='AKF_pos', color='blue', linewidth=1, linestyle="--")
    # # plt.plot(np.array(x_tcn_output_data)[:, 0], label='DKF_pos', color='red', linewidth=1)
    # # plt.plot(UKF_pose[start_size:start_size + validation_size], label='UKF_pos', color='purple', linewidth=1)
    # # plt.plot(EKF_pose[start_size:start_size + validation_size], label='EKF_pos', color='orange', linewidth=1)
    # # plt.plot(np.array(pose), label='AKF_pos', color='green', linewidth=1)
    # plt.xlabel('data')
    # plt.ylabel('Pos')
    # plt.legend(loc='upper right', ncol=2)
    # plt.title('Result Comparison')

    # # plt.figure(figsize=(8, 6))
    # plt.subplot(3, 1, 2)
    # x_true_noise = x_true
    # plt.plot(np.array(x_k_update_data)[::-1][start_size:start_size + validation_size, 1], label='AKF_vel', color='blue', linewidth=1, linestyle="--")
    # plt.plot(np.array(x_RTS_data)[::-1][start_size:start_size + validation_size, 1], label='RTS_vel', color='black', linewidth=1, linestyle="-.")
    # plt.plot(np.array(x_tcn_output_data)[:, 0], label='DKF_vel', color='red', linewidth=1)
    # # plt.plot(np.array(x_tcn_output_data)[::-1][:, 0], label='DKF_vel', color='red', linewidth=1)
    # # plt.plot(raw_data_all[start_size:start_size + validation_size, 1], label='true_vel', color='black', linewidth=1)
    # # plt.plot(UKF_vele[start_size:start_size + validation_size], label='UKF_vel', color='purple', linewidth=1)
    # # plt.plot(EKF_vele[start_size:start_size + validation_size], label='EKF_vel', color='orange', linewidth=1)
    # # plt.plot(np.array(vele), label='AKF_vel', color='green', linewidth=1)
    # plt.xlabel('data')
    # plt.ylabel('vel')
    # plt.legend(loc='upper right', ncol=2)
    # plt.title('Vel of Result Comparison')

    # # plt.figure(figsize=(8, 6))
    # plt.subplot(3, 1, 3)
    # x_true_noise = x_true
    # plt.plot(np.array(x_k_update_data)[start_size:start_size + validation_size, 2], label='AKF_acc', color='blue', linewidth=1, linestyle="--")
    # # plt.plot(np.array(x_tcn_output_data)[:, 2], label='DKF_acc', color='red', linewidth=1)
    # # plt.plot(raw_data_all[start_size:start_size + validation_size, 2], label='true_acc', color='black', linewidth=1)
    # # plt.plot(UKF_acce[start_size:start_size + validation_size], label='UKF_acc', color='purple', linewidth=1)
    # # plt.plot(EKF_acce[start_size:start_size + validation_size], label='EKF_acc', color='orange', linewidth=1)
    # # plt.plot(np.array(acce), label='AKF_acc', color='green', linewidth=1)
    # plt.xlabel('data')
    # plt.ylabel('acc')
    # plt.legend(loc='upper right', ncol=2)
    # plt.title('Acc of Result Comparison')
    # plt.tight_layout()

    # #---------------------------------------------------------------取後200筆數據所有結果比較---------------------------------------------------#
    # # plt.figure()
    # plt.figure(figsize=(8, 6))
    # plt.subplot(3, 1, 1)
    # x_true_noise = x_true
    # # plt.plot(x_true_noise[start_size:start_size + validation_size], label='True_x1_add_noise', color='black', linewidth=3)
    # # plt.plot(raw_data_all[start_size + validation_size-500:start_size + validation_size, 0], label='true_pos', color='black', linewidth=1)
    # plt.plot(np.array(x_k_update_data)[start_size + validation_size-200:start_size + validation_size, 0], label='AKF_pos', color='blue', linewidth=1, linestyle="--")
    # # plt.plot(np.array(x_tcn_output_data)[validation_size-500:validation_size, 0], label='DKF_pos', color='red', linewidth=1)
    # # plt.plot(UKF_pose[start_size + validation_size-500:start_size + validation_size], label='UKF_pos', color='purple', linewidth=1)
    # # plt.plot(EKF_pose[start_size + validation_size-500:start_size + validation_size], label='EKF_pos', color='orange', linewidth=1)
    # # plt.plot(np.array(pose), label='AKF_pos', color='green', linewidth=1)
    # plt.xlabel('data')
    # plt.ylabel('Pos')
    # plt.legend(loc='upper right', ncol=2)
    # plt.title('Result comparison after 1000 datas')

    # # plt.figure(figsize=(8, 6))
    # plt.subplot(3, 1, 2)
    # x_true_noise = x_true
    # plt.plot(np.array(x_k_update_data)[::-1][start_size + validation_size-200:start_size + validation_size, 1], label='AKF_vel', color='blue', linewidth=1, linestyle="--")
    # plt.plot(np.array(x_RTS_data)[::-1][start_size + validation_size-200:start_size + validation_size, 1], label='RTS_vel', color='black', linewidth=1, linestyle="-.")
    # plt.plot(np.array(x_tcn_output_data)[validation_size-200:validation_size, 0], label='DKF_vel', color='red', linewidth=1)
    # # plt.plot(np.array(x_tcn_output_data)[::-1][validation_size-200:validation_size, 0], label='DKF_vel', color='red', linewidth=1)
    # # plt.plot(raw_data_all[start_size + validation_size-500:start_size + validation_size, 1], label='true_vel', color='black', linewidth=1)
    # # plt.plot(UKF_vele[start_size + validation_size-500:start_size + validation_size], label='UKF_vel', color='purple', linewidth=1)
    # # plt.plot(EKF_vele[start_size + validation_size-500:start_size + validation_size], label='EKF_vel', color='orange', linewidth=1)
    # # plt.plot(np.array(vele), label='AKF_vel', color='green', linewidth=1)
    # plt.xlabel('data')
    # plt.ylabel('vel')
    # plt.legend(loc='upper right', ncol=2)
    # plt.title('Vel of Result comparison after 1000 datas')

    # # plt.figure(figsize=(8, 6))
    # plt.subplot(3, 1, 3)
    # x_true_noise = x_true
    # plt.plot(np.array(x_k_update_data)[start_size + validation_size-200:start_size + validation_size, 2], label='AKF_acc', color='blue', linewidth=1, linestyle="--")
    # # plt.plot(np.array(x_tcn_output_data)[validation_size-500:validation_size, 2], label='DKF_acc', color='red', linewidth=1)
    # # plt.plot(raw_data_all[start_size + validation_size-500:start_size + validation_size, 2], label='true_acc', color='black', linewidth=1)
    # # plt.plot(UKF_acce[start_size + validation_size-500:start_size + validation_size], label='UKF_acc', color='purple', linewidth=1)
    # # plt.plot(EKF_acce[start_size + validation_size-500:start_size + validation_size], label='EKF_acc', color='orange', linewidth=1)
    # # plt.plot(np.array(acce), label='AKF_acc', color='green', linewidth=1)
    # plt.xlabel('data')
    # plt.ylabel('acc')
    # plt.legend(loc='upper right', ncol=2)
    # plt.title('Vel of Result comparison after 1000 datas')
    # plt.tight_layout()

    plt.show()