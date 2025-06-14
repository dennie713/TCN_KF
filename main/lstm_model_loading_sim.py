import cupy as cp
# import numpy as cp
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import LSTM, dataset_arrange
import setLSTMConfig
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # 參數設置
    start_size = 0
    validation_size = 12000
    # 輸入模擬資料
    # 選擇輸入資料
    scara = 1
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

    # 參數設置
    setConfig = setLSTMConfig.LSTMConfig()
    x_input_size, x_output_size, hidden_size, num_layers, dropout = setConfig.getLSTMConfig()

    # 解標準化
    # 讀取
    data = np.load('lstm_normalization/normalizer.npz')
    x_mean = data['x_mean']
    x_std = data['x_std']
    y_mean = data['y_mean']
    y_std = data['y_std']
    print("y_mean =", y_mean)
    print("y_std =", y_std)

    # 加載模型
    x_lstm_model_loaded = LSTM.LSTM_KF(x_input_size, hidden_size, x_output_size, num_layers, dropout)  # 創建模型實例
    x_lstm_model_loaded.load_state_dict(torch.load('main/lstm_model/lstm_model_in3_out3_hid64_layer2_epo202.pth', weights_only=True))  # 加載權重
    x_lstm_model_loaded.eval()  # 將模型設置為評估模式
    x_lstm_model_loaded = x_lstm_model_loaded.to(device)
    # hidden_size = 128
    # start_size = 0
    # validation_size = 15000 # diff: 14683 # same: 23000
    data_set_size = start_size + validation_size

    # 輸入標準化
    standardization = 1
    if standardization == 1:
        x_input_data_all_mean = x_input_data_all.mean(axis=0, keepdims=True)
        x_input_data_all_std = x_input_data_all.std(axis=0, keepdims=True) + 1e-8
        x_input_data_all_norm = (x_input_data_all - x_input_data_all_mean) / x_input_data_all_std

    start_time = time.time()
    x_tcn_output_data = []

    # --------x_model loading --------#
    x_lstm_output_data = []
    for k in range(start_size, start_size + validation_size):
        # print("k =", k)
        # x_tel = cp.array(x_true) - cp.array(x_k_update_data)
        if standardization == 1:
            x_input_data = x_input_data_all_norm[k]
        else:
            x_input_data = x_input_data_all[k]
        # x_input_data = x_input_data_all[k]
        x_input_data = torch.tensor(cp.hstack(x_input_data), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        x_input_tensor = x_input_data.clone().detach().to(device)
        # x_input_tensor = x_input_data.permute(0, 2, 1)

        # 使用模型進行推斷
        with torch.no_grad():  # 禁用梯度計算以提高推斷效率
            x_lstm_output = x_lstm_model_loaded(x_input_tensor)  # 獲取模型的輸出
            x_lstm_output_denorm = x_lstm_output.detach().cpu() * y_std + y_mean
            x_lstm_output_delog = torch.exp(x_lstm_output_denorm)
        x_lstm_output_data.append(x_lstm_output_delog.detach().cpu().numpy().flatten())
        # print("x LSTM Output:", x_lstm_output[:, :3])  # 輸出結果
    # print('-----------------')

    # print("x =",cp.reshape(cp.array(x_lstm_output_data)[-1, :3], (3, 1)))

    end_time = time.time()
    # 先将 cupy 数组转换为 numpy 数组
    x_lstm_output_data_np = x_lstm_output_data
    # P_lstm_output_data_np = P_lstm_output_data
    # 使用 numpy.savetxt 将其保存到 txt 文件中
    os.makedirs('main/lstm_result', exist_ok=True)
    np.savetxt('main/lstm_result/x_lstm_output.txt', x_lstm_output_data_np, delimiter=' ')
    # np.savetxt('./result/P_lstm_output_data_motor.txt', P_lstm_output_data_np, delimiter=' ')

    # print("Plotting ...")
    # 估測狀態匯出
    # plt.figure()
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    x_true_noise = x_true
    # plt.plot(x_true_noise[start_size:start_size + validation_size], label='True_x1_add_noise', color='black', linewidth=3)
    # plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 0].get(), label='LKF_pos', linewidth=2)
    plt.plot(cp.array(x_lstm_output_data)[:, 0].get(), label='DKF_pos',  linewidth=1)
    # plt.plot(raw_data_all[start_size:start_size + validation_size, 0], label='true_pos', linewidth=1)
    plt.xlabel('Pos')
    plt.ylabel('value')
    plt.legend()
    plt.title('LSTM result')

    # plt.figure()
    plt.subplot(3, 1, 2)
    x_true_noise = x_true
    # plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 1].get(), label='LKF_vel', linewidth=2)

    plt.plot(cp.array(x_lstm_output_data)[:, 1].get(), label='DKF_vel', linewidth=1)
    # plt.plot(raw_data_all[start_size:start_size + validation_size, 1], label='true_vel', linewidth=1)
    plt.xlabel('Vel')
    plt.ylabel('value')
    # plt.legend()
    # plt.title('Vel of estimate vs true')

    # plt.figure()
    plt.subplot(3, 1, 3)
    x_true_noise = x_true
    # plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 2].get(), label='LKF_acc', linewidth=2)
    plt.plot(cp.array(x_lstm_output_data)[:, 2].get(), label='DKF_acc', linewidth=1)
    # plt.plot(raw_data_all[start_size:start_size + validation_size, 2], label='true_acc', linewidth=1)
    plt.xlabel('Acc')
    plt.ylabel('value')
    # plt.legend()
    # plt.title('Acc of estimate vs true')

    # # 估測狀態誤差匯出
    # x_k_update_data = cp.array(x_k_update_data).reshape(-1, 1)  # reshape to 2D
    # x_true = cp.array(x_true).reshape(-1, 1)  # reshape to 2D
    # plt.figure()
    # a = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 0] - cp.array(x_true)[start_size:start_size + validation_size, 0])
    # plt.plot(a.get(), label='LKF_x1_err', color='black', linewidth=2)
    # # b = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 1] - cp.array(x_true)[start_size:start_size + validation_size, 1])
    # # plt.plot(b.get(), label='LKF_x2', color='green', linewidth=2)
    # # e = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 2] - cp.array(x_true)[start_size:start_size + validation_size, 2])
    # # plt.plot(e.get(), label='LKF_x3', color='purple', linewidth=2)
    # c = cp.abs(cp.array(x_lstm_output_data)[:, 0] - cp.array(x_true)[start_size:start_size + validation_size, 0])
    # plt.plot(c.get(), label='DKF_x1_err', color='blue', linewidth=1)
    # # d = cp.abs(cp.array(x_lstm_output_data)[:, 1] - cp.array(x_true)[start_size:start_size + validation_size, 1])
    # # plt.plot(d.get(), label='DKF_x2', color='red', linewidth=1)
    # # f = cp.abs(cp.array(x_lstm_output_data)[:, 2] - cp.array(x_true)[start_size:start_size + validation_size, 2])
    # # plt.plot(f.get(), label='DKF_x3', color='red', linewidth=1)
    # plt.xlabel('data')
    # plt.ylabel('estimate value')
    # plt.legend()
    # plt.title('pos error between true_pos')
    # # print("Finish Plotting ...")

    plt.show()