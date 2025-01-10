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
import TCN, dataset_arrange, AKF
import setTCNConfig
from TCN import TemporalConvNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # 參數設置
    start_size = 10000
    validation_size = 5000
    data_set_size = start_size + validation_size

    # 輸入模擬資料
    path1 = 'sim_data/dataset/x_data_all_scara_15000_Q12_with_noise.txt'
    path2 = 'sim_data/dataset/P_data_all_scara_15000_Q12_with_noise.txt'
    path3 = 'sim_data/dataset/raw_data_all_scara_15000_Q12_with_noise.txt'
    x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_input_data_all, P_data, P_k_update_data, KCP_data, P_input_data_all, raw_data_all, x_k_predict_data  = dataset_arrange.loadSimData(path1, path2, path3)
    
    # print("x_input_data_all.shape =", x_input_data_all.shape)
    # print("x_input_data_all =", x_input_data_all)
    # 模型參數
    setConfig = setTCNConfig.TCNConfig()
    input_size, output_size, kernel_size,  stride, dropout, num_channels = setConfig.getTCNConfig()

    # 加載模型
    path = 'sim_data/model/x_tcn_fea7_ker5_num[32]_epo300_scara_paper_adj.pth'
    x_tcn_model_loaded = TCN.TemporalConvNet(num_inputs=input_size, num_channels=num_channels, num_classes=output_size, kernel_size=kernel_size,  stride=stride, dropout=dropout)
    x_tcn_model_loaded.load_state_dict(torch.load(path, weights_only=True))  # 加載權重
    x_tcn_model_loaded.eval()  # 將模型設置為評估模式  
    x_tcn_model = x_tcn_model_loaded.to(device)

    start_time = time.time()
    x_tcn_output_data = []
    for k in range(start_size, 4999):
        # print("k =", k)
        # x_tel = cp.array(x_true) - cp.array(x_k_update_data)
        # print("x_input_data_all[k] =", x_input_data_all[k])
        x_input_data = x_input_data_all[k]
        x_input_data = torch.tensor(cp.hstack(x_input_data), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        # x_input_tensor = x_input_data.clone().detach().to(device)
        x_input_tensor = x_input_data.permute(0, 2, 1)

        # 使用模型進行推斷
        with torch.no_grad():  # 禁用梯度計算以提高推斷效率
            x_tcn_output = x_tcn_model_loaded(x_input_tensor)  # 獲取模型的輸出
        x_tcn_output_data.append(x_tcn_output.detach().cpu().numpy().flatten())
        # print("x TCN Output:", x_tcn_output[:, :3].cpu().numpy())  # 輸出結果
    end_time = time.time()
    x_tcn_output_data_np = x_tcn_output_data
    np.savetxt('sim_data/result/x_tcn_output_data_sim.txt', x_tcn_output_data_np, delimiter=' ')

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


    print("-----------------------------------------")
    print(path)
    print("-----------------------------------------")
    print("TCN 僅估計X的時間: ", end_time - start_time)
    print("TCN 僅估計X平均一筆的時間: ", (end_time - start_time)/validation_size)
    # 估測狀態匯出
    # plt.figure()
    # # x_true_noise = x_true
    # # plt.plot(x_true[start_size:start_size + validation_size, 0], label='True_x1', color='black', linewidth=3)
    # # plt.plot(x_true[start_size:start_size + validation_size, 1], label='True_x2', color='blue', linewidth=3)
    # plt.plot(x_true_noise[start_size:start_size + validation_size, 0], label='True_x1_add_noise', color='black', linewidth=3)
    # plt.plot(x_true_noise[start_size:start_size + validation_size, 1], label='True_x2_add_noise', color='blue', linewidth=3)
    
    # # print("x_k_update_data =", x_k_update_data)
    # plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 0].get(), label='LKF_x1', color='orange', linewidth=2)
    # plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 1].get(), label='LKF_x2', color='cyan', linewidth=2)
    # # plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 2].get(), label='LKF_x3', color='pink', linewidth=2)
    # plt.plot(cp.array(x_tcn_output_data)[:, 0].get(), label='DKF_x1', color='purple', linewidth=1)
    # plt.plot(cp.array(x_tcn_output_data)[:, 1].get(), label='DKF_x2', color='red', linewidth=1)
    # # plt.plot(cp.array(x_tcn_output_data)[:, 2].get(), label='DKF_x3', color='green', linewidth=1)
    # plt.xlabel('data')
    # plt.ylabel('value')
    # plt.legend()
    # plt.title('estimate vs true :x1 x2')

    # 估測狀態匯出
    print("x_tcn_output_data.shape =", np.array(x_tcn_output_data).shape)
    # plt.figure()
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    x_true_noise = x_true
    # plt.plot(x_true_noise[start_size:start_size + validation_size], label='True_x1_add_noise', color='black', linewidth=3)
    plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 0].get(), label='LKF_pos', color='blue', linewidth=1, linestyle="--")
    plt.plot(cp.array(x_tcn_output_data)[:, 0].get(), label='DKF_pos', color='red', linewidth=1)
    plt.plot(raw_data_all[start_size:start_size + validation_size, 0], label='true_pos', color='black', linewidth=1)
    # plt.plot(cp.array(pose).get(), label='AKF_pos', color='green', linewidth=1)
    plt.xlabel('data')
    plt.ylabel('Pos')
    plt.legend()
    plt.title('TCN result')

    # plt.figure()
    plt.subplot(3, 1, 2)
    x_true_noise = x_true
    plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 1].get(), label='LKF_vel', color='blue', linewidth=1, linestyle="--")
    plt.plot(cp.array(x_tcn_output_data)[:, 1].get(), label='DKF_vel', color='red', linewidth=1)
    plt.plot(raw_data_all[start_size:start_size + validation_size, 1], label='true_vel', color='black', linewidth=1)
    # plt.plot(cp.array(vele).get(), label='AKF_vel', color='green', linewidth=1)
    plt.xlabel('data')
    plt.ylabel('value')
    # plt.legend()
    # plt.title('Vel of estimate vs true')

    # plt.figure()
    plt.subplot(3, 1, 3)
    x_true_noise = x_true
    plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 2].get(), label='LKF_acc', color='blue', linewidth=1, linestyle="--")
    plt.plot(cp.array(x_tcn_output_data)[:, 2].get(), label='DKF_acc', color='red', linewidth=1)
    plt.plot(raw_data_all[start_size:start_size + validation_size, 2], label='true_acc', color='black', linewidth=1)
    # plt.plot(cp.array(acce).get(), label='AKF_acc', color='green', linewidth=1)
    plt.xlabel('data')
    plt.ylabel('value')
    # plt.legend()
    # plt.title('Acc of estimate vs true')

    # # 估測狀態誤差匯出
    # plt.figure()
    # # x_k_update_data = cp.array(x_k_update_data).reshape(-1, 1)
    # # print("x_k_update_data =", x_k_update_data)
    # # print("x_true =", x_true)
    # a = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 0] - cp.array(x_true)[start_size:start_size + validation_size, 0])
    # plt.plot(a.get(), label='LKF_x1', color='orange', linewidth=1)
    # b = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 1] - cp.array(x_true)[start_size:start_size + validation_size, 1])
    # plt.plot(b.get(), label='LKF_x2', color='cyan', linewidth=1)
    # # e = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 2] - cp.array(x_true)[start_size:start_size + validation_size, 2])
    # # plt.plot(e.get(), label='LKF_x2', color='purple', linewidth=2)
    # c = cp.abs(cp.array(x_tcn_output_data)[:, 0] - cp.array(x_true)[start_size:start_size + validation_size, 0])
    # plt.plot(c.get(), label='DKF_x1', color='purple', linewidth=1)
    # d = cp.abs(cp.array(x_tcn_output_data)[:, 1] - cp.array(x_true)[start_size:start_size + validation_size, 1])
    # plt.plot(d.get(), label='DKF_x2', color='red', linewidth=1)
    # # f = cp.abs(cp.array(x_tcn_output_data)[:, 2] - cp.array(x_true)[start_size:start_size + validation_size, 2])
    # # plt.plot(f.get(), label='DKF_x2', color='red', linewidth=1)
    # plt.xlabel('data')
    # plt.ylabel('estimate value')
    # plt.legend()
    # plt.title('estimate pos vel acc')

    # print("-----------------------------------------")
    # print("LKF_x1 mean error :", np.mean(a))
    # print("LKF_x2 mean error :", np.mean(b))
    # print("DKF_x1 mean error :", np.mean(c))
    # print("DKF_x2 mean error :", np.mean(d))
    # print("-----------------------------------------")

    plt.show()