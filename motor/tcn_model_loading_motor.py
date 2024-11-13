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
from TCN import TemporalConvNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # 模型參數
    setConfig = setTCNConfig.TCNConfig()
    input_size, output_size, kernel_size,  stride, dropout, num_channels = setConfig.getTCNConfig()

    # 輸入模擬資料
    path1 = 'motor_dataset/Motor_x_data_ips750_g50_addnoise.txt'
    path2 = 'motor_dataset/Motor_P_data_ips750_g50_addnoise.txt' 
    x_data, x_true, x_k_update_data, x_cmd, km_y_data, x_tel, x_input_data_all, P_data, P_k_update_data, KCP_data, P_input_data_all = dataset_arrange.loadMotorData(path1, path2)
    
    # 參數設置
    start_size = 0
    validation_size = len(x_k_update_data)
    data_set_size = start_size + validation_size

    # 加載模型
    path = 'motor/motor_model/x_tcn_model_fea10_ker8_num[64]_epo300.pth'
    x_tcn_model_loaded = TCN.TemporalConvNet(num_inputs=input_size, num_channels=num_channels, num_classes=output_size, kernel_size=kernel_size,  stride=stride, dropout=dropout)
    x_tcn_model_loaded.load_state_dict(torch.load(path, weights_only=True))  # 加載權重
    x_tcn_model_loaded.eval()  # 將模型設置為評估模式  
    x_tcn_model = x_tcn_model_loaded.to(device)

    start_time = time.time()
    x_tcn_output_data = []
    for k in range(start_size, start_size + validation_size):
        # print("k =", k)
        # x_tel = cp.array(x_true) - cp.array(x_k_update_data)
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
    np.savetxt('motor/motor_result/x_tcn_output_data_sim.txt', x_tcn_output_data_np, delimiter=' ')

    print("-----------------------------------------")
    print(path)
    print("-----------------------------------------")
    print("TCN 僅估計X的時間: ", end_time - start_time)
    print("TCN 僅估計X平均一筆的時間: ", (end_time - start_time)/validation_size)
    
    # # 估測狀態匯出
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

    plt.figure()
    x_true_noise = x_true
    plt.plot(cp.array(x_cmd)[:, 0].get(), label='CMD_POS', color='black', linewidth=3)
    plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 0].get(), label='LKF_POS', color='orange', linewidth=2)
    plt.plot(cp.array(x_tcn_output_data)[:, 0].get(), label='TCN_POS', color='purple', linewidth=1)
    plt.xlabel('data')
    plt.ylabel('value')
    plt.legend()
    plt.title('Pos of estimate vs true')

    plt.figure()
    x_true_noise = x_true
    plt.plot(cp.array(x_cmd)[:, 1].get(), label='CMD_VEL', color='blue', linewidth=3)
    plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 1].get(), label='LKF_VEL', color='cyan', linewidth=2)
    plt.plot(cp.array(x_tcn_output_data)[:, 1].get(), label='TCN_VEL', color='red', linewidth=1)
    plt.xlabel('data')
    plt.ylabel('value')
    plt.legend()
    plt.title('Vel of estimate vs true')

    plt.figure()
    x_true_noise = x_true
    plt.plot(cp.array(x_cmd)[:, 2].get(), label='CMD_ACC', color='gray', linewidth=3)
    plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 2].get(), label='LKF_ACC', color='pink', linewidth=2)
    plt.plot(cp.array(x_tcn_output_data)[:, 2].get(), label='TCN_ACC', color='green', linewidth=1)
    plt.xlabel('data')
    plt.ylabel('value')
    plt.legend()
    plt.title('Acc of estimate vs true')

    # 估測狀態誤差匯出
    # x_k_update_data = cp.array(x_k_update_data).reshape(-1, 1)  # reshape to 2D
    # x_true = cp.array(x_true).reshape(-1, 1)  # reshape to 2D
    # plt.figure()
    # a = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 0] - cp.array(x_true)[start_size:start_size + validation_size, 0])
    # plt.plot(a.get(), label='LKF_x1_err', color='black', linewidth=2)
    # # b = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 1] - cp.array(x_true)[start_size:start_size + validation_size, 1])
    # # plt.plot(b.get(), label='LKF_x2', color='green', linewidth=2)
    # # e = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 2] - cp.array(x_true)[start_size:start_size + validation_size, 2])
    # # plt.plot(e.get(), label='LKF_x3', color='purple', linewidth=2)
    # c = cp.abs(cp.array(x_tcn_output_data)[:, 0] - cp.array(x_true)[start_size:start_size + validation_size, 0])
    # plt.plot(c.get(), label='DKF_x1_err', color='blue', linewidth=1)
    # d = cp.abs(cp.array(x_lstm_output_data)[:, 1] - cp.array(x_true)[start_size:start_size + validation_size, 1])
    # plt.plot(d.get(), label='DKF_x2', color='red', linewidth=1)
    # f = cp.abs(cp.array(x_lstm_output_data)[:, 2] - cp.array(x_true)[start_size:start_size + validation_size, 2])
    # plt.plot(f.get(), label='DKF_x3', color='red', linewidth=1)
    # plt.xlabel('data')
    # plt.ylabel('estimate value')
    # plt.legend()
    # plt.title('pos error between true_pos')
    
    d = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 0] - cp.array(x_cmd)[start_size:start_size + validation_size, 0])
    # plt.plot(a.get(), label='LKF_x1_err', color='black', linewidth=2)
    e = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 1] - cp.array(x_cmd)[start_size:start_size + validation_size, 1])
    # plt.plot(a.get(), label='LKF_x1_err', color='black', linewidth=2)
    f = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 2] - cp.array(x_cmd)[start_size:start_size + validation_size, 2])
    g = cp.abs(cp.array(x_tcn_output_data)[:, 0] - cp.array(x_cmd)[start_size:start_size + validation_size, 0])
    # plt.plot(a.get(), label='LKF_x1_err', color='black', linewidth=2)
    h = cp.abs(cp.array(x_tcn_output_data)[:, 1] - cp.array(x_cmd)[start_size:start_size + validation_size, 1])
    # plt.plot(a.get(), label='LKF_x1_err', color='black', linewidth=2)
    i = cp.abs(cp.array(x_tcn_output_data)[:, 2] - cp.array(x_cmd)[start_size:start_size + validation_size, 2])

    print("-----------------------------------------")
    # print("LKF_x1 mean error :", np.mean(a.get()))
    # # print("LKF_x2 mean error :", np.mean(b))
    # print("DKF_x1 mean error :", np.mean(c.get()))
    print("LKF_pos mean error :", np.mean(d.get()))
    print("LKF_vel mean error :", np.mean(e.get()))
    print("LKF_acc mean error :", np.mean(f.get()))
    print("TCN_pos mean error :", np.mean(g.get()))
    print("TCN_vel mean error :", np.mean(h.get()))
    print("TCN_acc mean error :", np.mean(i.get()))
    # print("DKF_x2 mean error :", np.mean(d))
    print("-----------------------------------------")

    plt.show()