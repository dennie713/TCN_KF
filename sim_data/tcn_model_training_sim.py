import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import pylab
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
<<<<<<< HEAD
import TCN, dataset_arrange
from TCN import TemporalConvNet
import setTCNConfig
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
=======
import TCN, dataset_arrange, LogCoshLoss, OriKF
from TCN import TemporalConvNet
import setTCNConfig
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
>>>>>>> 924f379 (v3)
    
# 測試 TCN 模型
if __name__ == "__main__":
    # 訓練參數設置
    epoch = 300
<<<<<<< HEAD
    traning_size = 10000
    batch_size = 100
    data_set_size = traning_size

    # 輸入模擬資料
    path1 = 'sim_data/dataset/x_data_all_scara_15000_with_noise_adj.txt'
    path2 = 'sim_data/dataset/P_data_all_scara_15000_with_noise_adj.txt'
    path3 = 'sim_data/dataset/raw_data_all_scara_15000_with_noise_adj.txt'
    x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_input_data_all, P_data, P_k_update_data, KCP_data, P_input_data_all, raw_data_all, x_k_predict_data  = dataset_arrange.loadSimData(path1, path2, path3)
    
    setConfig = setTCNConfig.TCNConfig()
    input_size, output_size, kernel_size,  stride, dropout, num_channels = setConfig.getTCNConfig()

    x_tcn_model = TCN.TemporalConvNet(num_inputs=input_size, num_classes=output_size, kernel_size=kernel_size,  stride=stride, dropout=dropout, num_channels=num_channels)
    x_tcn_model = x_tcn_model.to(device)
    # input_tensor = torch.tensor(np.vstack(x_input_data_all), dtype=torch.float32).unsqueeze(1).to(device)
    # 定義損失函數和優化器
    x_loss_fn = nn.MSELoss()  # 可根據任務選擇合適的損失函數，這裡以 MSE 為例
    x_optimizer = optim.Adam(x_tcn_model.parameters(), lr=0.001)
=======
    start_size = 0
    traning_size = 14000
    batch_size = 500
    data_set_size = traning_size - start_size

    # 輸入模擬資料
    # path1 = 'sim_data/dataset/x_data_all_AKF_28995_comb.txt'
    # path2 = 'sim_data/dataset/P_data_all_AKF_28995_comb.txt'
    # path3 = 'sim_data/dataset/raw_data_all_AKF_28995_comb.txt'
    # path4 = 'sim_data/dataset/Q_save_AKF_28995_comb.txt'
    # path5 = 'sim_data/dataset/x_input_data_all_KF_28995_comb.txt'
    path1 = 'sim_data/dataset/x_data_all_AKF_15000_exp2_2.txt'
    path2 = 'sim_data/dataset/P_data_all_AKF_15000_exp2_2.txt'
    path3 = 'sim_data/dataset/raw_data_all_AKF_15000_exp2_2.txt'
    path4 = 'sim_data/dataset/Q_save_AKF_15000_exp2_2.txt'
    path5 = 'sim_data/dataset/x_input_data_all_KF_28995_comb.txt'
    Q_data = np.loadtxt(path4, delimiter=' ')
    x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_input_data_all, P_data, P_k_update_data, KCP_data, P_input_data_all, raw_data_all, x_k_predict_data  = dataset_arrange.loadSimData(path1, path2, path3, path4)
    
    # 一維KF數據輸出
    # x_data_all, P_data_all, x_input_data_all = OriKF.KalmanFilter(0.001, x_true_noise)
    # x_k_update_data = np.array(x_k_update_data)
    # x_input_data_all = np.loadtxt(path5, delimiter=' ')

    setConfig = setTCNConfig.TCNConfig()
    input_size, output_size, kernel_size,  stride, dropout, num_channels = setConfig.getTCNConfig()
    x_tcn_model = TCN.TemporalConvNet(num_inputs=input_size, num_classes=output_size, kernel_size=kernel_size,  stride=stride, dropout=dropout, num_channels=num_channels)
    x_tcn_model = x_tcn_model.to(device)
    # input_tensor = torch.tensor(np.vstack(x_input_data_all), dtype=torch.float32).unsqueeze(1).to(device)
    
    # 定義損失函數和優化器
    x_loss_fn = nn.MSELoss() 

    # LogCoshLoss 損失函數
    # LogCoshLoss_loss_fn = LogCoshLoss.LogCoshLoss()

    x_optimizer = optim.Adam(x_tcn_model.parameters(), lr=0.0001) # 0.001
>>>>>>> 924f379 (v3)

    x_y_true_all = []
    x_y_pred_all = []
    x_loss_data = []
    x_rmse_loss_data = []
    x_rmse_total_data = []
<<<<<<< HEAD
    total_epoch = epoch
    for epoch in range(epoch + 1):
        x_total_loss = 0
        P_total_loss = 0

        # 創建批次數據
        x_input_data = []
        for i in range(0, traning_size, batch_size):
            batch_x_input_data_all = x_input_data_all[i:i+batch_size] # me
            # 添加到批次列表中
            x_input_data = batch_x_input_data_all# me
            # 將數據轉換為張量，並添加一個維度以符合 LSTM 的輸入格式
            x_input_tensor = torch.tensor(cp.vstack(x_input_data), dtype=torch.float32).unsqueeze(2).to(device)
=======
    val_rmse_data = [] 
    total_epoch = epoch
    # 訓練Q
    # x_input_data_all = Q_data
    # x_k_update_data = Q_data
    x_k_update_data = np.concatenate((Q_data[:, 0].reshape(-1, 1), Q_data[:, 4].reshape(-1, 1), Q_data[:, 8].reshape(-1, 1)), axis=1)

    # 分割訓練集與驗證集
    train_size = int(1 * data_set_size)  # 80% for training
    # print("train_size =", train_size)
    val_size = traning_size - train_size  # 20% for validation
    train_x_data = x_input_data_all[:train_size]
    val_x_data = x_input_data_all[train_size:]
    train_y_data = x_k_update_data[:train_size]
    val_y_data = x_k_update_data[train_size:]

    # 透過標準差來初始化可學習參數
    std_1 = train_y_data[:, 0].std().item()
    std_2 = train_y_data[:, 1].std().item()
    std_3 = train_y_data[:, 2].std().item()

    # 訓練過程
    # validation_interval = 1
    for epoch in range(epoch+1):
        x_total_loss = 0
        x_rmse_loss_data.clear()

        # 創建批次數據
        x_tcn_model.train()
        x_input_data = []
        for i in range(start_size, train_size, batch_size):
            # print("i =", i)
            # batch_x_input_data_all = x_input_data_all[i:i+batch_size] # me
            batch_x_input_data_all = train_x_data[i:i+batch_size]
            # print("batch_x_input_data_all=", batch_x_input_data_all)
            # 添加到批次列表中
            x_input_data = batch_x_input_data_all# me
            # print("x_input_data =", x_input_data)
            # 將數據轉換為張量，並添加一個維度以符合 TCN 的輸入格式
            x_input_tensor = torch.tensor(np.vstack(x_input_data), dtype=torch.float32).unsqueeze(2).to(device)
            # print("x_input_tensor =", x_input_tensor)
>>>>>>> 924f379 (v3)
            # x_input_tensor = torch.tensor(cp.vstack(x_input_data), dtype=torch.float32).to(device)
            # print("shape :", x_input_tensor.shape)
            # x_input_tensor = x_input_tensor.permute(0, 2, 1) 
            # LSTM進行狀態估計
            x_tcn_output = x_tcn_model(x_input_tensor)
<<<<<<< HEAD
            
            # 計算損失
            x_target = torch.tensor(cp.array(x_k_update_data)[:,:], dtype=torch.float32).to(device)
            x_loss = x_loss_fn(x_tcn_output[:batch_size, :3], x_target[i:i+batch_size,:3]) #可以得到一個epoch中每筆資料的mse
=======
            # print("x_tcn_output :", x_tcn_output)
            
            # 計算損失
            x_target = torch.tensor(train_y_data[i:i + batch_size], dtype=torch.float32).to(device)
            vel = x_target[:, 1:2]
            acc = x_target[:, 2:3]
            # ------------------------------Multi-task Learning損失函數------------------------------ #
            sigma1 = nn.Parameter(torch.tensor(std_1, dtype=torch.float32, requires_grad=True))
            sigma2 = nn.Parameter(torch.tensor(std_2, dtype=torch.float32, requires_grad=True))
            sigma3 = nn.Parameter(torch.tensor(std_3, dtype=torch.float32, requires_grad=True))

            loss_1 = x_loss_fn(x_tcn_output[:, 0], x_target[:, 0])
            loss_2 = x_loss_fn(x_tcn_output[:, 1], x_target[:, 1])
            loss_3 = x_loss_fn(x_tcn_output[:, 2], x_target[:, 2])

            # x_loss = (loss_1 / (2 * sigma2**2) + loss_2 / (2 * sigma3**2)) + torch.log(sigma2 * sigma3)
            x_loss = (loss_1 / (2 * sigma1**2) + loss_2 / (2 * sigma2**2) + loss_3 / (2 * sigma3**2)) + torch.log(sigma1 * sigma2 * sigma3)
            # ------------------------------mse損失函數------------------------------ #
            # x_loss = x_loss_fn(x_tcn_output[:batch_size, :], vel)
            # print("x_loss =", x_loss)
            # ------------------------------LogCoshLoss損失函數------------------------------ #
            # x_loss_fn = LogCoshLoss_loss_fn(x_tcn_output[:batch_size, :], x_target[:, 1:3])
            # x_loss = x_loss_fn

            # x_loss = x_loss_fn(x_tcn_output[:batch_size, :3], x_target[i:i+batch_size,:3]) #可以得到一個epoch中每筆資料的mse
>>>>>>> 924f379 (v3)
            # x_target = torch.tensor(cp.array(x_input_data_all)[:, 1:4], dtype=torch.float32).to(device)
            # x_loss = loss_fn(x_tcn_output[0:batch_size, :3], x_target[i:i+batch_size]) #可以得到一個epoch中每筆資料的mse
            x_loss_data.append(x_loss.item()) 
            x_rmse_loss = torch.sqrt(x_loss) #可以得到一個epoch中每筆資料的rmse
            x_rmse_loss_data.append(x_rmse_loss.item())
            x_total_loss += x_rmse_loss.item()

            # 保存真實值和預測值
            x_y_true_all.append(x_true.flatten())
            x_y_pred_all.append(x_tcn_output.detach().cpu().numpy().flatten())
            # print("x_y_pred_all =", x_y_pred_all)

            # 反向傳播和參數更新
            x_optimizer.zero_grad()
            x_loss.backward()
            x_optimizer.step()
        x_rmse_total = cp.sqrt(cp.mean(cp.array(x_rmse_loss_data)**2)) #可以得到每一個epoch的rmse
        x_rmse_total_data.append(x_rmse_total)
<<<<<<< HEAD
        if epoch % 1 == 0:
            print(f'----------------------------------------')
            print(f'|Epoch : {epoch}/{total_epoch} | x_Loss_RMSE : {x_rmse_total.item():.4f}|')

    # 計算 RMSE
    x_y_true_all = cp.array(x_y_true_all)
    x_y_pred_all = cp.array(x_y_pred_all)

    # x result儲存模型
    full_path = "sim_data/model/x_tcn_model_fea{}_ker{}_num{}_epo{}.pth".format(input_size, kernel_size, num_channels, total_epoch)
=======

        # 驗證過程
        # if epoch % validation_interval == 0:
        # x_tcn_model.eval()
        # with torch.no_grad():
        #     val_x_input_tensor = torch.tensor(np.vstack(val_x_data), dtype=torch.float32).unsqueeze(2).to(device)
        #     val_tcn_output = x_tcn_model(val_x_input_tensor)
        #     val_target = torch.tensor(val_y_data, dtype=torch.float32).to(device)
        #     # ------------------------------mse損失函數------------------------------ #
        #     val_loss_fn = x_loss_fn(val_tcn_output[:val_size, :], val_target[:, 1:3])
        #     # ------------------------------LogCoshLoss損失函數------------------------------ #
        #     # val_loss_fn = LogCoshLoss_loss_fn(val_tcn_output[:val_size, :], val_target[:, 1:3])

        #     val_loss = val_loss_fn
        #     val_rmse_loss = torch.sqrt(val_loss)
        # val_rmse_data.append(val_rmse_loss.item())
        # print(f"|Epoch: {epoch}/{total_epoch} | Validation RMSE: {val_rmse_loss.item():.4f}|")

        # x_optimizer.zero_grad()
        # x_loss.backward()
        # x_optimizer.step()
        # x_tcn_model.train()

        if epoch % 1 == 0:
            print(f'-------------------------------------')
            print(f'|Epoch: {epoch}/{total_epoch} | x_Loss_RMSE: {x_rmse_total:.4f}|')
            # print(f'|Epoch : {epoch}/{total_epoch} | x_Loss_RMSE : {x_rmse_total.item():.4f}|')

    # 計算 RMSE
    # x_y_true_all = cp.array(x_y_true_all)
    # x_y_pred_all = cp.array(x_y_pred_all)

    # x result儲存模型
    full_path = "sim_data/model/TCN_fea{}_ker{}_num{}_epo{}.pth".format(input_size, kernel_size, num_channels, total_epoch)
>>>>>>> 924f379 (v3)
    print(full_path)
    torch.save(x_tcn_model.state_dict(), full_path)
    print("-------- x Model saved successfully --------")  

    # --------x 狀態估測誤差模型-------- #
    plt.figure(figsize=(12, 6))
    plt.plot(cp.array(x_rmse_loss_data).get(), label='loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('x_loss')
    plt.legend()
    plt.title('x RMSE for every data in each epoch')

    plt.figure(figsize=(12, 6))
    Epoch = cp.arange(1, len(x_rmse_total_data) + 1)
    plt.plot(Epoch.get(), cp.array(x_rmse_total_data).get(), label='loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('x_loss')
    plt.legend()
    plt.title('Epoch vs RMSE')

<<<<<<< HEAD
=======
    # plt.figure(figsize=(12, 6))
    # Epoch = np.arange(1, len(x_rmse_total_data) + 1)
    # plt.plot(Epoch, cp.array(x_rmse_total_data).get(), label='Training RMSE', color='blue')
    # # plt.plot(Epoch, cp.array(val_rmse_data).get(), label='Validation RMSE', color='red')
    # plt.xlabel('Epoch')
    # plt.ylabel('RMSE')
    # plt.legend()
    # plt.title('Epoch vs RMSE')

    # plt.figure(figsize=(12, 6))
    # plt.plot(cp.array(x_rmse_loss_data).get(), label='Training loss RMSE', color='blue')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    plt.title('Training RMSE per batch')

>>>>>>> 924f379 (v3)
    # plt.show() 
    pylab.show()