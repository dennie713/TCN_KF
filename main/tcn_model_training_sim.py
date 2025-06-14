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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import TCN, dataset_arrange, LogCoshLoss, OriKF
from TCN import TemporalConvNet
import setTCNConfig, EarlyStopping
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

    
# 測試 TCN 模型
if __name__ == "__main__":
    # 訓練參數設置
    epoch = 800
    start_size = 1000 # 350
    traning_size = 12000
    batch_size = 256
    data_set_size = traning_size - start_size

    # 輸入模擬資料
    # path1 = 'sim_data/dataset/x_data_all_AKF_28995_comb.txt'
    # path2 = 'sim_data/dataset/P_data_all_AKF_28995_comb.txt'
    # path3 = 'sim_data/dataset/raw_data_all_AKF_28995_comb.txt'
    # path4 = 'sim_data/dataset/Q_save_AKF_28995_comb.txt'
    # path5 = 'sim_data/dataset/x_input_data_all_KF_28995_comb.txt'
    # 選擇輸入資料
    scara = 2
    if scara == 1:
        path1 = 'main/dataset/Real_AKF_OLS_6axis2_n=10_n1n2=20_12000/x_data_all_AKF.txt'
        path2 = 'main/dataset/Real_AKF_OLS_6axis2_n=10_n1n2=20_12000/P_data_all_AKF.txt'
        path3 = 'main/dataset/Real_AKF_OLS_6axis2_n=10_n1n2=20_12000/raw_data_all_AKF.txt'
        path4 = 'main/dataset/Real_AKF_OLS_6axis2_n=10_n1n2=20_12000/Q_data_all_AKF.txt'
        # path5 = 'main/dataset/Real_AKF_OLS_6axis2_n=10_n1n2=20_12000/x_RTS_AKF.txt'
        # path6 = 'main/dataset/Real_AKF_OLS_6axis2_n=10_n1n2=20_12000/K_RTS_AKF.txt'
        path7 = 'main/dataset/Real_AKF_OLS_6axis2_n=10_n1n2=20_12000/est_err_data_all_AKF.txt'
        path8 = 'main/dataset/Real_AKF_OLS_6axis2_n=10_n1n2=20_12000/G_tel_data_all_AKF.txt'
    elif scara == 2:
        path1 = 'main/dataset/Real_AKF_OLS_6axis3_n=10_n1n2=20_12000/x_data_all_AKF.txt'
        path2 = 'main/dataset/Real_AKF_OLS_6axis3_n=10_n1n2=20_12000/P_data_all_AKF.txt'
        path3 = 'main/dataset/Real_AKF_OLS_6axis3_n=10_n1n2=20_12000/raw_data_all_AKF.txt'
        path4 = 'main/dataset/Real_AKF_OLS_6axis3_n=10_n1n2=20_12000/Q_data_all_AKF.txt'
        # path5 = 'main/dataset/Real_AKF_OLS_6axis3_n=10_n1n2=20_12000/x_RTS_AKF.txt'
        # path6 = 'main/dataset/Real_AKF_OLS_6axis3_n=10_n1n2=20_12000/K_RTS_AKF.txt'
        path7 = 'main/dataset/Real_AKF_OLS_6axis3_n=10_n1n2=20_12000/est_err_data_all_AKF.txt'
        path8 = 'main/dataset/Real_AKF_OLS_6axis3_n=10_n1n2=20_12000/G_tel_data_all_AKF.txt'
    elif scara == 3:
        path1 = 'main/dataset/Real_AKF_OLS_scara1_n=10_12000/x_data_all_AKF.txt'
        path2 = 'main/dataset/Real_AKF_OLS_scara1_n=10_12000/P_data_all_AKF.txt'
        path3 = 'main/dataset/Real_AKF_OLS_scara1_n=10_12000/raw_data_all_AKF.txt'
        path4 = 'main/dataset/Real_AKF_OLS_scara1_n=10_12000/Q_data_all_AKF.txt'
        path5 = 'main/dataset/Real_AKF_OLS_scara1_n=10_12000/x_RTS_AKF.txt'
        # path6 = 'main/dataset/Real_AKF_OLS_scara1_n=10_12000/K_RTS_AKF.txt'
        path7 = 'main/dataset/Real_AKF_OLS_scara1_n=10_12000/est_err_data_all_AKF.txt'
        # path8 = 'main/dataset/Real_AKF_OLS_scara1_n=10_12000/G_tel_data_all_AKF.txt'
    elif scara == 4:
        path1 = 'main/dataset/Real_AKF_OLS_scara2_n=10_12000/x_data_all_AKF.txt'
        path2 = 'main/dataset/Real_AKF_OLS_scara2_n=10_12000/P_data_all_AKF.txt'
        path3 = 'main/dataset/Real_AKF_OLS_scara2_n=10_12000/raw_data_all_AKF.txt'
        path4 = 'main/dataset/Real_AKF_OLS_scara2_n=10_12000/Q_data_all_AKF.txt'
        path5 = 'main/dataset/Real_AKF_OLS_scara2_n=10_12000/x_RTS_AKF.txt'
        # path6 = 'main/dataset/Real_AKF_OLS_scara2_n=10_12000/K_RTS_AKF.txt'
        path7 = 'main/dataset/Real_AKF_OLS_scara2_n=10_12000/est_err_data_all_AKF.txt'
        # path8 = 'main/dataset/Real_AKF_OLS_scara2_n=10_12000/G_tel_data_all_AKF.txt'
    # Q_data = np.loadtxt(path4, delimiter=' ')
    # x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_input_data_all, P_data, P_k_update_data, KCP_data, P_input_data_all, raw_data_all, x_k_predict_data, Q_data_all = dataset_arrange.loadSimData(path1, path2, path3, path4, path7, path8)
    x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_input_data_all, P_data, P_k_update_data, KCP_data, P_input_data_all, raw_data_all, x_k_predict_data, Q_data_all = dataset_arrange.loadSimData(path1, path2, path3, path4, path7)

    if scara == 3 or scara == 4:
        x_RTS_data = np.loadtxt(path5, delimiter=' ').reshape(-1, 3) # RTS平滑後的結果
        # print("x_RTS_data.shape =", x_RTS_data.shape)

    setConfig = setTCNConfig.TCNConfig()
    input_size, output_size, kernel_size,  stride, dropout, num_channels = setConfig.getTCNConfig()
    x_tcn_model = TCN.TemporalConvNet(num_inputs=input_size, num_classes=output_size, kernel_size=kernel_size,  stride=stride, dropout=dropout, num_channels=num_channels)
    print('x_tcn_model =', x_tcn_model)
    # paper lstm model setting
    # x_tcn_model = TCN.TCN_Q(input_dim=1)
    x_tcn_model = x_tcn_model.to(device)
    # input_tensor = torch.tensor(np.vstack(x_input_data_all), dtype=torch.float32).unsqueeze(1).to(device)
    
    # 早停函數
    early_stopper = EarlyStopping.EarlyStopping(patience=10, verbose=True)

    # 定義損失函數和優化器
    x_loss_fn = nn.MSELoss() 

    # LogCoshLoss 損失函數
    LogCoshLoss_loss_fn = LogCoshLoss.LogCoshLoss()
    StableLogCoshLoss_loss_fn = LogCoshLoss.StableLogCoshLoss()

    x_optimizer = optim.Adam(x_tcn_model.parameters(), lr=1e-6) # 0.0001
    x_scheduler = ReduceLROnPlateau(x_optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    x_y_true_all = []
    x_y_pred_all = []
    x_loss_data = []
    x_rmse_loss_data = []
    x_rmse_total_data = []

    val_rmse_data = [] 
    total_epoch = epoch
    # 訓練Q
    # x_input_data_all = Q_data
    # x_k_update_data = Q_data
    # x_k_update_data = np.concatenate((Q_data[:, 0].reshape(-1, 1), Q_data[:, 4].reshape(-1, 1), Q_data[:, 8].reshape(-1, 1)), axis=1)

    train_x_data = x_input_data_all[start_size:traning_size, :]
    # train_x_data = x_input_data_all[1000:traning_size, :]
    # print("train_x_data =", train_x_data)

    # 卡爾曼之後的結果
    # train_y_data = x_k_update_data[start_size:traning_size, :]
    # RTS平滑後的結果
    # train_y_data = x_RTS_data[start_size:traning_size, :]
    # train_y_data = K_RTS_data[start_size:traning_size, :]
    train_y_data = Q_data_all[start_size:traning_size, :]#.reshape(-1, 1)
    # train_y_data = Q_data_all[1000:traning_size, :]
    # print("train_y_data =", train_y_data)

    # 標準化
    standardization = 1
    if standardization == 1:
        x_mean = train_x_data.mean(axis=0, keepdims=True)
        x_std = train_x_data.std(axis=0, keepdims=True) + 1e-8  # 避免除以 0

        # 換成LOG標準化
        # train_y_data_log = np.log(train_y_data + 1e-8)  # 防止 log(0)
        # train_y_data = train_y_data_log
        y_mean = train_y_data.mean(axis=0, keepdims=True)
        y_std = train_y_data.std(axis=0, keepdims=True) + 1e-8

        train_x_data_norm = (train_x_data - x_mean) / x_std
        train_y_data_norm = (train_y_data - y_mean) / y_std

        # 儲存於 dict 或寫入檔案
        normalizer = {
            'x_mean': x_mean,
            'x_std': x_std,
            'y_mean': y_mean,
            'y_std': y_std
        }
        # 儲存
        np.savez('tcn_normalization/normalizer.npz', x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    else:
        # train_x_data_log = np.log(train_x_data + 1e-8)  # 防止 log(0)
        # train_y_data_log = np.log(train_y_data + 1e-8)
        # train_x_data = train_x_data_log
        train_y_data = train_y_data
        train_x_data = train_x_data
        # train_y_data = train_y_data
        # train_y_data = np.log(train_y_data + 1e-8)  # 防止 log(0)


    # 透過標準差來初始化可學習參數
    # std_1 = train_y_data[:, 0].std().item()
    # std_2 = train_y_data[:, 1].std().item()
    # std_3 = train_y_data[:, 2].std().item()

    # 訓練過程
    # validation_interval = 1
    for epoch in range(epoch+1):
        x_total_loss = 0
        x_rmse_loss_data.clear()

        # 創建批次數據
        x_tcn_model.train()
        x_input_data = []
        for i in range(0, batch_size):
            # print("i =", i)
            # batch_x_input_data_all = x_input_data_all[i:i+batch_size] # me
            if standardization == 1:
                batch_x_input_data_all = train_x_data_norm[i:i+batch_size]
            else:
                batch_x_input_data_all = train_x_data[i:i+batch_size]
            # 添加到批次列表中
            x_input_data = batch_x_input_data_all# me
            # print("x_input_data =", x_input_data)
            # 將數據轉換為張量，並添加一個維度以符合 TCN 的輸入格式
            x_input_tensor = torch.tensor(np.vstack(x_input_data), dtype=torch.float32).unsqueeze(2).to(device)

            # LSTM進行狀態估計
            x_tcn_output = x_tcn_model(x_input_tensor)
            # print("x_tcn_output :", x_tcn_output)
            
            # 計算損失
            if standardization == 1:
                x_target = torch.tensor(train_y_data_norm[i:i + batch_size].copy(), dtype=torch.float32).to(device)
            else:
                # x_target = torch.tensor(train_y_data[i:i + batch_size], dtype=torch.float32).to(device)
                x_target = torch.tensor(train_y_data[i:i + batch_size].copy(), dtype=torch.float32).to(device)
            # vel_tar = x_target[:, 1]
            # acc_tar = x_target[:, 2]
            # ------------------------------Multi-task Learning損失函數------------------------------ #
            # sigma1 = nn.Parameter(torch.tensor(std_1, dtype=torch.float32, requires_grad=True))
            # sigma2 = nn.Parameter(torch.tensor(std_2, dtype=torch.float32, requires_grad=True))
            # sigma3 = nn.Parameter(torch.tensor(std_3, dtype=torch.float32, requires_grad=True))

            # loss_1 = x_loss_fn(x_tcn_output[:, 0], x_target[:, 0])
            # loss_2 = x_loss_fn(x_tcn_output[:, 1], x_target[:, 1])
            # loss_3 = x_loss_fn(x_tcn_output[:, 2], x_target[:, 2])

            # 多個損失函數的加權和
            # x_loss = (loss_1 / (2 * sigma1**2) + loss_2 / (2 * sigma2**2)) + torch.log(sigma1 * sigma2)
            # x_loss = (loss_1 / (2 * sigma1**2) + loss_2 / (2 * sigma2**2) + loss_3 / (2 * sigma3**2)) + torch.log(sigma1 * sigma2 * sigma3)
            # x_loss = (loss_3 / (2 * sigma3**2)) + torch.log(sigma3)
            # x_loss = loss_1
            # ------------------------------mse損失函數------------------------------ #
            # x_loss = x_loss_fn(x_tcn_output[:batch_size, :], acc_tar)
            # x_loss = x_loss_fn(x_tcn_output[:batch_size, :], x_target)

            # ------------------------------LogCoshLoss損失函數------------------------------ #
            # x_loss = LogCoshLoss_loss_fn(x_tcn_output[:batch_size, :], acc_tar)
            x_loss = StableLogCoshLoss_loss_fn(x_tcn_output[:batch_size, :], x_target)
            
            # ------------------------------輸出的Tr(Q)作為損失函數------------------------------ #
            # x_loss = x_tcn_output.sum(dim=1).mean()

            # x_loss = x_loss + 0.2 * Q_loss # 損失函數 = mse損失函數 + Q_loss

            x_loss_data.append(x_loss.item()) 
            x_rmse_loss = torch.sqrt(x_loss) #可以得到一個epoch中每筆資料的rmse
            x_rmse_loss_data.append(x_rmse_loss.item())
            x_total_loss += x_rmse_loss.item()

            # 保存真實值和預測值
            
            x_y_true_all.append(x_true.flatten())
            # if standardization == 1:
            #     x_pred_denorm = x_tcn_output.detach().cpu().numpy() * y_std + y_mean
            #     x_y_pred_all.append(x_pred_denorm.flatten())
            # else:
            x_y_pred_all.append(x_tcn_output.detach().cpu().numpy().flatten())
            # print("x_y_pred_all =", x_y_pred_all)

            # 反向傳播和參數更新
            x_optimizer.zero_grad()
            x_loss.backward()
            # 限制梯度範圍
            torch.nn.utils.clip_grad_norm_(x_tcn_model.parameters(), max_norm=1.0)
            x_optimizer.step()
            
        x_rmse_total = cp.sqrt(cp.mean(cp.array(x_rmse_loss_data)**2)) #可以得到每一個epoch的rmse
        x_rmse_total_data.append(x_rmse_total)
        x_scheduler.step(x_rmse_total)

        if epoch % 1 == 0:
            print(f'-------------------------------------')
            print(f'|Epoch: {epoch}/{total_epoch} | x_Loss_RMSE: {x_rmse_total:.6f}|')
            # print(f'|Epoch : {epoch}/{total_epoch} | x_Loss_RMSE : {x_rmse_total.item():.4f}|')
        if scara == 1 or scara == 2:
            full_path = "main/tcn_model/TCN_6axis_fea{}_ker{}_num{}_epo{}.pth".format(input_size, kernel_size, num_channels, total_epoch)
        else:
            full_path = "main/tcn_model/TCN_scara_fea{}_ker{}_num{}_epo{}.pth".format(input_size, kernel_size, num_channels, total_epoch)
        # full_path = "main/tcn_model/TCN_fea{}_ker{}_num{}_epo{}.pth".format(input_size, kernel_size, num_channels, total_epoch)
        early_stopper(x_rmse_total, x_tcn_model, path=full_path)

        if early_stopper.early_stop:  
            if scara == 1 or scara == 2:
                final_stop_path = "main/tcn_model/TCN_6axis_fea{}_ker{}_num{}_epo{}.pth".format(input_size, kernel_size, num_channels, total_epoch)
            else:
                final_stop_path = "main/tcn_model/TCN_scara_fea{}_ker{}_num{}_epo{}.pth".format(input_size, kernel_size, num_channels, total_epoch)
            # final_stop_path = "main/tcn_model/TCN_fea{}_ker{}_num{}_epo{}.pth".format(input_size, kernel_size, num_channels, epoch)
            torch.save(x_tcn_model.state_dict(), final_stop_path)
            print("Early stopping triggered!")
            print("model path:", final_stop_path)
            break
        elif epoch == total_epoch:
            # x result儲存模型
            # full_path = "main/tcn_model/TCN_fea{}_ker{}_num{}_epo{}.pth".format(input_size, kernel_size, num_channels, total_epoch)

            print("model path:", full_path)
            torch.save(x_tcn_model.state_dict(), full_path)
            print("-------- Model saved successfully --------")  
    # 計算 RMSE
    # x_y_true_all = cp.array(x_y_true_all)
    # x_y_pred_all = cp.array(x_y_pred_all)

    

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
    # plt.title('Training RMSE per batch')

    # plt.show() 
    pylab.show()