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
import TCN, dataset_arrange
from TCN import TemporalConvNet
import setTCNConfig
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
# 測試 TCN 模型
if __name__ == "__main__":
    # 訓練參數設置
    epoch = 300
    traning_size = 10000
    batch_size = 2500
    data_set_size = traning_size

    # 輸入模擬資料
    path1 = 'sim_dataset/x_data_all_15000_0.001.txt'
    path2 = 'sim_dataset/P_data_all_15000_0.001.txt'
    x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_obsve, x_input_data_all, x_k_predict_data, P_data, P_k_update_data, KCP_data, P_input_data_all = dataset_arrange.loadSimData(path1, path2)
    
    setConfig = setTCNConfig.TCNConfig()
    input_size, output_size, kernel_size,  stride, dropout, num_channels = setConfig.getTCNConfig()

    x_tcn_model = TCN.TemporalConvNet(num_inputs=input_size, num_classes=output_size, kernel_size=kernel_size,  stride=stride, dropout=dropout, num_channels=num_channels)
    x_tcn_model = x_tcn_model.to(device)
    # input_tensor = torch.tensor(np.vstack(x_input_data_all), dtype=torch.float32).unsqueeze(1).to(device)
    # 定義損失函數和優化器
    x_loss_fn = nn.MSELoss()  # 可根據任務選擇合適的損失函數，這裡以 MSE 為例
    x_optimizer = optim.Adam(x_tcn_model.parameters(), lr=0.001)

    x_y_true_all = []
    x_y_pred_all = []
    x_loss_data = []
    x_rmse_loss_data = []
    x_rmse_total_data = []
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
            # x_input_tensor = torch.tensor(cp.vstack(x_input_data), dtype=torch.float32).to(device)
            # print("shape :", x_input_tensor.shape)
            # x_input_tensor = x_input_tensor.permute(0, 2, 1) 
            # LSTM進行狀態估計
            x_tcn_output = x_tcn_model(x_input_tensor)
            
            # 計算損失
            x_target = torch.tensor(cp.array(x_k_update_data)[1:,:], dtype=torch.float32).to(device)
            x_loss = x_loss_fn(x_tcn_output[1:batch_size, :2], x_target[i+1:i+batch_size,:2]) #可以得到一個epoch中每筆資料的mse
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
        if epoch % 1 == 0:
            print(f'----------------------------------------')
            print(f'|Epoch : {epoch}/{total_epoch} | x_Loss_RMSE : {x_rmse_total.item():.4f}|')

    # 計算 RMSE
    x_y_true_all = cp.array(x_y_true_all)
    x_y_pred_all = cp.array(x_y_pred_all)

    # x result儲存模型
    full_path = "sim/sim_model/x_tcn_model_fea{}_ker{}_num{}_epo{}.pth".format(input_size, kernel_size, num_channels, total_epoch)
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

    # plt.show() 
    pylab.show()