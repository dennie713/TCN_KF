import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import LSTM, dataset_arrange, LogCoshLoss
import setLSTMConfig, EarlyStopping
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
        # 訓練參數設置
        epoch = 500
        start_size = 0  # 訓練資料起始位置
        traning_size = 12000   # diff: 14683 ；diff_2: 29366 # same: 23000
        batch_size = 100
        data_set_size = traning_size - start_size

        setConfig = setLSTMConfig.LSTMConfig()
        x_input_size, x_output_size, hidden_size, num_layers, dropout = setConfig.getLSTMConfig()
        # x初始化LSTM模型
        x_lstm_model = LSTM.LSTM_KF(x_input_size, hidden_size, x_output_size, num_layers, dropout)
        x_lstm_model = x_lstm_model.to(device)
        x_optimizer = torch.optim.Adam(x_lstm_model.parameters(), lr=1e-4, weight_decay=1e-5)
        x_loss_fn = nn.MSELoss()


        # 馬達實際資料
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

        # 早停函數
        early_stopper = EarlyStopping.EarlyStopping(patience=10, verbose=True)

        # 定義損失函數和優化器
        x_loss_fn = nn.MSELoss() 

        # LogCoshLoss 損失函數
        LogCoshLoss_loss_fn = LogCoshLoss.LogCoshLoss()
        StableLogCoshLoss_loss_fn = LogCoshLoss.StableLogCoshLoss()

        x_y_true_all = []
        x_y_pred_all = []
        x_loss_data = []
        x_rmse_loss_data = []
        x_rmse_total_data = []
        # P_y_true_all = []
        # P_y_pred_all = []
        # P_loss_data = []
        # P_rmse_loss_data = []
        # P_rmse_total_data = []
        # print("x_input_data=", x_input_data_all.shape)

        train_x_data = x_input_data_all[start_size:traning_size, :]
        # print("train_x_data =", train_x_data)

        # 卡爾曼之後的結果
        # train_y_data = x_k_update_data[start_size:traning_size, :]
        # RTS平滑後的結果
        # train_y_data = x_RTS_data[start_size:traning_size, 2]
        # train_y_data = K_RTS_data[::-1]
        train_y_data = Q_data_all[start_size:traning_size, :]#.reshape(-1, 1)
        # print("train_y_data =", train_y_data)

        # 標準化
        standardization = 1
        if standardization == 1:
                x_mean = train_x_data.mean(axis=0, keepdims=True)
                x_std = train_x_data.std(axis=0, keepdims=True) + 1e-8  # 避免除以 0

                train_y_data_log = np.log(train_y_data + 1e-8)  # 避免 log(0) 的情況
                train_y_data = train_y_data_log
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
                np.savez('lstm_normalization/normalizer.npz', x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
        else:
                train_x_data = train_x_data
                train_y_data = train_y_data

        total_epoch = epoch
        for epoch in range(epoch + 1):
                x_total_loss = 0
                x_rmse_loss_data.clear()
                # P_total_loss = 0

                # 創建批次數據
                x_lstm_model.train()
                x_input_data = []
                for i in range(0, batch_size):
                        if standardization == 1:
                                batch_x_input_data_all = train_x_data_norm[i:i+batch_size]
                        else:
                                batch_x_input_data_all = train_x_data[i:i+batch_size]
                        # 添加到批次列表中
                        x_input_data = batch_x_input_data_all# me
                        # 將數據轉換為張量，並添加一個維度以符合 LSTM 的輸入格式
                        
                        x_input_tensor = torch.tensor(np.vstack(x_input_data), dtype=torch.float32).unsqueeze(0).to(device)
                        
                        # LSTM進行狀態估計
                        x_lstm_output = x_lstm_model(x_input_tensor)

                        # 計算損失
                        # x_target = torch.tensor(np.array(x_input_data_all)[1:,:2], dtype=torch.float32).to(device)
                        # x_loss = x_loss_fn(x_lstm_output[1:batch_size, :2], x_target[i+1:i+batch_size,:2]) #可以得到一個epoch中每筆資料的mse
                        # x_target = torch.tensor(np.array(x_k_update_data)[:, :3], dtype=torch.float32).to(device)
                        if standardization == 1:
                                x_target = torch.tensor(train_y_data_norm[i:i + batch_size].copy(), dtype=torch.float32).to(device)
                        else:
                                # x_target = torch.tensor(train_y_data[i:i + batch_size], dtype=torch.float32).to(device)
                                x_target = torch.tensor(train_y_data[i:i + batch_size].copy(), dtype=torch.float32).to(device)
                        # min1 = np.min(np.array(x_k_update_data)[:, 0])
                        # max1 = np.max(np.array(x_k_update_data)[:, 0])
                        # norm1 = max1 - min1
                        # x_lstm_output[:, 0] = (x_lstm_output[:, 0]-min1)/norm1
                        # x_target[:, 0] = (x_target[:, 0]-min1)/norm1

                        # min2 = np.min(np.array(x_k_update_data)[:, 1])
                        # max2 = np.max(np.array(x_k_update_data)[:, 1])
                        # norm2 = max2 - min2
                        # x_lstm_output[:, 1] = (x_lstm_output[:, 1]-min2)/norm2
                        # x_target[:, 1] = (x_target[:, 1]-min2)/norm2

                        # min3 = np.min(np.array(x_k_update_data)[:, 2])
                        # max3 = np.max(np.array(x_k_update_data)[:, 2])
                        # norm3 = max3 - min3
                        # x_lstm_output[:, 2] = (x_lstm_output[:, 2]-min3)/norm3
                        # x_target[:, 2] = (x_target[:, 2]-min3)/norm3

                        # x_loss0 = x_loss_fn(x_lstm_output[0:batch_size, 0], x_target[i:i+batch_size, 0])
                        # x_loss1 = x_loss_fn(x_lstm_output[0:batch_size, 1], x_target[i:i+batch_size, 1])
                        # x_loss2 = x_loss_fn(x_lstm_output[0:batch_size, 2], x_target[i:i+batch_size, 2])
                        # print(f'[pos_loss:{x_loss0} -- vel_loss:{x_loss1} -- acc_loss:{x_loss2}]')


                        # x_loss = x_loss_fn(x_lstm_output[0:batch_size, 0:3], x_target[i:i+batch_size, 0:3])
                        # ------------------------------mse損失函數------------------------------ #
                        # x_loss = x_loss_fn(x_lstm_output[:batch_size, :], acc_tar)
                        # x_loss = x_loss_fn(x_lstm_output[:batch_size, :], x_target)

                        # ------------------------------LogCoshLoss損失函數------------------------------ #
                        # x_loss = LogCoshLoss_loss_fn(x_lstm_output[:batch_size, :], acc_tar)
                        x_loss = StableLogCoshLoss_loss_fn(x_lstm_output[:batch_size, :], x_target)

                        # x_loss = x_loss_fn(x_lstm_output[0:batch_size, :1], x_target[i:i+batch_size]) #可以得到一個epoch中每筆資料的mse
                        x_loss_data.append(x_loss.item()) 
                        x_rmse_loss = torch.sqrt(x_loss) #可以得到一個epoch中每筆資料的rmse
                        x_rmse_loss_data.append(x_rmse_loss.item())
                        x_total_loss += x_rmse_loss.item()

                        # 保存真實值和預測值
                        # x_true = np.array(x_true)
                        x_y_true_all.append(x_true.flatten())
                        x_y_pred_all.append(x_lstm_output.detach().cpu().numpy().flatten())
                        # print("x_y_pred_all =", x_y_pred_all)

                        # 反向傳播和參數更新
                        x_optimizer.zero_grad()
                        x_loss.backward()
                        # 限制梯度範圍
                        torch.nn.utils.clip_grad_norm_(x_lstm_model.parameters(), max_norm=1.0)
                        x_optimizer.step()

                x_rmse_total = np.sqrt(np.mean(np.array(x_rmse_loss_data)**2)) #可以得到每一個epoch的rmse
                x_rmse_total_data.append(x_rmse_total)
                if epoch % 1 == 0:
                        print(f'-------------------------------------------------------------------')
                        print(f'|Epoch : {epoch}/{total_epoch} | x_Loss_RMSE : {x_rmse_total.item():.7f}|')
                        # print(f'|Epoch : {epoch}/{total_epoch} | x_Loss_RMSE : {x_rmse_total.item():.4f} | P_Loss_RMSE : {P_rmse_total.item():.4f}|')

                # 計算 RMSE
                # x_y_true_all = np.array(x_y_true_all)
                # x_y_pred_all = np.array(x_y_pred_all)
                # P_y_true_all = np.array(P_y_true_all)
                # P_y_pred_all = np.array(P_y_pred_all)

                # os.makedirs('motor/motor_model', exist_ok=True)
                # x result儲存模型
                # torch.save(x_lstm_model.state_dict(), 'main/lstm_model/lstm_model_in{}_out{}_hid{}_layer{}.pth'.format(x_input_size, x_output_size, hidden_size, num_layers))
                # print("-------- x Model saved successfully --------")

                full_path = 'main/lstm_model/lstm_model_in{}_out{}_hid{}_layer{}_epo{}.pth'.format(x_input_size, x_output_size, hidden_size, num_layers, total_epoch)
                early_stopper(x_rmse_total, x_lstm_model, path=full_path)

                if early_stopper.early_stop:  
                        final_stop_path = 'main/lstm_model/lstm_model_in{}_out{}_hid{}_layer{}_epo{}.pth'.format(x_input_size, x_output_size, hidden_size, num_layers, epoch)
                        torch.save(x_lstm_model.state_dict(), final_stop_path)
                        print("Early stopping triggered!")
                        print("model path:", final_stop_path)
                        break
                elif epoch == total_epoch:
                        # x result儲存模型
                        full_path = 'main/lstm_model/lstm_model_in{}_out{}_hid{}_layer{}_epo{}.pth'.format(x_input_size, x_output_size, hidden_size, num_layers, total_epoch)

                        print("model path:", full_path)
                        torch.save(x_lstm_model.state_dict(), full_path)
                        print("-------- Model saved successfully --------")  


        # --------狀態估測誤差模型-------- #
        plt.figure(figsize=(12, 6))
        plt.plot(np.array(x_rmse_loss_data), label='loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('x_loss')
        plt.legend()
        plt.title('x RMSE for every data in each epoch')

        plt.figure(figsize=(12, 6))
        Epoch = np.arange(1, len(x_rmse_total_data) + 1)
        plt.plot(Epoch, np.array(x_rmse_total_data), label='loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('x_loss')
        plt.legend()
        plt.title('Epoch vs RMSE')


        plt.show()
