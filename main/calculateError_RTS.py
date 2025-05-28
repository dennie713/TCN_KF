import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# print("===================================== Calculating Error =====================================")
def RMSE(y, y_true):
    """Calculate the Root Mean Square Error between true and predicted values."""
    return np.sqrt(np.mean((y - y_true) ** 2))

def MAE(y, y_true):
    """Calculate the Mean Absolute Error between true and predicted values."""
    return np.mean(np.abs(y - y_true))

def calError(true_vel, true_acc, vele, acce, RTS_vele, RTS_acce):
    #----------------------------------------儲存圖片-------------------------------------------#
    save_jpg = 0 # 0:no save, 1:save
    save_svg = 0 # 0:no save, 1:save

    # 計算RMSE -> RMSE_pos = np.sqrt(np.mean((pose - true_pos)**2))
    vele = np.array(vele).flatten()
    acce = np.array(acce).flatten()
    # ================================= AKF ================================
    AKF_RMSE_vele = RMSE(vele, true_vel)
    AKF_RMSE_acce = RMSE(acce, true_acc)
    # ================================= RTS ================================
    RTS_RMSE_vele = RMSE(RTS_vele, true_vel)
    RTS_RMSE_acce = RMSE(RTS_acce, true_acc)

    # print results
    print("--------------- Error Result ---------------")
    print("------------------- RMSE -------------------")
    print("AKF RMSE vel :", AKF_RMSE_vele)
    print("AKF RMSE acc :", AKF_RMSE_acce)
    print("RTS RMSE vel :", RTS_RMSE_vele)
    print("RTS RMSE acc :", RTS_RMSE_acce)
    print("--------------------------------------------")

    # 建立誤差數據
    error_data = {
        "Method": ["AKF", "AKF", "RTS", "RTS"],
        "Type":   ["RMSE_vel", "RMSE_acc", "RMSE_vel", "RMSE_acc"],
        "Value":  [AKF_RMSE_vele, AKF_RMSE_acce, RTS_RMSE_vele, RTS_RMSE_acce]
    }
    # 建立 DataFrame
    df = pd.DataFrame(error_data)
    # 存成 Excel
    # df.to_excel("error_record/RMSE_error_results.xlsx", index=False)

    # 計算MAE -> MAE_pos = np.mean(np.abs(pose - true_pos))
    # ================================= AKF ================================
    AKF_MAE_vele = MAE(vele, true_vel)
    AKF_MAE_acce = MAE(acce, true_acc)
    # ================================= RTS ================================
    RTS_MAE_vele = MAE(RTS_vele, true_vel)
    RTS_MAE_acce = MAE(RTS_acce, true_acc)

    # print results
    print("------------------- MAE -------------------")
    print("AKF MAE vel :", AKF_MAE_vele)
    print("AKF MAE acc :", AKF_MAE_acce)
    print("RTS MAE vel :", RTS_MAE_vele)
    print("RTS MAE acc :", RTS_MAE_acce)
    print("--------------------------------------------")

    # 建立誤差數據
    error_data = {
        "Method": ["AKF", "AKF", "RTS", "RTS"],
        "Type":   ["MAE_vel", "MAE_acc", "MAE_vel", "MAE_acc"],
        "Value":  [AKF_MAE_vele, AKF_MAE_acce, RTS_MAE_vele, RTS_MAE_acce]
    }
    # 建立 DataFrame
    df = pd.DataFrame(error_data)
    # 存成 Excel
    # df.to_excel("error_record/MAE_error_results.xlsx", index=False)

    # ================================== plot RMSE and MAE =====================================
    # plot RMSE
    plt.figure(figsize=(10, 7))
    plt.suptitle("RMSE and MAE Result -with RTS", fontsize=14, fontweight='bold')
    plt.subplot(2, 2, 1)
    plt.title("Vel RMSE Result")
    # x = ['AKF', 'LSF14', 'LSF28', 'TSE2', 'TSE3', 'BDE3', 'CFD', 'KF']
    # h = [AKF_RMSE_vele, LSF_RMSE_vele, LSF28_RMSE_vele, TSE2_RMSE_vele, TSE3_RMSE_vele, BDE3_RMSE_vele, CFD_RMSE_vele, KF_RMSE_vele]
    # c = ['skyblue', 'orange', 'green', 'red', 'purple', 'pink', 'brown', 'blue']
    x = ['AKF', 'RTS']
    h = [AKF_RMSE_vele, RTS_RMSE_vele]
    c = ['skyblue', 'orange']
    bars = plt.bar(x, h, color=c, width=0.2)
    for bar in bars:
        yval = bar.get_height()  # 取得 bar 的高度（數值）
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.5f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(x, x)
    plt.xlabel('Method')
    plt.ylabel('rad/s')
    # plt.legend()

    plt.subplot(2, 2, 2)
    x = np.arange(2)
    width = 0.4
    plt.title("Acc RMSE Result")
    x = ['AKF', 'RTS']
    h = [AKF_RMSE_acce, RTS_RMSE_acce]
    c = ['skyblue', 'orange']
    bars = plt.bar(x, h, color=c, width=0.2)
    for bar in bars:
        yval = bar.get_height()  # 取得 bar 的高度（數值）
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.5f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(x, x)
    plt.yscale('log')
    plt.xlabel('Method')
    plt.ylabel('rad/s^2')
    # plt.legend()

    plt.subplot(2, 2, 3)
    plt.title("Vel MAE Result")
    # x = ['AKF', 'LSF14', 'LSF28', 'TSE2', 'TSE3', 'BDE3', 'CFD', 'KF']
    # h = [AKF_MAE_vele, LSF_MAE_vele, LSF28_MAE_vele, TSE2_MAE_vele, TSE3_MAE_vele, BDE3_MAE_vele, CFD_MAE_vele, KF_MAE_vele]
    # c = ['skyblue', 'orange', 'green', 'red', 'purple', 'pink', 'brown', 'blue']
    x = ['AKF', 'RTS']
    h = [AKF_MAE_vele, RTS_MAE_vele]
    c = ['skyblue', 'orange']
    bars = plt.bar(x, h, color=c, width=0.2)
    for bar in bars:
        yval = bar.get_height()  # 取得 bar 的高度（數值）
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.5f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(x, x)
    plt.xlabel('Method')
    plt.ylabel('rad/s')
    # plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.title("Acc MAE Result")
    x = ['AKF', 'RTS']
    h = [AKF_MAE_acce, RTS_MAE_acce]
    c = ['skyblue', 'orange']
    bars = plt.bar(x, h, color=c, width=0.2)
    for bar in bars:
        yval = bar.get_height()  # 取得 bar 的高度（數值）
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.5f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(x, x)
    plt.yscale('log')
    plt.xlabel('Method')
    plt.ylabel('rad/s^2')
    # plt.legend()
    plt.tight_layout()
    # 存圖
    if save_svg == 1:
        plt.savefig("error_record/AKF與RTS誤差比較.svg", format="svg", dpi=300)
    if save_jpg == 1:
        plt.savefig("error_record/AKF與RTS誤差比較.jpg", format="jpg", dpi=300)
    # plt.close()

# ===================================== Calculating Error -after stabilization =====================================
def calError2(true_vel, true_acc, vele, acce, RTS_vele, RTS_acce):
    #----------------------------------------儲存圖片-------------------------------------------#
    save_jpg = 0 # 0:no save, 1:save
    save_svg = 0 # 0:no save, 1:save
    
    # 計算RMSE -> RMSE_pos = np.sqrt(np.mean((pose - true_pos)**2))
    vele = np.array(vele).flatten()
    acce = np.array(acce).flatten()
    # ================================= AKF ================================
    AKF_RMSE_vele = RMSE(vele, true_vel)
    AKF_RMSE_acce = RMSE(acce, true_acc)
    # ================================= RTS ================================
    RTS_RMSE_vele = RMSE(RTS_vele, true_vel)
    RTS_RMSE_acce = RMSE(RTS_acce, true_acc)

    # print results
    print("--------------- Error Result -after stabilization ---------------")
    print("------------------- RMSE -------------------")
    print("AKF RMSE vel :", AKF_RMSE_vele)
    print("AKF RMSE acc :", AKF_RMSE_acce)
    print("RTS RMSE vel :", RTS_RMSE_vele)
    print("RTS RMSE acc :", RTS_RMSE_acce)
    print("--------------------------------------------")

    # 建立誤差數據
    error_data = {
        "Method": ["AKF", "AKF", "RTS", "RTS"],
        "Type":   ["RMSE_vel", "RMSE_acc", "RMSE_vel", "RMSE_acc"],
        "Value":  [AKF_RMSE_vele, AKF_RMSE_acce, RTS_RMSE_vele, RTS_RMSE_acce]
    }
    # 建立 DataFrame
    df = pd.DataFrame(error_data)
    # 存成 Excel
    # df.to_excel("error_record/RMSE_error_results.xlsx", index=False)

    # 計算MAE -> MAE_pos = np.mean(np.abs(pose - true_pos))
    # ================================= AKF ================================
    AKF_MAE_vele = MAE(vele, true_vel)
    AKF_MAE_acce = MAE(acce, true_acc)
    # ================================= RTS ================================
    RTS_MAE_vele = MAE(RTS_vele, true_vel)
    RTS_MAE_acce = MAE(RTS_acce, true_acc)

    # print results
    print("------------------- MAE -------------------")
    print("AKF MAE vel :", AKF_MAE_vele)
    print("AKF MAE acc :", AKF_MAE_acce)
    print("RTS MAE vel :", RTS_MAE_vele)
    print("RTS MAE acc :", RTS_MAE_acce)
    print("--------------------------------------------")

    # 建立誤差數據
    error_data = {
        "Method": ["AKF", "AKF", "RTS", "RTS"],
        "Type":   ["MAE_vel", "MAE_acc", "MAE_vel", "MAE_acc"],
        "Value":  [AKF_MAE_vele, AKF_MAE_acce, RTS_MAE_vele, RTS_MAE_acce]
    }
    # 建立 DataFrame
    df = pd.DataFrame(error_data)
    # 存成 Excel
    # df.to_excel("error_record/MAE_error_results.xlsx", index=False)

    # ================================== plot RMSE and MAE =====================================
    # plot RMSE
    plt.figure(figsize=(10, 7))
    plt.suptitle("RMSE and MAE Result -with RTS -after stabilization", fontsize=14, fontweight='bold')
    plt.subplot(2, 2, 1)
    plt.title("Vel RMSE Result")
    # x = ['AKF', 'LSF14', 'LSF28', 'TSE2', 'TSE3', 'BDE3', 'CFD', 'KF']
    # h = [AKF_RMSE_vele, LSF_RMSE_vele, LSF28_RMSE_vele, TSE2_RMSE_vele, TSE3_RMSE_vele, BDE3_RMSE_vele, CFD_RMSE_vele, KF_RMSE_vele]
    # c = ['skyblue', 'orange', 'green', 'red', 'purple', 'pink', 'brown', 'blue']
    x = ['AKF', 'RTS']
    h = [AKF_RMSE_vele, RTS_RMSE_vele]
    c = ['skyblue', 'orange']
    bars = plt.bar(x, h, color=c, width=0.2)
    for bar in bars:
        yval = bar.get_height()  # 取得 bar 的高度（數值）
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.5f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(x, x)
    plt.xlabel('Method')
    plt.ylabel('rad/s')
    # plt.legend()

    plt.subplot(2, 2, 2)
    x = np.arange(2)
    width = 0.4
    plt.title("Acc RMSE Result")
    x = ['AKF', 'RTS']
    h = [AKF_RMSE_acce, RTS_RMSE_acce]
    c = ['skyblue', 'orange']
    bars = plt.bar(x, h, color=c, width=0.2)
    for bar in bars:
        yval = bar.get_height()  # 取得 bar 的高度（數值）
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.5f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(x, x)
    plt.yscale('log')
    plt.xlabel('Method')
    plt.ylabel('rad/s^2')
    # plt.legend()

    plt.subplot(2, 2, 3)
    plt.title("Vel MAE Result")
    # x = ['AKF', 'LSF14', 'LSF28', 'TSE2', 'TSE3', 'BDE3', 'CFD', 'KF']
    # h = [AKF_MAE_vele, LSF_MAE_vele, LSF28_MAE_vele, TSE2_MAE_vele, TSE3_MAE_vele, BDE3_MAE_vele, CFD_MAE_vele, KF_MAE_vele]
    # c = ['skyblue', 'orange', 'green', 'red', 'purple', 'pink', 'brown', 'blue']
    x = ['AKF', 'RTS']
    h = [AKF_MAE_vele, RTS_MAE_vele]
    c = ['skyblue', 'orange']
    bars = plt.bar(x, h, color=c, width=0.2)
    for bar in bars:
        yval = bar.get_height()  # 取得 bar 的高度（數值）
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.5f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(x, x)
    plt.xlabel('Method')
    plt.ylabel('rad/s')
    # plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.title("Acc MAE Result")
    x = ['AKF', 'RTS']
    h = [AKF_MAE_acce, RTS_MAE_acce]
    c = ['skyblue', 'orange']
    bars = plt.bar(x, h, color=c, width=0.2)
    for bar in bars:
        yval = bar.get_height()  # 取得 bar 的高度（數值）
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.5f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(x, x)
    plt.yscale('log')
    plt.xlabel('Method')
    plt.ylabel('rad/s^2')
    # plt.legend()
    plt.tight_layout()
    # 存圖
    if save_svg == 1:
        plt.savefig("error_record/AKF與RTS誤差比較_穩定後.svg", format="svg", dpi=300)
    if save_jpg == 1:
        plt.savefig("error_record/AKF與RTS誤差比較_穩定後.jpg", format="jpg", dpi=300)
    # plt.close()