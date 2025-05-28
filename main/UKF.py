import numpy as np

def UKF(dt, pos, SIGMA):
    def generate_sigma_points(x, P, alpha=1e-3, beta=2, kappa=0):
        n = len(x)  # State dimension
        lambda_ = alpha**2 * (n + kappa) - n
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = x.T
        
        sqrt_P = np.linalg.cholesky((n + lambda_) * P)
        for i in range(n):
            sigma_points[i + 1] = x.T + sqrt_P[i]
            sigma_points[i + n + 1] = x.T - sqrt_P[i]
        
        return sigma_points, lambda_
    
    def predict_sigma_points(sigma_points, A, B, u, Wt=0):
        # 初始化預測的Sigma點矩陣，形狀與輸入相同
        predicted_sigma_points = np.zeros_like(sigma_points)
        for i in range(sigma_points.shape[0]):
            sigma_point = sigma_points[i].reshape(-1) 
            predicted_point = np.dot(A, sigma_point) + np.dot(B, u).flatten() + Wt
            predicted_sigma_points[i] = predicted_point.flatten()
        return predicted_sigma_points

    def unscented_transform(sigma_points, weights, Q):
        x_pred = np.dot(weights, sigma_points)  # Predicted state mean
        P_pred = Q.copy()
        for i in range(sigma_points.shape[0]):
            diff = sigma_points[i] - x_pred
            P_pred += weights[i] * np.outer(diff, diff)
        return x_pred, P_pred

    def update_step(x_pred, P_pred, sigma_points, C, R, pos):
        z_pred = np.dot(C, x_pred)
        P_zz = np.dot(C, np.dot(P_pred, C.T)) + R
        P_xz = np.zeros((x_pred.shape[0], z_pred.shape[0]))
        for i in range(sigma_points.shape[0]):
            diff_x = sigma_points[i] - x_pred
            diff_z = np.dot(C, sigma_points[i]) - z_pred
            P_xz += np.outer(diff_x, diff_z)
        K = np.dot(P_xz, np.linalg.inv(P_zz))
        y = pos - z_pred
        x_updated = x_pred + np.dot(K, y)
        P_updated = P_pred - np.dot(K, np.dot(P_zz, K.T))
        return x_updated, P_updated

    # Initialize variables
    n = 3  # State dimension
    alpha = 0.1 #　1e-3
    beta = 1.2
    kappa = 0
    lambda_ = alpha**2 * (n + kappa) - n
    weights = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
    weights[0] = lambda_ / (n + lambda_)

    P = np.array([[1e-4, 0, 0],
                  [0, 1e-4, 0],
                  [0, 0, 1e-4]])  # Initial covariance
    # P = np.array([[1e-4, 1e-4, 1e-4],
    #                 [1e-4, 1e-4, 1e-4],
    #                 [1e-4, 1e-4, 1e-4]])

    C = np.array([[1, 0, 0]])  # Measurement model matrix

    # Q = SIGMA * np.array([[dt**4/4, dt**3/2, dt**2/2],
    #                       [dt**3/2, dt**2, dt],
    #                       [dt**2/2, dt, 1]])  # 系統過程噪聲
    Q = np.array([[1.64521896e-06, 1.35746221e-05, 1.00125342e-04],
                [1.35746221e-05, 1.17514489e-04, 2.37265766e-03],
                [1.00125342e-04, 2.37265766e-03, 1.02581682e+00]]) # scara
    Q = np.array([[1.25124714e+01, 1.27516429e+00, 1.23241483e+02],
                    [1.27516429e+00, 1.74838225e+03, 8.02527454e+05],
                    [1.23241483e+02, 8.02527454e+05, 5.35272247e+08]]) # sin
    # Q = np.array([[0.01, 1.27516429e+00, 1.23241483e+02],
    #                 [1.27516429e+00, 0.01, 8.02527454e+05],
    #                 [1.23241483e+02, 8.02527454e+05, 0.01]])
    # Q = np.array ([[3.47644979e+00,1.27516429e+00,1.23241483e+02],
    #                [1.27516429e+00,6.37682644e+02,8.02527454e+05],
    #                [1.23241483e+02,8.02527454e+05,7.83710376e+05]]) # AKF min error Q
    # Q = np.array([[0.1, 0, 0],
    #           [0, 0.01, 0],
    #           [0, 0, 0.001]])

    R = 0.01  # Measurement noise covariance (adjust as needed)

    # Initial state estimate
    x = np.zeros(3)

    pose = np.zeros(len(pos))
    vele = np.zeros(len(pos))
    acce = np.zeros(len(pos))

    for i in range(len(pos)):
        # Generate sigma points
        sigma_points, lambda_ = generate_sigma_points(x, P)
        
        # Propagate sigma points through the process model (prediction step)
        sigma_points_pred = predict_sigma_points(sigma_points, A=np.array([[1, dt, 0.5 * dt**2],
                                                                          [0, 1, dt],
                                                                          [0, 0, 1]]),
                                                 B=np.array([[0.5 * dt**2],
                                                             [dt],
                                                             [1]]),
                                                 u=np.zeros((1, 1)))  # Assuming zero control input
        
        # Unscented Transform for predicted state and covariance
        x_pred, P_pred = unscented_transform(sigma_points_pred, weights, Q)
        
        # Measurement update (correction step)
        x_updated, P_updated = update_step(x_pred, P_pred, sigma_points_pred, C, R, pos[i])
        
        # Store the updated state
        x = x_updated
        P = P_updated
        
        pose[i] = x[0]
        vele[i] = x[1]
        acce[i] = x[2]

    return pose, vele, acce