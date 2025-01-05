import numpy as np

class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_dt_meas, y_dt_meas):
        self.dt = dt

        self.u = np.matrix([[u_x],[u_y]])

        self.xk = np.matrix([[0], [0], [0], [0]])

        self.matA = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        self.matB = np.matrix([[(self.dt**2)/2, 0],
                            [0,(self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])

        self.matH = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        self.matQ = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2

        self.matR = np.matrix([[x_dt_meas**2,0],
                           [0, y_dt_meas**2]])

        self.matP = np.eye(self.matA.shape[1])
        
                
    def predict(self):
        self.xk = np.dot(self.matA, self.xk) + np.dot(self.matB, self.u)
        self.matP = np.dot(np.dot(self.matA, self.matP), self.matA.T) + self.matQ
        return self.xk[0:2]
    
    
    def update(self, z):

        # S = H*P*H'+R
        S = np.dot(self.matH, np.dot(self.matP, self.matH.T)) + self.matR

        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.matP, self.matH.T), np.linalg.inv(S))  #Eq.(11)

        self.xk = np.round(self.xk + np.dot(K, (z - np.dot(self.matH, self.xk))))   #Eq.(12)

        I = np.eye(self.matH.shape[1])

        self.matP = (I - np.dot(K,self.matH)) * self.matP   #Eq.(13)
        return self.xk[0:2]

