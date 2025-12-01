import numpy as np

class KalmanFilter:
    def __init__(self):

        self.x = np.array([[0], [0], [0], [0], [0], [0]])  # State Vector 

        self.P = np.diag([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])  # Initial Covariance Matrix

        self.u = np.array([[0], [0], [0], [0], [0], [0]])  # External Motion

        self.F = np.array([[1, 0, 0, 1, 0, 0],
                           [0, 1, 0, 0, 1, 0],
                           [0, 0, 1, 0, 0, 1],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1],])
        
        self.H = np.array([[1, 0, 0, 0, 0, 0], 
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0]])
        
        self.R = np.diag([0.5, 0.5, 1.0])

        self.Q = np.diag([0.25, 0.25, 0.25, 1.0, 1.0, 1.0]) * 0.1**2

        self.I = np.identity(6)

        self.undetected_count = 0
        
        self.label = None

    def predict(self):
        self.x = np.dot(self.F, self.x) + self.u
        # Covariance Prediction
        self.P = np.dot(np.dot(self.F, self.P), np.transpose(self.F)) + self.Q
        return self.x[0:3][0]  # return predicted position (x, y, z)

    def update(self, Z):
        self.y = Z - np.dot(self.H, self.x)
        # Covariance
        self.S = np.dot(np.dot(self.H, self.P), np.transpose(self.H)) + self.R
        # Kalman Gain               
        self.K = np.dot(np.dot(self.P, np.transpose(self.H)), np.linalg.pinv(self.S))
        # State Update
        self.x = self.x + np.dot(self.K, self.y)    
        # Covariance Update
        self.P = np.dot((self.I - np.dot(self.K, self.H)),self.P)





    