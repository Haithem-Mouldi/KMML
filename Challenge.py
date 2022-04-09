import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from sklearn.metrics import accuracy_score
import time 
from scipy import optimize
from sklearn import svm 


ppc = 8
orientations = 9
cells_per_block = 2
C = 10.


############# Data Importation:
abs_path = os.getcwd()
path_to_data = os.path.join(abs_path,'KaggleData')
print(path_to_data)

## Preparing the data:
df_train = pd.read_csv(path_to_data+'/Xtr.csv', header = None)
df_test = pd.read_csv(path_to_data+'/Xte.csv', header = None)
df_labels = pd.read_csv(path_to_data+'/Ytr.csv')

y = df_labels['Prediction'].to_numpy()
X_train = df_train.iloc[:,:-1].to_numpy(dtype = float)
X_test = df_test.iloc[:,:-1].to_numpy(dtype = float)

X_train_splitted = np.array(np.hsplit(X_train,3))

RGB = np.dstack((X_train_splitted[0],X_train_splitted[1],X_train_splitted[2]))
data_RGB = [np.reshape(RGB[i],(32,32,3)) for i in range(RGB.shape[0])]
data_GRAY = [color.rgb2gray(data_RGB[i]) for i in range(len(data_RGB))]
############



#### HOG IMPLEMENTATION: 
def gradient_magnitude(GradientX, GradientY):
    return np.linalg.norm(np.array([GradientX,GradientY]))

def gradient_direction(GradientX, GradientY):
    eps = 1e-7
    grad_direction = np.arctan(GradientY/(GradientX+eps))
    grad_direction = np.rad2deg(grad_direction) % 180
    return grad_direction

def Calculate_gradient(image):
    horizontal_mask = np.array([-1,0,1])
    vertical_mask = np.array([1,0,-1])
    n,m = image.shape
    new_image = np.zeros((n+2,m+2))
    new_image[1:n+1,1:m+1] = image
    gradient_matrix_x = np.zeros((n,m))
    gradient_matrix_y = np.zeros((n,m))
    gradient_magnitudes = np.zeros((n,m))
    gradient_directions = np.zeros((n,m))
    for i in range(1,n+1):
      for j in range(1,n+1):
        gradient_matrix_x[i-1,j-1] = (new_image[i,j-1:j+2] * horizontal_mask).sum()
        gradient_matrix_y[i-1,j-1] = (new_image[i-1:i+2,j] * vertical_mask).sum()
        gradient_magnitudes[i-1,j-1] = gradient_magnitude(gradient_matrix_x[i-1,j-1],gradient_matrix_y[i-1,j-1])
        gradient_directions[i-1,j-1] = gradient_direction(gradient_matrix_x[i-1,j-1],gradient_matrix_y[i-1,j-1])
    return gradient_matrix_x, gradient_matrix_y, gradient_magnitudes, gradient_directions

def HOG_cell_histogram(cell_direction, cell_magnitude, orientations):
    hist_bins = np.arange(0,180,int(180/orientations))
    HOG_cell_hist = np.zeros(orientations)
    cell_size = cell_direction.shape[0]
    
    for i in range(cell_size):
        for j in range(cell_size):
            curr_direction = cell_direction[i, j]
            curr_magnitude = cell_magnitude[i, j]

            hist_bins_bis = np.hstack((hist_bins,180))
            idx = np.where(hist_bins_bis <= curr_direction)[0][-1] +1

            HOG_cell_hist[idx%orientations - 1] += (orientations/180)*(hist_bins_bis[idx] - curr_direction)*curr_magnitude
            HOG_cell_hist[idx%orientations] += (orientations/180)*(curr_direction - hist_bins_bis[idx -1])*curr_magnitude
    return HOG_cell_hist

def HOG(image,pixels_per_cell, cells_per_block, orientations):
    gradient_x, gradient_y, gradient_magnitudes, gradient_directions = Calculate_gradient(image)
    n, m = image.shape
    n_hblocks = int(m/pixels_per_cell) -1
    n_vblocks = int(n/pixels_per_cell) - 1

    HOGS = []
    for i in range(n_vblocks):
      for j in range(n_hblocks):
        non_normalized_hists = []
        for ki in range(cells_per_block):
          for kj in range(cells_per_block):
              start_i = (i+ki)*pixels_per_cell
              end_i = (i+ki+1)*pixels_per_cell
              start_j = (j+kj)*pixels_per_cell
              end_j = (j+kj+1)*pixels_per_cell
              
              cell_directions = gradient_directions[start_i:end_i,start_j:end_j]
              cell_magnitudes = gradient_magnitudes[start_i:end_i,start_j:end_j]

              HOG_block = HOG_cell_histogram(cell_directions,cell_magnitudes,orientations)
              non_normalized_hists.extend(HOG_block)
        non_normalized_hists = np.array(non_normalized_hists)
        HOGS.extend(non_normalized_hists/np.linalg.norm(non_normalized_hists))
        
    return np.array(HOGS)
###############


############### Kernel SVM Implementation:

###### Kernels Tested:
def linear_kernel(x1,x2):
  return x1.dot(x2)

def chi2_kernel(x1,x2,gamma = .05):
   p = x1.shape[0]
   res = 0
   for k in range(p):
     denom = x1[k] - x2[k]
     nom = x1[k] + x2[k]
     if nom != 0:
         res += denom**2 / nom 
   
   return np.exp(-gamma * res)

def rbf_kernel(x1,x2, gamma = .2):
    # gamma = 1. / x1.shape[0]
    res = np.linalg.norm(x1-x2)**2
    return np.exp(-gamma* res)

def laplacian_kernel(x1,x2,gamma = .04):
  return np.exp(-gamma* np.sum(np.abs(x1-x2)))

def Hellinger_kernel(x1,x2):
  return np.sum(np.sqrt(x1*x2))

###### KernelSVM 

class HomeMadeKernelSvm:
    
    def __init__(self, C, kernel):
        self.C = C                               
        self.kernel = kernel          
        self.alpha = None
        self.supVecs = None
    

    def fit(self, X,y):
        # self._X, self._y = X, y
        self.labels = np.unique(y)
        self.n_labels = len(self.labels)
        # self._K = self.kernel(X, X)
        
        # OneVsAll
        BinaryModels = {}
        finish = len(self.labels)
        for idx, label in enumerate(self.labels):
            BinaryModels[label] = {}
            y_binarized = np.array([1. if yi == label else -1. for yi in y])
            supVecs, supAlphaY, supY = self._fit(X, y_binarized)
            BinaryModels[label]['y'] = y_binarized
            BinaryModels[label]['supAlphaY'] = supAlphaY
            BinaryModels[label]['supVecs'] = supVecs
            BinaryModels[label]['supY'] = supY

        self.BinaryModels = BinaryModels

    def predict(self, X):
        return np.argmax(np.array([self._predict(X, self.BinaryModels[label]['supVecs'], 
                                              	    self.BinaryModels[label]['supAlphaY'],
                                               		self.BinaryModels[label]['supY']) for label in self.labels]),
                                               	    axis = 0)

    def _fit(self, X, y):
        '''
        fit for binary classification, will be used as a setting
        stone for one vs all classification
        '''
        n, p = X.shape

        K = np.apply_along_axis(lambda x1 : np.apply_along_axis(lambda x2:  self.kernel(x1, x2), 1, X),
                                  1, X)   
        yp = y.reshape(-1, 1)
        GramKy = K * np.matmul(yp, yp.T) 

        def Lagrangian(G, alpha):
            return  0.5 * alpha.dot(alpha.dot(G)) - alpha.sum() 

        def LagrangianDerivative(G, alpha):
            return alpha.dot(G) -  np.ones_like(alpha)  

        A = np.vstack((-np.eye(n), np.eye(n)))           
        b = np.hstack((np.zeros(n), self.C * np.ones(n))) 
        constraints = ({'type': 'eq',   'fun': lambda alpha : np.dot(alpha, y),     'jac': lambda alpha: y},
                       {'type': 'ineq', 'fun': lambda alpha : b - np.dot(A, alpha), 'jac': lambda alpha: -A})

        
        result = optimize.minimize(fun=lambda alpha: Lagrangian(GramKy, alpha),
                                   x0=np.ones(n), 
                                   method='SLSQP', 
                                   jac=lambda alpha: LagrangianDerivative(GramKy, alpha), 
                                   constraints=constraints)
        alpha = result.x
        epsilon = 1e-6
        supInds = alpha > epsilon
        supVecs = X[supInds]
        supAlphaY = y[supInds] * alpha[supInds]
        return supVecs, supAlphaY, y[supInds]
    
    def _predict(self, X, supVecs, supAlphaY, supY):
        """ Predict y values in {-1, 1} """
        def predict_(x):
            x1 = np.apply_along_axis(lambda s: self.kernel(s, x), 1, supVecs)
            x2 = x1 * supAlphaY
            return np.sum(x2)

        d_supp = np.apply_along_axis(predict_, 1, supVecs)
        Bias = np.mean(supY - d_supp)
        d = np.apply_along_axis(predict_, 1, X)
        return d + Bias

######### Data augmentation, Horizontal flips


def HorizontalFlip(image):
    '''
    flips horizontally gray images
    '''
    flipped_image = image.copy()
    n,m = image.shape
    for j in range(m):
            flipped_image[:, j] = image[:, m-j-1]
    return flipped_image


data_GRAY_flipped = [HorizontalFlip(data_GRAY[i]) for i in range(len(data_GRAY))]
data_GRAY.extend(data_GRAY_flipped)
augmented_data = np.array(data_GRAY)
y_augmented = np.hstack((y,y))
#########


tmp = time.time()
print('Starting the HOG on training images')
hog_features = []
for  i, image in enumerate(augmented_data):
    fd = HOG(image,  pixels_per_cell= ppc, cells_per_block= cells_per_block, orientations= orientations)
    hog_features.append(fd)
print('HOG time ,', time.time()- tmp)

hog_features = np.array(hog_features)
clf = HomeMadeKernelSvm(C = 10., kernel= rbf_kernel)
print('Starting training')
clf.fit(hog_features,y_augmented)


########## Creating the submission file

X_test_splitted = np.array(np.hsplit(X_test,3))
RGB_test = np.dstack((X_test_splitted[0],X_test_splitted[1],X_test_splitted[2]))
testdata_RGB = [np.reshape(RGB_test[i],(32,32,3)) for i in range(RGB_test.shape[0])]
testdata_GRAY = [color.rgb2gray(testdata_RGB[i]) for i in range(len(testdata_RGB))]

print('Starting the HOG on test data')
hog_features_test = []
for  i, image in enumerate(testdata_GRAY):
    fd = HOG(image,  pixels_per_cell= ppc, cells_per_block= cells_per_block, orientations= orientations)
    hog_features_test.append(fd)

hog_features_test = np.array(hog_features_test)

print('Starting the prediction on test data')
y_pred_test = clf.predict(hog_features_test)

Ids = np.arange(1,hog_features_test.shape[0]+1)
df_submission = pd.DataFrame.from_dict({'Id':Ids,'Prediction':y_pred_test}).reset_index(drop = True)
pd.DataFrame.to_csv(df_submission, os.path.join(path_to_data,'submission.csv'),index = False)

print('Submission File created!')