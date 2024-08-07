#====================================================================
# Function is used to specify the matrix A for evaluating the specific
# Heat capacity of the mix ture
#=====================================================================
def matrixB_coefs():

    matrixB_coef  = np.array([[3.69, -0.3838, 964., 186E4, 194.7],           #CO2  [0]
                     [0.002653, 0.7452, 12., 0.0, 20.4],            #H2   [1]
                     [0.00059882, 0.6863, 57.13, 501.92, 81.7],     #CO   [2]
                     [6.2041E-6, 1.3973, 1.3973, 0.0, 373.2],       #H2O  [3]
                     [8.3983E-6, 1.4268, -49.654, 0.0, 111.7],      #CH4  [4]
                     [0.000073869, 1.1689, 500.73, 0.0, 184.5],     #C2H6 [5]
                     [-1.12, 0.10972, -9834.6, -7535800., 231.1],   #C3H8 [6] 
                     [0.051094, 0.45253, 5455.5, 1979800., 272.7],  #C4H10[7] 
                     [8.6806E-6, 1.4559, 299.72, -29403., 169.4],   #CH2  [8]
                     [8.6806E-6, 1.4559, 299.72, -29403., 169.4],   #C2H4 [9]
                     [0.0000449, 1.2018, 421., 0.0, 225.4],         #C3H6 [10]
                     [0.000096809, 1.1153, 781.82, 0.0, 266.9]])     #C4H8 [11]

    return matrixB_coef
