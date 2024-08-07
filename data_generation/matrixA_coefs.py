#====================================================================
# Function is used to specify the matrix A for evaluating the specific
# Heat capacity of the mix ture
#=====================================================================
def matrixA_coefs():

##    matrixA_coef = [[5.457, 1.045e-3, 0.0, -1.1570e-5],    #CO2  [0]
##                     [3.249, 0.422e-3, 0.0,  0.0830e-5],    #H2   [1]
##                     [3.376, 0.557e-3, 0.0, -0.0310e-5],    #CO   [2]
##                     [3.470, 1.450e-3, 0.0,  0.2100e-5],    #H2O  [3]
##                     [1.702, 9.081e-3, -2.164, 0.0],        #CH4  [4]
##                     [1.131, 19.225e-3, -5.561, 0.0],       #C2H6 [5]
##                     [1.213, 28.875e-3, -8.824, 0.0],       #C3H8 [6] 
##                     [1.935, 36.915e-3, -11.402, 0.0],      #C4H10[7] 
##                     [1.702, 9.081e-3, -2.164, 0.0],        #CH2  [8]
##                     [1.424, 14.394e-3, -4.392, 0.0],       #C2H4 [9]
##                     [1.637, 22.706e-3, -6.915, 0.0],       #C3H6 [10]
##                     [1.967, 31.630e-3, -9.873, 0.0]]       #C4H8 [11]
    
    matrixA_coef = [[24.99735, 55.18696, -33.69137, 7.948387, -0.136638],                                #CO2
                     [33.066178, -11.363417, 11.432816, -2.772874, -0.158558],                            #H2
                     [25.56759, 6.096130, 4.054656, -2.671301, 0.131021],                                 #CO
                     [-203.6060, 1523.290, -3196.413, 2474.455, 3.855326],                                #H2O
                     [-0.703029, 108.4773, -42.52157, 5.862788, 0.678565],                                #CH4
                     [6.08160924e+00,  1.73582462e+02, -6.69190557e+01,  9.08912042, 1.29136464e-01],     #C2H6 
                     [1.26607748e+01,  2.32070434e+02, -7.03448841e+01,  6.07253269e-01, 2.69572460e-02], #C3H8   
                     [2.21581184e+01,  2.92589342e+02, -8.58164045e+01, -1.01398492e+00, 6.34967683e-03], #C4H10   
                     [31.96823, 6.783603, 12.51890, -5.696265, -0.031115],                                #CH2 
                     [-6.387880, 184.4019, -112.9718, 28.49593, 0.315540],                                #C2H4//
                     [1.39948834e+01,  1.97060433e+02, -7.80916341e+01,  1.08145463e+01, 2.85826563e-02], #C3H6
                     [1.45857434e+01,  2.77858536e+02, -1.13097635e+02,  1.60063005e+01, 3.38738504e-02]] #C4H8
                     

    return matrixA_coef
