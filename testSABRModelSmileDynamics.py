import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))

import numpy as np
import matplotlib.pyplot as plt
import pandas

from QuantLibWrapper import SabrModel, MCSimulation

# SabrModel( S(t), T, alpha, beta, nu, rho )
model1 = SabrModel(0.05,5.0,0.0420, 0.1000,0.5000,0.3 )
model2 = SabrModel(0.05,5.0,0.0420, 0.7000,0.5000,0.0 )
model3 = SabrModel(0.05,5.0,0.0420, 0.9000,0.0001,0.0 )
# ATM calibration
print(model1.calibrateATM(0.01), model2.calibrateATM(0.01), model3.calibrateATM(0.01))
# Strikes
strikes = [ (i+1)/1000 for i in range(100) ]
# implied volatility
vols1 = [model1.normalVolatility(strike) for strike in strikes]
vols2 = [model2.normalVolatility(strike) for strike in strikes]
vols3 = [model3.normalVolatility(strike) for strike in strikes]

# smile dynamics
S_ = [ 0.020, 0.035, 0.050, 0.065, 0.080 ]
vols1_ = []
vols2_ = []
vols3_ = []
backBone1_ = []
backBone2_ = []
backBone3_ = []
for S in S_:
    model1.forward = S
    model2.forward = S
    model3.forward = S
    vols1_.append([model1.normalVolatility(strike) for strike in strikes])
    vols2_.append([model2.normalVolatility(strike) for strike in strikes])
    vols3_.append([model3.normalVolatility(strike) for strike in strikes])
    backBone1_.append(model1.normalVolatility(S))
    backBone2_.append(model2.normalVolatility(S))
    backBone3_.append(model3.normalVolatility(S))

plt.figure()
plt.plot(strikes,vols1, 'b-', label='beta=0.1,nu=0.5,rho=0.3')
plt.plot(strikes,vols2, 'r-', label='beta=0.7,nu=0.5,rho=0.0')
plt.plot(strikes,vols3, 'g-', label='beta=0.9,nu=0.0,rho=0.0')
plt.legend()
plt.xlabel('Strike')
plt.ylabel('Normal Volatility')
plt.xlim((0.0, 0.10))
plt.ylim((0.005, 0.020))

plt.figure()
for k in range(len(S_)):
    plt.plot(strikes,vols1_[k], 'b:', label='S='+str(S_[k]))
plt.plot(S_,backBone1_, 'bo-')
plt.legend()
plt.xlabel('Strike')
plt.ylabel('Normal Volatility')
plt.xlim((0.0, 0.10))
plt.ylim((0.005, 0.020))

plt.figure()
for k in range(len(S_)):
    plt.plot(strikes,vols2_[k], 'r:', label='S='+str(S_[k]))
plt.plot(S_,backBone2_, 'ro-')
plt.legend()
plt.xlabel('Strike')
plt.ylabel('Normal Volatility')
plt.xlim((0.0, 0.10))
plt.ylim((0.005, 0.020))

plt.figure()
for k in range(len(S_)):
    plt.plot(strikes,vols3_[k], 'g:', label='S='+str(S_[k]))
plt.plot(S_,backBone3_, 'go-')
plt.legend()
plt.xlabel('Strike')
plt.ylabel('Normal Volatility')
plt.xlim((0.0, 0.10))
plt.ylim((0.003, 0.020))

plt.show()

