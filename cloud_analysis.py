"""
Author: Jack Carlin
Task: Data Analysis for Solar Radiation Lab
"""
#%%
import pysolar.solar as sol
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import datetime
from scipy.optimize import curve_fit
import numpy.ma as ma
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import numpy.MaskedArray as ma
#%%
filename = '200124_Overcast.csv'
data = pd.read_csv(filename)

time = data['Time'].to_numpy()
solar = np.divide(data['Solarimeter (mV)'].to_numpy(), 0.01498)
light = np.multiply(data['Lightmeter (kLux)'].to_numpy(), 7.9)
#%%
times = []
for i in range(len(time)):
    h, m = time[i].split(':')
    t = datetime.datetime(2020, int(filename[2:4]), int(filename[4:6]), int(h), int(m), 0, 0, tzinfo=datetime.timezone.utc)
    times.append(t)
    
#%%
sza = []
for i in range(len(times)):
    angle = 90 - sol.get_altitude(51.506, -0.167, times[i])
    sza.append(np.radians(angle))

def intensity_curve(x, I, u):
    return I*np.exp(-u*(1/(np.cos(x))))

p0 = (500, 0.35)
trc = np.divide(solar, intensity_curve(np.array(sza), 818, 0.343))
#trc = np.delete(trc, 15)
#sza = np.delete(sza, 15)
#params, cov = curve_fit(intensity_curve, sza, solar, p0 = p0)
angles = np.linspace(np.min(sza), np.max(sza), 200)
spl = CubicSpline(np.sort(sza), np.flip(trc))
#plt.plot(angles, intensity_curve(np.array(angles), params[0], params[1]), color='forestgreen', label='Intensity Fit')
#plt.plot(sza, light, 'x', color='midnightblue', label='Data')
plt.plot(angles, spl(angles), color='blue')
plt.plot(sza, trc, 'x', color='k')
plt.xlabel('Solar Zenith Angle (Radians)', fontsize=13)
plt.ylabel('Transmissivity', fontsize=13)
plt.legend(fontsize='large')
plt.show()
#%%
a1 = 0.58
b1 = 0.74
b2 = -0.1612
b3 = -0.8343
k1 = 1.9785
k2 = 0.2828
k3 = 2.3042
alpha = 0.15
c = 0.1365
d = 0.1291
def a(tau):
    return a1 + (1 - a1)*np.exp(-k1*tau)
def b(tau):
    return b1*(1 + b2*np.exp(-k2*tau) + b3*np.exp(-k3*tau))
optical_depths = np.zeros(len(trc))
for i in range(len(trc)):
    optical_depths[i] = ((a1 + b1 + np.cos(sza[i])) / trc[i] - 1) / (c - d * alpha)

angles = np.linspace(np.min(sza), np.max(sza), 200)
spl = CubicSpline(np.sort(sza), np.flip(optical_depths))
plt.plot(angles, spl(angles), color='gold')
plt.plot(sza, optical_depths, 'x', color='crimson')
plt.xlabel('Solar Zenith Angle (Radians)', fontsize=13)
plt.ylabel('Optical Depth', fontsize=13)




    
    
    
