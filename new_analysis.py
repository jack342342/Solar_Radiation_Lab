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
#%%
filename = 'Total_Clear.csv'
data = pd.read_csv(filename)

time = data['Time'].to_numpy()
solar = np.divide(data['Solarimeter (mV)'].to_numpy(), 0.01498)
light = np.multiply(data['Lightmeter (kLux)'].to_numpy(), 7.9)
day = data['Day'].to_numpy()
#diff = np.divide(data['Diffusive Flux (mV)'].to_numpy(), 0.01498)

#%%
times = []
for i in range(len(time)):
    h, m = time[i].split(':')
    t = datetime.datetime(2020, 1, day[i], int(h), int(m), 0, 0, tzinfo=datetime.timezone.utc)
    times.append(t)
    
#%%

r = 4.498639356712896
factor = 1 - (1/r) 


sza = []
for i in range(len(times)):
    angle = 90 - sol.get_altitude(51.506, -0.167, times[i])
    sza.append(np.radians(angle))

#%%
filename2 = '200119_Clear.csv'
data2 = pd.read_csv(filename2)

time2 = data2['Time'].to_numpy()
solar2 = np.divide(data2['Solarimeter (mV)'].to_numpy(), 0.01498)
light2 = np.multiply(data2['Lightmeter (kLux)'].to_numpy(), 7.9)
diff2 = np.divide(data2['Diffusive Flux (mV)'].to_numpy(), 0.01498)

times2 = []
for i in range(len(time2)):
    h, m = time2[i].split(':')
    t = datetime.datetime(2020, 1, int(filename2[4:6]), int(h), int(m), 0, 0, tzinfo=datetime.timezone.utc)
    times2.append(t)

sza2 = []
for i in range(len(times2)):
    angle = 90 - sol.get_altitude(51.506, -0.167, times2[i])
    sza2.append(np.radians(angle))

r = solar2 / diff2

r_mean = np.mean(r)
r_err = np.std(r)

factor = 1 - 1 / r_mean
#%%

def intensity_curve(x, I, u):
    return I*np.exp(-u*(1/(np.cos(x))))
p0 = (500, 0.35)
light_err = r_err * light / r_mean ** 2 + (np.array(sza) - 1.29) * 40
solar_err = r_err * solar / r_mean ** 2 + (np.array(sza) - 1.29) * 40
light2 = light * factor
solar2 = solar * factor

params2, cov2 = curve_fit(intensity_curve, sza, solar2, p0 = p0, sigma = solar_err, absolute_sigma = True)
angles = np.linspace(np.min(sza), np.max(sza), 200)
#plt.plot(sza, intensity_curve(np.array(sza), , 0.343), color='blue')
plt.plot(np.array(angles) * 180/np.pi, intensity_curve(np.array(angles), params2[0], params2[1]), color='crimson', label='Intensity Fit')
plt.errorbar(np.array(sza) * 180/np.pi, solar2, yerr = solar_err, fmt = 'o', ms = 1, color='black', label='Data')

#plt.plot(sza, trc, 'x')
plt.xlabel('Solar Zenith Angle ($^\circ$)', fontsize=15)
plt.ylabel('Direct Solar Flux ($W/m^{2}$)', fontsize=15)
plt.legend(fontsize='large')
plt.grid()
plt.show()
    
    
    
#%%
def intensity_curve(x, I, u):
    return I*np.exp(-u*(1/(np.cos(x))))
p0 = (500, 0.35)
#trc = np.divide(solar, intensity_curve(np.array(sza), 818, 0.343))
light_ = light*factor
#solar_ = solar*factor
params2, cov2 = curve_fit(intensity_curve, sza, solar, p0 = p0)
angles = np.linspace(np.min(sza), np.max(sza), 200)
#plt.plot(sza, intensity_curve(np.array(sza), , 0.343), color='blue')
plt.plot(angles, intensity_curve(np.array(angles), params2[0], params2[1]), color='forestgreen', label='Intensity Fit')
plt.plot(sza, solar, 'x', color='midnightblue', label='Data')
#plt.plot(sza, trc, 'x')
plt.xlabel('Solar Zenith Angle (Radians)', fontsize=13)
plt.ylabel('Direct Solar Flux ($W/m^{2}$)', fontsize=13)
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
    #def f(tau):
        #return trc[i] - ((a1 + b1 * np.cos(sza[i])) / (1 + (c - d * alpha) * tau))
    #optical_depths[i] = minimize(f, 10)['x']
    optical_depths[i] = ((a1 + b1 + np.cos(sza[i])) / trc[i] - 1) / (c - d * alpha)
    
#%%
n0 = 103000/(1.38064852e-23*280)
sigma = params2[1]/(n0*7000)
#%%
import spherical_0 as s

params = {
   'axes.labelsize': 20,
   'font.size': 16,
   'xtick.labelsize': 15,
   'ytick.labelsize': 15,
   'figure.figsize': [8, 8/1.618]
   }
plt.rcParams.update(params)

def adjusted_fit(f, sigma, I0):
    exp = I0 * np.exp(-sigma * f)
    return exp

def intensity_curve(x, I, u):
    return I*np.exp(-u*(1/(np.cos(x))))

p0 = (500, 0.35)
params2, cov2 = curve_fit(intensity_curve, sza, solar2, p0 = p0, sigma=solar_err, absolute_sigma = True)
angles = np.linspace(np.min(sza), np.max(sza), 200)
f = np.zeros(len(sza))
for i in range(0, len(sza)):
    f[i] = s.curved_path_integrator(10, 1e5, 1e-40, 0.5, sza[i], 6.6e-4) # tolerance in Z matches mean error from pysolar
    print(f'Point {i + 1} complete.')

sort_indices = np.array(sza).argsort()
sza_sort = np.array(sza)[sort_indices]
f_sort = f[sort_indices]

params3, cov3 = curve_fit(adjusted_fit, f, solar2, p0 = [1e-29, 1], sigma = solar_err, absolute_sigma = True)
plt.errorbar(np.array(sza) * 180/np.pi, solar2, yerr = solar_err, fmt = 'o', ms = 1, color='black', label='Data')
plt.plot(np.array(sza_sort) * 180 / np.pi, adjusted_fit(f_sort, params3[0], params3[1]), color='cornflowerblue', ls='-', label = 'Improved Fit')
plt.plot(angles * 180 / np.pi, intensity_curve(np.array(angles), params2[0], params2[1]), color='forestgreen', ls='--', label='Initial Fit')
plt.xlabel('Solar Zenith Angle ($^\circ$)')
plt.ylabel('Solar Irradiance ($W/m^2$)')
plt.grid()
plt.legend()
plt.show()




