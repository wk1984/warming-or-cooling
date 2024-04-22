#!/usr/bin/env python3
import os
import numpy as np
import h5py

def repeat_cycle(met, d):
    Svar = []
    for i in range(d):
        Svar = np.append(Svar, met[0:365])
        # print(i)
    return Svar

basepath = os.path.abspath(os.path.join(os.getcwd(), os.pardir,os.pardir))

t_avg = -8.8
t_amp = 30
phase_shift = 260
doy = np.arange(1,366)

T = t_avg + (t_amp/2) * np.sin(2*np.pi*(doy+phase_shift)/365)
g_avg = 86.8
g_amp = 271-50
G = g_avg+30 + (g_amp/2) * np.sin(2*np.pi*(doy+phase_shift)/365)

ws = np.full((365),4)
rh = np.full((365),0.8)

rain_sum = 300
snow_sum = 50

rain = np.where(T>0, rain_sum/np.sum(T>0),0)
snow = np.where(T<=0, snow_sum/np.sum(T<=0),0)

swin = repeat_cycle(G, 200)
t = repeat_cycle(T, 200)
rr = repeat_cycle(rain, 200)
sn = repeat_cycle(snow, 200)
rh = repeat_cycle(rh, 200)
ws = repeat_cycle(ws, 200)

#%%
tim = np.arange(0.0, 86400 * 365 * 200, 86400)
with h5py.File(basepath + '/data/cw_spinup.h5', 'w') as hf:
    hf.create_dataset('time [s]', data=tim)
    hf.create_dataset('air temperature [K]', data=t+273.15)
    hf.create_dataset('incoming shortwave radiation [W m^-2]', data=swin)
    hf.create_dataset('precipitation rain [m s^-1]', data=rr/86400000)
    hf.create_dataset('precipitation snow [m SWE s^-1]', data=sn/86400000)
    hf.create_dataset('relative humidity [-]', data=rh)
    hf.create_dataset('wind speed [m s^-1]', data=ws)
hf.close()
#%%
years = 22
rain_i = rain.copy()
rain_i_50more = rain.copy()
rain_i_50less = rain.copy()
p_old = rain_i[[166, 196, 227]]
p_new = np.array([32,45,50])
rain_i[[166, 196, 227]] = p_new
# rain_i_50more[[166, 196, 227]] = p_new*1.5
# rain_i_50less[[166, 196, 227]] = p_new*0.5
rr_c = repeat_cycle(rain, years)

swin = repeat_cycle(G, years)
t = repeat_cycle(T, years)
rr = repeat_cycle(rain, 10)
rr_i = np.concatenate((rr, rain_i, rr,rain))
# rr_i_50more = np.concatenate((rr, rain_i_50more, rr, rain))
# rr_i_50less = np.concatenate((rr, rain_i_50less, rr, rain))
# rr_i_multi = np.concatenate((rr, rain_i, rain_i, rr))


sn = repeat_cycle(snow, years)
rh = repeat_cycle(rh, years)
ws = repeat_cycle(ws, years)

print("p old = " + str(p_old))
print("p new = " + str(p_new))
# rr8 = repeat_cycle(rain,8)
# rr_i_multi3 = np.concatenate((rr8, rain_i, rain_i, rain_i, rr, rain))
# rr_i_multi4 = np.concatenate((rr8, rain_i, rain_i, rain_i, rain_i, rr))

#%%
tim = np.arange(0.0, 86400 * 365 * 22, 86400)
with h5py.File(basepath + '/data/cw_irrigation.h5', 'w') as hf:
    hf.create_dataset('time [s]', data=tim)
    hf.create_dataset('air temperature [K]', data=t+273.15)
    hf.create_dataset('incoming shortwave radiation [W m^-2]', data=swin)
    hf.create_dataset('precipitation rain [m s^-1]', data=rr_i/86400000)
    hf.create_dataset('precipitation snow [m SWE s^-1]', data=sn/86400000)
    hf.create_dataset('relative humidity [-]', data=rh)
    hf.create_dataset('wind speed [m s^-1]', data=ws)
hf.close()

with h5py.File(basepath + '/data/cw_control.h5', 'w') as hf:
    hf.create_dataset('time [s]', data=tim)
    hf.create_dataset('air temperature [K]', data=t+273.15)
    hf.create_dataset('incoming shortwave radiation [W m^-2]', data=swin)
    hf.create_dataset('precipitation rain [m s^-1]', data=rr_c/86400000)
    hf.create_dataset('precipitation snow [m SWE s^-1]', data=sn/86400000)
    hf.create_dataset('relative humidity [-]', data=rh)
    hf.create_dataset('wind speed [m s^-1]', data=ws)
hf.close()