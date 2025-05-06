import numpy as np

def saha_eqn(Ne, Pe, Te):
    
    Xe_H21 = 13.6
    Xe_He21 = 24.587
    Xe_He32 = 54.416
    rHe = 0.1
    
    lg_w_H21 = np.log10(1.0) + 2.5*np.log10(Te) - 5040*Xe_H21/Te - np.log10(Pe) - 0.48
    lg_w_He21 = np.log10(4.0) + 2.5*np.log10(Te) - 5040*Xe_He21/Te - np.log10(Pe) - 0.48
    lg_w_He32 = np.log10(1.0) + 2.5*np.log10(Te) - 5040*Xe_He32/Te - np.log10(Pe) - 0.48
    w_H21 = np.power(10,lg_w_H21)
    w_He21 = np.power(10,lg_w_He21)
    w_He32 = np.power(10,lg_w_He32)
    
    i0 = w_H21/(1 + w_H21)
    j1 = w_He21/(1 + w_He21 + w_He21*w_He32)
    j2 = (w_He21*w_He32)/(1 +  w_He21 + w_He21*w_He32)
    
    be = i0 + rHe*(j1+2*j2)
    N_H = Ne/be
    N_H1 = N_H*(1-i0)
    N_H2 = N_H*i0
    N_He = rHe*N_H
    N_He1 = (1-j1-j2)*N_He
    N_He2 = j1*N_He
    
    return N_H, N_H1, N_H2, N_He, N_He1, N_He2



def kai_cal(rho,pres,Te,wavelength):
    N_H, N_H1, N_H2, N_He, N_He1, N_He2 = saha_eqn(rho,pres,Te)
    
    wavelength=wavelength
    wave_ref = 171
    segma_H1 = 5.16*10**(-20)
    segma_He1 = 9.25*10**(-19)
    segma_He2 = 7.17*10**(-19)
    s_H1 = (wavelength/wave_ref)**3*segma_H1
    s_He1 = (wavelength/wave_ref)**2*segma_He1
    s_He2 = (wavelength/wave_ref)**2.75*segma_He2
    
    Xe_H21 = 13.6
    Xe_He21 = 24.587
    Xe_He32 = 54.416
    rHe = 0.1
    
    w_H21 = np.power(10,np.log10(1.0) + 2.5*np.log10(Te) - 5040*Xe_H21/Te - np.log10(pres) - 0.48)
    w_He21 = np.power(10,np.log10(4.0) + 2.5*np.log10(Te) - 5040*Xe_He21/Te - np.log10(pres) - 0.48)
    w_He32 = np.power(10,np.log10(1.0) + 2.5*np.log10(Te) - 5040*Xe_He32/Te - np.log10(pres) - 0.48)
    
    #w_H21 = np.nan_to_num(w_H21, nan=0.0)
    #w_He21 = np.nan_to_num(w_He21, nan=0.0)
    #w_He32 = np.nan_to_num(w_He32, nan=0.0)
    
    i0 = w_H21/(1 + w_H21)
    j1 = w_He21/(1 + w_He21 + w_He21*w_He32)
    j2 = (w_He21*w_He32)/(1 +  w_He21 + w_He21*w_He32)
    
    be = i0 + rHe*(j1+2*j2)
    
    N_H = rho/be
    N_H1 = N_H*(1-i0)
    N_He1 = (1-j1-j2)*rHe*N_H
    N_He2 = j1*rHe*N_H
    
    kai = N_H1*s_H1 + N_He1*s_He1 + N_He2*s_He2
    kai = np.nan_to_num(kai, nan=0)
    #kai=kai-kai
    # 设置最大值
    #max_value = 10e-1
    
    # 将超过最大值的元素替换为 3e20
    #kai[kai > max_value] = 10e-1

    return kai

def J_cal(rho,Te,wavelength):
    resline=np.load('./AIA_response/'+str(wavelength)+'_response_line.npz')
    res_line=resline['a']
    te_log=res_line[:,0]
    para=res_line[:,1]
    temperature_log=np.log10(Te)
    res=np.interp(temperature_log, te_log, para)
    res[(temperature_log < min(te_log)) | (temperature_log > max(te_log))] = 0
    J=res*(rho)**2
    if wavelength == 193 :
        J=(rho)**2
    return J



def cal_euv(rho,pres,Te,wavelength,absorption):
    if absorption:
        kai=kai_cal(rho,pres,Te,wavelength)
    else:
        kai=np.zeros(rho.shape)
    # 设置最大值
    #max_value = 10e12
    
    # 将超过最大值的元素替换为 10e20
    #kai[kai > max_value] = 10e12
    
    J=J_cal(rho,Te,wavelength)
    return kai,J

def cal_radio(rho,pres,Te):
    N_H, N_H1, N_H2, N_He, N_He1, N_He2 = saha_eqn(rho,pres,Te)

    '''data['rho']=_data['rho']
    data['Te']=_data['t']
    
    _data={}
    
    rho=data['rho']
    Te=data['Te']
    pres=Ne*Te'''
    
    N_H, N_H1, N_H2, N_He, N_He1, N_He2 = saha_eqn(rho,pres,Te)
    
    h = sc.h*1e7
    k_b = sc.k*1e7
    niu = 3*1e8
    
    kai = np.where( Te <=2e5,(0.00978 * (rho/niu**2/Te**1.5) * (N_H2+N_He1+4*N_He2) * (18.2+1.5*np.log(Te)-np.log(niu))),(0.00978 * (rho/niu**2/Te**1.5) * (N_H2+N_He1+4*N_He2) * (24.5+np.log(Te)-np.log(niu))))
    kai = np.nan_to_num(kai, nan=0)
    # 设置最大值
    #max_value = 10e12
    
    # 将超过最大值的元素替换为 3e20
    #kai[kai > max_value] = 10e12
    
    J = np.where( Te <= 2e5,(3.772e-38 * (rho/Te**0.5) * np.exp(-1*h*niu/k_b/Te) * (N_H2+N_He1+4*N_He2) * (18.2+1.5*np.log(Te)-np.log(niu))),(3.772e-38 * (rho/Te**0.5) * np.exp(-1*h*niu/k_b/Te) * (N_H2+N_He1+4*N_He2) * (24.5+np.log(Te)-np.log(niu))))
    return kai,J

def cal_white_light(rho,R):
    a1 = 8.69e-7
    a2 = 0.37
    a3 = 0.63
    r=R

    # 计算cfunc
    cfunc = 4./3. - np.sqrt(1. - 1./r**2) - ((1. - 1./r**2)**1.5) / 3.

    # 计算df
    df = (r - 1./r)*(5 - 1./r**2)*np.log((1. + 1./r) / np.sqrt(1. - 1./r**2))

    # 计算dfunc
    dfunc = 1./8 * (5. + 1./r**2 - df)

    # 计算bf
    bf = a1 * (a2*cfunc + a3*dfunc)

    I0=1

    sigma=7.95e-26

    u=0.56

    tomson_scatter=I0*rho*10**9*np.pi*sigma/2*((1-u)*cfunc+u*dfunc)

    tomson_scatter = np.nan_to_num(tomson_scatter, nan=0.0)
    kai = np.zeros(tomson_scatter.shape)
    J = tomson_scatter

    return kai, J
    