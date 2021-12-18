import numpy as np
import glob
import os
import sys
import h5py
from astropy.io import fits
from specutils.spectra import Spectrum1D
from astropy import units as u
from specutils.fitting.continuum import fit_continuum
from astropy.modeling.polynomial import Chebyshev1D

# set the files to processed
fl = glob.glob('t_files_toi530/*t.fits')

# set orders to  be preprocessed
orders = np.arange(0, 11, 1)
orders = np.hstack((orders, np.arange(16, 22, 1)))
orders = np.hstack((orders, np.arange(28, 37, 1)))
orders = np.hstack((orders, np.arange(43, 45, 1)))

# set and create the folder to store the temporary files
csv_dir = 'temp_for_R'
os.makedirs(csv_dir)

# telluric lines information
tel_mask = np.loadtxt('telluric_mask_950-2440.txt')

# set the name of the output file
if len(sys.argv) > 1:
    out_file_name = sys.argv[1]
else:
    out_file_name = 'pre_data.hdf5'

# Remove emiision lines
def remove_emission(w, f):
    pmin = 500
    pmax = 4088 - 500
    
    f[:,0:pmin] = np.nan
    f[:,pmax:] = np.nan
    
    f = np.where(f >0 ,f, np.nan )  #f小于0置为nan

    pixel_range = pr = 150            #计算方差时每一段非nan的数据个数
    pixel_range_max = prm = 300         #由于谱线之间有间隙，该值为每一段最多跨越pixel个数
    
    del_pixel = 3                     #去除发射线时不正常点两侧删除像素个数
        
    for i, line in enumerate(f):            # i:行索引  去除过于离谱的线 减少建模影响
        uplim = np.nanpercentile(line, 80)
        temp = np.where(line > 2 * uplim)[0]
        for j in range(- del_pixel, del_pixel+1):
            k = temp + j
            k = k[k >= 0]
            k = k[k < len(line)]
            f[i][k] = np.nan
        
    for i, line in enumerate(f):         #第一次拟合
        spectrum = Spectrum1D(flux = line * u.photon , spectral_axis = w[i]*u.nm)
        fitted_continuum = fit_continuum(spectrum,model=Chebyshev1D(5))
        y_fit = fitted_continuum(w[i]*u.nm)
                        
        f_list = np.where(~np.isnan(line))[0]        # 非nan数据点的位置(pixel)   
        list_left = f_list[0]                       #每一段左侧
        list_right = min(f_list[min(pr,f_list.size-1)],4087)
        #list_left/right 指的是左右pixel位置
        while list_right < f_list[-1]:
            
            if list_right - list_left <= prm :        # 此时认为像素断裂程度不算太大，当作一段计算

                w_part = w[list_left : list_right]
                line_part = line[list_left : list_right]
                y_fit_part = y_fit.value[list_left : list_right]

                
                var = np.nanvar(line_part/y_fit_part)    # 计算方差

                
                del_list = np.array([],dtype=int)       #有问题pixel列表
                for k in np.arange(0,list_right - list_left):      
                    if (line_part[k]/y_fit_part[k] - 1) > 3*np.sqrt(var):    
                        del_list = np.append(del_list,int(k))  

                for l in del_list:                        # 删除发射线和它附近的pixel
                    f[i][list_left + l - del_pixel : min(list_left + l + del_pixel +1,4087)] = np.nan            
                
                if list_right == f_list[-1]:
                    break
                else:
                #步进1/2区间检测，ind_left为list_left的index值,前一个区间的left为前一个区间的中间值，可调整
                    ind_left = np.where(f_list == list_left)[0][0]
                    ind_right = np.where(f_list == list_right)[0][0]
                
                    list_left = f_list[int((ind_left + ind_right)/2)]
                    if ind_left + pr < f_list.size:
                        list_right = f_list[ind_left + pr]
                    else:
                        list_right = f_list[-1]
                        
                        
            else :      #像素间间距过大
                w_part = w[list_left : list_right]
                line_part = line[list_left : list_right]
                y_fit_part = y_fit.value[list_left : list_right]
                
                
                var = np.nanvar(line_part/y_fit_part)
                
                
                del_list = np.array([],dtype=int)
                for k in np.arange(0,list_right - list_left):
                    if (line_part[k]/y_fit_part[k] - 1) > 3*np.sqrt(var):
                        del_list = np.append(del_list,int(k))
                        
                for l in del_list:
                    f[i][list_left + l - del_pixel : min(list_left + l + del_pixel +1,4087)] = np.nan                                    
                                                            
                if list_right == f_list[-1]:
                    break
                else:
                    #步进1/2区间检测，ind_left为list_left的index值,...
                    ind_left = np.where(f_list == list_left)[0][0]
                    ind_right = np.where(f_list == list_right)[0][0]
                    
                    list_left = f_list[int((ind_left + ind_right)/2)]
                    if ind_left + pr < f_list.size:
                        list_right = f_list[ind_left + pr]
                    else:
                        list_right = f_list[-1]
    
    return w, f

# Transit original spectra to csv files (to normalize the spectra with R code)
def fits2csv(fits_e, orders, csv_dir):
    x = fits_e[2].data[orders]
    spes = fits_e[1].data[orders]

    # remove the emission lines
    x, spes = remove_emission(x, spes)
    x = x[:, :-700]
    spes = spes[:, :-700]
    # deal with nan
    ith = np.isnan(spes)
    spes[ith] = 0

    for i, order in enumerate(orders):
        output = np.stack((x[i], spes[i]), axis=-1)
        np.savetxt(f'{csv_dir}/order{str(order).zfill(2)}.csv', output,
                   delimiter=',', header='wv,flux', comments='',fmt='%f')

# preprocess the spectra
def treat_spe(e):
    fits2csv(e, orders, csv_dir)
    # normalize the spectra with R code
    os.system('Rscript R_normalize.R')
    csv_fl = glob.glob(f'{csv_dir}/*.csv')
    x, spes, ivars = [], [], []
    for csv_f in csv_fl:
        data = np.loadtxt(csv_f, skiprows=1, delimiter=',')
        x.append(data[:, 0])
        spes.append(data[:, 2])
        ivars.append(data[:, 1])
        os.remove(csv_f)
    x = np.array(x)
    spes = np.array(spes)
    ivars = np.array(ivars)
    # in specific orders, mask out the lines which is higher than 1.2 after normalizing (very likely to be emission lines)
    for i, spe in enumerate(spes):
        if (28 <= orders[i] <= 36):
            ith = np.where(spe > 1.2)[0]
            for j in range(-2, 3):
                arg = ith + j
                arg = arg[arg >= 0]
                arg = arg[arg < len(spe)]
                spes[i][arg] = 1
                ivars[i][arg] = 0
    # mask out tellurics
    for i in range(len(tel_mask) // 4):
        beg = tel_mask[(4*i), 0]
        end = tel_mask[(4*i+3), 0]
        spes[(x >= beg) & (x <= end)] = 1
        ivars[(x >= beg) & (x <= end)] = 0

    # return the results
    return np.log(spes), np.log(x), ivars

# Save the results as hdf5 files
def pre(fl, out_f):
    data, ivars, xs, pipeline_rvs, pipeline_sigmas = [], [], [], [], []
    dates, bervs, airms, drifts = [], [], [], []
    for f in fl:
        print(f)
        e = fits.open(f)
        d, x, ivar = treat_spe(e)
        data.append(d)
        xs.append(x)
        ivars.append(ivar)
        dates.append(e[1].header['BJD'])
        bervs.append(e[1].header['BERV']*1000)
        airms.append(e[0].header['AIRMASS'])
        pipeline_rvs.append(0.0)
        pipeline_sigmas.append(0.0)
        drifts.append(0.0)
    data = np.transpose(data, (1, 0, 2))
    xs = np.transpose(xs, (1, 0, 2))
    ivars = np.transpose(ivars, (1, 0, 2))
    write_data(data, ivars, xs, pipeline_rvs, pipeline_sigmas, dates, bervs, airms, drifts, fl, out_f)

def write_data(data, ivars, xs, pipeline_rvs, pipeline_sigmas, dates, bervs, airms, drifts, filenames, hdffile):
    h = h5py.File(hdffile, 'w')
    dset = h.create_dataset('data', data=data)
    dset = h.create_dataset('ivars', data=ivars)
    dset = h.create_dataset('xs', data=xs)
    dset = h.create_dataset('pipeline_rvs', data=pipeline_rvs)
    dset = h.create_dataset('pipeline_sigmas', data=pipeline_sigmas)
    dset = h.create_dataset('dates', data=dates)
    dset = h.create_dataset('bervs', data=bervs)
    dset = h.create_dataset('airms', data=airms)
    dset = h.create_dataset('drifts', data=drifts)
    filenames = [a.encode('utf8') for a in filenames] # h5py workaround
    dset = h.create_dataset('filelist', data=filenames)
    h.close()


pre(fl, out_file_name)

# delete the folder for the temporary files
os.removedirs(csv_dir)
