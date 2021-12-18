import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import wobble
import matplotlib as mpl

# 设置绘图格式
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['figure.figsize'] = (12.0, 4.0)

# 设置读入数据的文件名称
if len(sys.argv) > 1:
    in_file_name = sys.argv[1]
else:
    in_file_name = 'pre_data.hdf5'    

# 读入数据
data = wobble.Data(in_file_name)

# 设置使用的hdf5里的orders
orders = np.arange(0, data.R)

# 设置参数
day = 2  #后面绘制光谱时所选的日期
result_RV_name = 'RV.txt'  #输出最终RV的文件名
spe_dir = 'spe_results_plots'  #存光谱图的文件夹
RV_dir = 'RV_results_each_order'  #存各order所得RV的文件夹
RV_filename = 'RV_result.png'  #结合所有order的RV的图的文件名
result_data_name = 'all_results.hdf5'  #存结果数据的hdf5文件的文件名
regu_file_star = 'regularization_file/default_star.hdf5'  #设置使用的regularization的文件名

# create folders
if not os.path.exists(spe_dir):
    os.makedirs(spe_dir)
if not os.path.exists(RV_dir):
    os.makedirs(RV_dir)

# 对各order进行处理并绘图
results = wobble.Results(data=data)
for r in orders:
    # 处理各order
    print('starting order {0} of {1}'.format(r+1, len(data.orders)))
    model = wobble.Model(data, results, r)
    starting_rvs = -1. * np.copy(data.bervs) + np.mean(data.bervs)
    shifted_xs = data.xs[r] + np.log(wobble.doppler(starting_rvs[:, None], tensors=False))
    x_beg, x_end = np.min(shifted_xs), np.max(shifted_xs)
    dx = 7e-6
    tem_x = np.arange(x_beg-10*dx, x_end+10*dx, dx)
    tem_y = np.zeros_like(tem_x)
    model.add_star('star', regularization_par_file=regu_file_star, learning_rate_template=0.005, learning_rate_rvs=1000,
                    template_xs=tem_x, template_ys=tem_y)
    wobble.optimize_order(model, niter=4000)

    # 绘光谱图
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[2, 1]}, 
                               sharex=True, figsize = (12.0, 6.0))
    ax1.plot(np.exp(data.xs[r][day]), np.exp(data.ys[r][day]), 'k.', ms=6)
    ax1.plot(np.exp(data.xs[r][day]), np.exp(results.star_ys_predicted[r][day]), label='star')
    ax1.legend()

    residual = np.exp(data.ys[r][day]) - np.exp(results.star_ys_predicted[r][day])
    ax2.plot(np.exp(data.xs[r][day]), residual, 'k.', ms=4)
    ax2.set_ylabel('residual')
    ax2.set_ylim(-0.2, 0.2)
    fig.savefig(f'{spe_dir}/order{r}.png')
    plt.close()

    # 绘制RV图
    plt.errorbar(results.dates, results.star_rvs[r] + results.bervs, 
             1./np.sqrt(results.star_ivars_rvs[r]), 
             fmt='o', ms=5, elinewidth=1)
    plt.xlabel('JD')
    plt.ylabel(r'RV (m s$^{-1}$)')
    plt.savefig(f'{RV_dir}/order{r}.png')
    plt.clf()

    # 输出各order RV
    out_RV = np.stack((data.dates, results.star_rvs[r] + results.bervs, 1./np.sqrt(results.star_ivars_rvs[r])))
    out_RV = np.transpose(out_RV)
    np.savetxt(f'{RV_dir}/order{r}.txt', out_RV, header='dates RV(m/s) RV_err(m/s)', fmt='%.12g %.17g %.17g')

# combine rv的函数
def vank(rvmat,simple=False):
    # return weighted averaged RVs through a simple vanking with no outlier rejection
    nords = np.shape(rvmat)[0]
    nfiles = np.shape(rvmat)[1]
    rvmat_old = copy.deepcopy(rvmat)  # well this did nothing; still modified rvmat
    
    # mean of each order across nights should be set to 0
    for iord in range(nords):
        rvmat[iord,:] = rvmat[iord,:] - np.mean(rvmat[iord,:])
        
    # Next, variance of each order
    rvmed = np.median(rvmat,axis=0)  # median RV of all chunks of each night
    rvmarr = np.repeat(rvmed,nords)
    rvmarr = np.transpose(np.reshape(rvmarr,(nfiles,nords)))
    drv = np.abs(rvmat - rvmarr)  # Delta_ij in thesis equation 2.8
    if simple:
        drv_median = np.median(drv, axis=1)
        ord_wt = 1.0/drv_median**2
        wrv = np.dot(np.transpose(rvmat),ord_wt)/np.sum(ord_wt)
        rverr = np.std(rvmat,axis=0)/np.sqrt(nfiles)
        wij = 0
    else:
        sigmai = np.std(drv,axis=1)  # weight of each chunk
        sigmamat = np.repeat(sigmai,nfiles)
        sigmamat = np.reshape(sigmamat,(nords,nfiles))
        rj = np.median(np.abs(drv)/sigmamat, axis=0)  # adjusted weight of each night
        wij = 1.0/(np.outer(sigmai,rj)**2)
        w1percent = np.percentile(wij,5.0)
        low_wt_ind = np.where(wij <= w1percent)
        wij[low_wt_ind] = np.nan
        
        wrv = np.nansum(rvmat*wij,axis=0)/np.nansum(wij,axis=0)
        rverr = 1.0/np.sqrt(np.nansum(wij,axis=0))
    
    return wrv,rverr

# 组合各order并输出结果
results.apply_drifts('star')
results.apply_bervs('star')  #考虑BERV
combined_rv, combined_sigma = vank(results.star_rvs[orders])

# 绘制RV
plt.errorbar(data.dates, combined_rv, combined_sigma, fmt='o', ms=5, elinewidth=1)
plt.xlabel('JD')
plt.ylabel(r'RV (m s$^{-1}$)')
plt.savefig(RV_filename)

# 输出RV
out_RV = np.stack((data.dates, combined_rv, combined_sigma))
out_RV = np.transpose(out_RV)
np.savetxt(result_RV_name, out_RV, header='dates RV(m/s) RV_err(m/s)', fmt='%.12g %.17g %.17g')
