import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.fft import fft
from scipy.signal.windows import hamming

# 读取JSON文件中的配置
with open('D:\LASSO\dataset_60GHz\drop\\2024_1_8_data220240108-160332\RadarIfxAvian_00\config.json', 'r') as file:
    config = json.load(file)

# 从JSON配置中提取参数
sample_rate = config['device_config']['fmcw_single_shape']['sample_rate_Hz']
num_samples_per_chirp = config['device_config']['fmcw_single_shape']['num_samples_per_chirp']
num_chirps_per_frame = config['device_config']['fmcw_single_shape']['num_chirps_per_frame']
num_rx_antennas = len(config['device_config']['fmcw_single_shape']['rx_antennas'])

# 假设 data 是形状为 [num_frames, num_rx_antennas, num_chirps_per_frame, num_samples_per_chirp] 的雷达数据数组
data = np.load("D:\LASSO\dataset_60GHz\drop\\2024_1_8_data220240108-160332\RadarIfxAvian_00\\radar.npy")

# 获取多普勒频谱
def get_doppler_spectrum(data, sample_rate, window_function):
    num_frames, num_rx, num_chirps, num_samples = data.shape
    # 频谱分析的结果数组
    doppler_spectrum = np.zeros((num_frames, num_samples))

    for frame in range(num_frames):
        # 使用窗函数
        windowed_signal = data[frame, 0, 0, :] * window_function
        # 执行FFT
        fft_result = fft(windowed_signal)
        # 只关注频率在人体运动范围内的部分
        doppler_spectrum[frame, :] = np.abs(fft_result)

    # 计算频率轴
    freq = np.fft.fftfreq(num_samples, d=1/sample_rate)
    return freq, doppler_spectrum

# 创建窗函数
window_function = hamming(num_samples_per_chirp)

# 获取多普勒频谱
freq, doppler_spectrum = get_doppler_spectrum(data, sample_rate, window_function)

# 绘制二维多普勒频谱图
plt.imshow(doppler_spectrum, aspect='auto', extent=[freq.min(), freq.max(), 0, 100])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Frame')
plt.title('2D Doppler Spectrum')
plt.colorbar(label='Magnitude')
plt.show()
