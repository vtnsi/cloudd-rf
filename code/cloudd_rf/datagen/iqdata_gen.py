import numpy
from fractions import Fraction
import matplotlib.pyplot as pyplot
from scipy.signal import resample_poly

# Datagen Imports
import cloudd_rf.datagen.modems as modems

class iqdata_gen:
    def __init__(self, obs_int=10000, num_sensors=1, rng=numpy.random.default_rng()):
        self.obs_int = obs_int
        self.num_sensors = num_sensors
        self.rng = rng
        self.noise_std = 1.0

    def gen_iq(self, modulation, burst_list):
        time = []
        for x in range(0, self.obs_int):
            time.append(x)

        iq_data = []
        for k in range(self.num_sensors):
            iq_data.append(self.rng.normal(0.0, self.noise_std/numpy.sqrt(2.0), burst_list[k].duration) + 1.0j*(self.rng.normal(0.0, self.noise_std/numpy.sqrt(2.0), burst_list[k].duration)))

        if modulation[1][0] == 'ask' or modulation[1][0] == 'psk' or modulation[1][0] == 'qam':
            beta = self.rng.uniform(0,1)
            modem = modems.ldapm(sps=2, mod_type=modulation[1][0], mod_order=modulation[1][1], filt_type='rrc', beta=beta, span=10, trim=0)
        elif modulation[1][0] == 'constant' or modulation[1][0] == 'p_fmcw' or modulation[1][0] == 'n_fmcw':
            modem = modems.tone(sps=2, mod_type=modulation[1][0])
        samps = modem.gen_samps(self.obs_int)

        for k in range(self.num_sensors):
            ratio = Fraction(0.5/burst_list[k].bandwidth).limit_denominator(10)
            samps = resample_poly(samps, ratio.numerator, ratio.denominator)
            samps = samps[0:burst_list[k].duration]
            samps = samps*[numpy.exp(2.0j*numpy.pi*burst_list[k].cent_freq*x) for x in range(0, len(samps))]
            burst_list[k].bandwidth = 0.5/(ratio.numerator/ratio.denominator)

            if modulation[1][0] != 'constant':
                snr_db = burst_list[k].snr + 10.0*numpy.log10(1.0/(ratio.numerator/ratio.denominator))
            else:
                snr_db = burst_list[k].snr + 10.0*numpy.log10(burst_list[k].bandwidth)            
            snr_index = numpy.where(numpy.abs(samps) != 0)
            snr_lin = 10.0**(snr_db/20.0)
            sig_pow_req = (self.noise_std**2)*snr_lin
            sig_pow = numpy.sqrt(numpy.mean(numpy.abs(samps[snr_index])**2))
            sig_amp = sig_pow_req/sig_pow
            samps = sig_amp * samps

            iq_data[k] += samps

        return iq_data, burst_list

    def plot_iq(self, iq_data, image_width=1000, image_height=500, fft_size=256, overlap=255, ax=[]):
        if ax == []:
            pixel_size = 1/pyplot.rcParams['figure.dpi']
            fig_size = (image_width*pixel_size, image_height*pixel_size)

            fig, ax = pyplot.subplots()
            fig.set_size_inches(fig_size)
            pyplot.xlim([0, self.obs_int])
            pyplot.ylim([-1.0/2.0, 1.0/2.0])
            pyplot.xlabel('Time [samples]')
            pyplot.ylabel('Normalized Frequency [f/F_s]')
            pyplot.xticks(numpy.linspace(0, self.obs_int, 11))
            pyplot.yticks(numpy.linspace(-1.0/2.0, 1.0/2.0, 21))

        pyplot.specgram(iq_data, fft_size, 1.0, noverlap=overlap, cmap='nipy_spectral')

        return ax
