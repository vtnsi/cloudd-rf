class burst_def:
    def __init__(self, bandwidth, cent_freq, duration, snr):
        self.bandwidth = bandwidth
        self.cent_freq = cent_freq
        self.duration = duration
        self.snr = snr

    def get_low_freq(self, pos=0):
        if pos == 0:
            return self.cent_freq - self.bandwidth/2.0
        else:
            return self.cent_freq - self.bandwidth/2.0 + 0.5

    def get_high_freq(self, pos=0):
        if pos == 0:
            return self.cent_freq + self.bandwidth/2.0
        else:
            return self.cent_freq + self.bandwidth/2.0 + 0.5

    def __repr__(self):
        return 'Burst with bandwidth: {:.3f}, center frequency: {:.3f}, duration: {}, snr: {:.3f}'.format(self.bandwidth, self.cent_freq, self.duration, self.snr)
