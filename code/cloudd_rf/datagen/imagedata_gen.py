import numpy as np
import matplotlib
import matplotlib.pyplot as pyplot

class imagedata_gen:
    def __init__(self, image_width=1000, image_height=500, fft_size=256, overlap=255):
        self.image_width = image_width
        self.image_height = image_height
        self.fft_size = fft_size
        self.overlap = overlap
        self.pixel_size = 1 / pyplot.rcParams["figure.dpi"]

    def gen_image(self, file_name, burst_list, iq_data, show_plot):
        fig = pyplot.figure(figsize=(self.image_width * self.pixel_size, self.image_height * self.pixel_size),frameon=False)

        ax = pyplot.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        _, _, _, _ = pyplot.specgram(iq_data, NFFT=self.fft_size, Fs=1.0, noverlap=self.overlap, cmap="nipy_spectral")
        pyplot.savefig(file_name)

        if show_plot:
            pyplot.show()