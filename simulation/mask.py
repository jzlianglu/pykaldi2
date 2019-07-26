import numpy as np
from . import simulation


class MaskEstimator:
    """
    Esitmate the oracle/ideal time-frequency mask given parallel clean and distorted waveforms.
    For each time-frequency bin of the spectrogram, a mask value of 1 means that the bin is dominated by speech, and a
    value of 0 means it is dominated by noise or other non-speech energy.
    """
    def __init__(self, analyzer, snr_threshold=0.5, clean_mask_type='count', clean_mask_energy_threshold=0.997):
        """
        :param analyzer: a SpectrumAnalyzer object
        :param snr_threshold: threshold for SNR mask.
        :param clean_mask_type: how to get mask from clean speech.
        :param power_percentage_threshold: if we get
        """
        self._analyzer = analyzer
        self._snr_threshold = snr_threshold
        self._clean_mask_type = clean_mask_type
        self._clean_mask_energy_threshold = clean_mask_energy_threshold

    def get_mask_from_parallel_data(self, clean, distorted, vad=None, use_soft_mask=False):
        """
        Generate ideal binary time-frequency mask from clean/distorted parallel waveforms.
        :param clean: 1D array containing clean waveform
        :param distorted: 1D array containing distorted waveform that has the same shape as clean.
        :param vad: 1D array containing the VAD information. The unit is frame, not sample.
        :param use_soft_mask: whether to use soft mask whose value is in range [0, 1]. If False, return binary mask.
        :return: a time-frequency mask matrix
        """
        clean_spec = self._analyzer.analyze(clean)
        power_clean = np.abs(clean_spec) ** 2
        n_fr = clean_spec.shape[1]

        distorted_spec = self._analyzer.analyze(distorted)
        noise_spec = distorted_spec - clean_spec

        if use_soft_mask:
            power_distorted = np.abs(distorted_spec) ** 2
            mask_snr = np.minimum(1, power_clean / power_distorted)
        else:
            power_noise = np.abs(noise_spec) ** 2
            snr = 10 * np.log10(power_clean / np.maximum(power_noise, np.finfo(np.float32).eps))
            mask_snr = snr > self._snr_threshold

        if vad is not None:
            vad_clean = vad > 0.5
            vad = np.convolve(vad_clean, np.ones(5,1), mode='same')
        elif n_fr > 30:
            # use energy based VAD, to be implemented
            vad_clean = np.ones((1,n_fr))
        else:
            vad_clean = np.ones((1,n_fr))

        mask_clean = self._get_mask_from_clean(power_clean)

        mask_combined = mask_snr * mask_clean

        mask_combined_vad = mask_combined.astype(np.float32) * vad_clean.astype(np.float32)

        #self._show_mask()

        return mask_combined_vad

    def _get_mask_from_clean(self, power_clean):
        """
        Generate mask from clean spectrogram by either using an energy floor or selecting TF bins in reverse order
        sorted by energy until the selected bins' total energy exceeds a predefined percentage of the total energy.
        """
        if self._clean_mask_type == 'floor':
            pass    # to be implemented

        elif self._clean_mask_type == 'count':
            power_clean_sort = np.sort(power_clean.reshape(power_clean.size,1), axis=0)
            power_clean_cumsum = np.cumsum(power_clean_sort)
            cutoff_idx = np.where( power_clean_cumsum < (1.0-self._clean_mask_energy_threshold) * power_clean_cumsum[-1] )
            cutoff_threshold = power_clean_sort[cutoff_idx[0][-1]]
            mask = power_clean > cutoff_threshold
        else:
            mask = np.ones(power_clean.shape)

        return mask

    def _show_mask(self):
        import matplotlib.pyplot as plt
        plt.subplot(231)
        simulation.imagesc(np.log(power_clean), title="Clean log spectrum")
        plt.plot(vad_clean.transpose() * power_clean.shape[0] * 0.8)
        plt.show()
        plt.subplot(232)
        simulation.imagesc(np.log(np.abs(distorted_spec) ** 2), title="Distorted log spectrum")
        plt.subplot(233)
        simulation.imagesc(mask_snr.astype(float), title="SNR based mask")
        plt.subplot(234)
        simulation.imagesc(mask_clean.astype(float), title="Clean mask")
        plt.subplot(235)
        simulation.imagesc(mask_combined.astype(float), title="Combined mask")
        plt.subplot(236)
        simulation.imagesc(mask_combined_vad.astype(float), title="Combined mask with VAD")
