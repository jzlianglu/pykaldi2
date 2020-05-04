import numpy as np
import sys, os
sys.path.append('..')
from utils import utils


class MaskEstimator:
    def __init__(self, analyzer):
        self.analyzer = analyzer

    def get_mask_from_parallel_data(self, clean, distorted, vad=None, use_soft_mask=False, threshold=0.5, clean_mask_type='count', power_percentage_threshold=0.997):
        clean_spec = self.analyzer.analyze(clean)
        power_clean = np.abs(clean_spec) ** 2
        n_fr = clean_spec.shape[1]

        distorted_spec = self.analyzer.analyze(distorted)
        noise_spec = distorted_spec - clean_spec

        if use_soft_mask:
            power_distorted = np.abs(distorted_spec) ** 2
            mask_snr = np.minimum(1, power_clean / power_distorted)
        else:
            power_noise = np.abs(noise_spec) ** 2
            snr = 10 * np.log10(power_clean / np.maximum(power_noise, np.finfo(np.float32).eps))
            mask_snr = snr > threshold

        if vad is not None:
            vad_clean = vad > 0.5
            vad = np.convolve(vad_clean, np.ones(5,1), mode='same')
        elif n_fr > 30:
            # use energy based VAD, to be implemented
            vad_clean = np.ones((1,n_fr))
        else:
            vad_clean = np.ones((1,n_fr))

        mask_clean = self.get_mask_from_clean(power_clean, clean_mask_type=clean_mask_type, power_percentage_threshold=power_percentage_threshold)

        mask_combined = mask_snr * mask_clean

        mask_combined_vad = mask_combined.astype(np.float32) * vad_clean.astype(np.float32)

        if 0:
            import matplotlib.pyplot as plt
            plt.subplot(231)
            utils.imagesc(np.log(power_clean), title="Clean log spectrum")
            plt.plot(vad_clean.transpose()*power_clean.shape[0]*0.8)
            plt.show()
            plt.subplot(232)
            utils.imagesc(np.log(np.abs(distorted_spec)**2), title="Distorted log spectrum")
            plt.subplot(233)
            utils.imagesc(mask_snr.astype(float), title="SNR based mask")
            plt.subplot(234)
            utils.imagesc(mask_clean.astype(float), title="Clean mask")
            plt.subplot(235)
            utils.imagesc(mask_combined.astype(float), title="Combined mask")
            plt.subplot(236)
            utils.imagesc(mask_combined_vad.astype(float), title="Combined mask with VAD")

        return mask_combined_vad

    def get_mask_from_clean(self, power_clean, vad=None, clean_mask_type='count', power_percentage_threshold=0.997):
        if clean_mask_type == 'floor':
            pass

        elif clean_mask_type == 'count':
            power_clean_sort = np.sort(power_clean.reshape(power_clean.size,1), axis=0)
            power_clean_cumsum = np.cumsum(power_clean_sort)
            cutoff_idx = np.where( power_clean_cumsum < (1.0-power_percentage_threshold) * power_clean_cumsum[-1] )
            cutoff_threshold = power_clean_sort[cutoff_idx[0][-1]]
            mask = power_clean > cutoff_threshold
        else:
            mask = np.ones(power_clean.shape)

        return mask