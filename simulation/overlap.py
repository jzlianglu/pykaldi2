import numpy as np


class _PlacedUtterance:
    def __init__(self, utt_idx, wav, spk, start_sample, end_sample):
        self.utt_idx = utt_idx
        self.wav = wav
        self.spk = spk
        self.start_sample = start_sample
        self.end_sample = end_sample


class OverlapSimulator:
    """
    Given a list of utterances, and their speaker IDs, simulate the overlapping pattern of a meeting.
    """
    def __init__(self, overlap_ratio=0.1, sil_range=[0.0, 3.0], fs=16000):
        """
        overlap_ratio: desired overlap ratio, defined as 2 * d_overlap / d_total, where d_overlap is the duration of
        overlapped region and d_total is the duration of he source speech.
        sil_range: the min and max of silence between two consecutive sentences.
        fs: sampling rate
        """
        self.overlap_ratio = overlap_ratio
        self.sil_range = sil_range
        self.fs = fs

    def simulate(self, utt_list, utt_id_list, spk_list, required_duration=600, init_sil=3):
        """
        Paremeters:
            utt_list: a list of numpy arrays, each representing the waveforms of one sentence. The first dimension of the
            arrays represent time sample index.
            utt_id_list: list of strings, representing the utterance IDs. len(utt_id_list) should be the same as len(utt_list)
            spk_list: list of strings, representing the speaker IDs. len(spk_list) should be the same as len(utt_list)
            required_duration: the desired length of the output waveform in terms of seconds.
        Returns:
            mixed_wav: overlapped speech waveforms
            utt_label: a list of dictionaries, each represent the information of a segment.
            overlap ratio: a number that represent the overlap ratio in percentage.
        """
        n_sample = self.fs * required_duration
        uniq_spks = list(set(spk_list))
        n_spk = len(uniq_spks)

        utt_samples = [i.shape[0] for i in utt_list]
        # make sure that we have enough non-repeating sentences to generate the mixed speech.
        # assert np.sum(utt_samples) > (required_duration-init_sil)*(1+self.overlap_ratio)

        # get a random utterance order
        rand_utt_list = np.arange(len(utt_list))
        np.random.shuffle(rand_utt_list)

        end_sample = init_sil*self.fs
        placed_utts = list()

        overlap_target = [self.overlap_ratio]   # the list keep track of the target overlap ratio for the rest of the mixed speech
        overlap_ratio_history = []

        for utt_idx in rand_utt_list:
            # for each new sentence, randomly choose its starting point to satisfy two requirements:
            # 1. the overall overlapping ratio should approach the self.overlap_ratio
            # 2. the silence gap between two consecutive sentences satisfy self.sil_range
            # 3. there is no overlap between two sentences of the same speaker.

            curr_utt = utt_list[utt_idx]
            curr_spk = spk_list[utt_idx]
            if len(placed_utts) == 0:   # first sentence is placed right after the initial silence
                start_sample = end_sample
                end_sample += curr_utt.size
                placed_utts.append(_PlacedUtterance(utt_idx, curr_utt, curr_spk, start_sample, end_sample))
            else:
                # decide the allowable starting points for the new sentence
                if overlap_target[-1] == 0:
                    start_limit = placed_utts[-1].end_sample + self.sil_range[0]*self.fs
                else:
                    # 1. start position should be no earlier than the start position of the previous sentence
                    start_limit = placed_utts[-1].start_sample
                    # 2. the start position should be later than the speaker's previous sentence
                    spk_prev_end_sample = 0
                    for i in reversed(range(len(placed_utts))):
                        if placed_utts[i].spk == curr_spk:
                            spk_prev_end_sample = placed_utts[i].end_sample
                            break
                    start_limit = np.maximum(spk_prev_end_sample, start_limit)

                end_limit = placed_utts[-1].end_sample + self.sil_range[1]*self.fs

                # find the start position that exactly matches the desired overlapping ratio
                overlapped_samples_with_prev_utt = [np.maximum(0, i.end_sample-placed_utts[-1].start_sample) for i in placed_utts[:-1]]
                two_utt_duration = [placed_utts[-1].wav.size, curr_utt.size]
                curr_n_overlap_sample = np.sum(two_utt_duration+overlapped_samples_with_prev_utt) * overlap_target[-1]/2 - np.sum(overlapped_samples_with_prev_utt)
                if overlap_target[-1] == 0:
                    a = 1
                desired_start_position = placed_utts[-1].end_sample - curr_n_overlap_sample
                if desired_start_position < start_limit:
                    desired_start_position = start_limit
                if desired_start_position > end_limit:
                    desired_start_position = end_limit

                if end_limit < start_limit:
                    # in some case, the start_limit is even longer than the start_limit, e.g. a longer sentence of
                    # speaker A followed by a short sentence of speaker B, then curr_spk is A again. In this case,
                    # simply sample a silence gap
                    final_start_position = np.random.randint(start_limit, start_limit+self.sil_range[1]*self.fs)
                else:
                    # sample whether the current sentence should overlap with previous sentence
                    if np.random.uniform() > 0.0:
                        std = (end_limit-start_limit)/12
                        final_start_position = self._sample_gaussian_with_limits(mean=desired_start_position, std=std,
                                                                                limits=[start_limit, end_limit])
                    else:
                        final_start_position = np.random.randint(placed_utts[-1].end_sample, placed_utts[-1].end_sample + self.sil_range[1] * self.fs)

                final_start_position = int(final_start_position)
                end_sample = final_start_position + curr_utt.size
                placed_utts.append(_PlacedUtterance(utt_idx, curr_utt, curr_spk, final_start_position, end_sample))

            overlap_ratio_history.append(self._comp_overlap_ratio(placed_utts))
            # update overlap ratio target for the rest of the sentences
            latest_filled_sample = np.max([i.end_sample for i in placed_utts])
            new_target = self.overlap_ratio * n_sample - latest_filled_sample * overlap_ratio_history[-1]
            new_target /= n_sample - latest_filled_sample
            new_target = np.maximum(np.minimum(1.0, new_target), 0.0)
            overlap_target.append(new_target)

            if end_sample >= n_sample:
                break

        ratio = self._comp_overlap_ratio(placed_utts)
        if 0:
            import matplotlib.pyplot as plt
            plt.subplot(2,1,1)
            self._plot_utts(placed_utts)
            plt.title(ratio)
            plt.subplot(2,1,2)
            plt.plot(overlap_ratio_history)
            plt.plot(overlap_target)
            plt.legend(['overlap history', 'overlap target'])
            plt.show()

        # generate the mixed speech

        mixed_wav = np.zeros((placed_utts[-1].end_sample, n_spk))
        utt_label = []
        for i in range(len(placed_utts)):
            curr_spk = placed_utts[i].spk
            channel = uniq_spks.index(curr_spk)
            idx1 = placed_utts[i].start_sample
            idx2 = placed_utts[i].end_sample
            mixed_wav[idx1:idx2, channel:channel+1] = placed_utts[i].wav
            utt_label.append((idx1, idx2, curr_spk, utt_id_list[placed_utts[i].utt_idx]))

        if 0:
            for i in range(n_spk):
                plt.plot(mixed_wav[:,i]+i)
            wav_sum = np.sum(mixed_wav, axis=1, keepdims=True)
            wav_sum /= np.max(np.abs(wav_sum))
            plt.plot(wav_sum+i+1)

        return mixed_wav, utt_label, overlap_ratio_history[-1]

    def _comp_overlap_ratio(self, placed_utts):
        max_sample = np.max([i.end_sample for i in placed_utts])
        overlap_indivator = np.zeros(max_sample)
        for utt in placed_utts:
            overlap_indivator[utt.start_sample:utt.end_sample] +=1

        source_duration_sum = np.sum(overlap_indivator)
        overlap_duration_sum = np.sum(overlap_indivator[np.where(overlap_indivator>1)[0]])
        overlap_ratio = float(overlap_duration_sum) / float(source_duration_sum)

        return overlap_ratio

    def _plot_utts(self, placed_utts):
        max_sample = np.max([i.end_sample for i in placed_utts])
        spks = list(set([i.spk for i in placed_utts]))
        spks.sort()
        for i in range(len(spks)):
            spk_indivator = np.zeros(max_sample)
            for utt in placed_utts:
                if utt.spk == spks[i]:
                    spk_indivator[utt.start_sample:utt.end_sample] += 1
            plt.plot(spk_indivator+i*1.05)
            plt.text(0, i*1.05+0.5, spks[i])

        overlap_indivator = np.zeros(max_sample)
        for utt in placed_utts:
            overlap_indivator[utt.start_sample:utt.end_sample] +=1
        plt.plot(overlap_indivator + (i+1)*1.05)
        plt.text(0, (i+1) * 1.05 + 0.5, 'overlap indicator')

    def _sample_gaussian_with_limits(self, mean, std, limits):
        if limits[0]<=mean and limits[1]>=mean:
            pass
        else:
            a = 1
        assert std>0

        valid = False
        n_trial = 1000
        for i in range(n_trial):
            data=np.random.normal(loc=float(mean),scale=float(std),size=1)
            if data >= limits[0] and data <= limits[1]:
                valid=True
                break
        if not valid:
            print("Warning: no valid data obtained after %d trial. Use mean value instead." % n_trial)
            data = mean
        if 0:
            data2 = np.random.normal(loc=float(mean), scale=float(std), size=10000)
            plt.hist(data2, 100)
            plt.plot([limits[0], limits[0]], [0, 100])
            plt.plot([limits[1], limits[1]], [0, 100])

        return data


def _test():
    fs = 16000
    n_spk = 5
    n_utt_per_spk = 50
    utt_dur_range = [2.0, 15.0]

    utt_list = list()
    spk_list = list()
    for i in range(n_spk):
        utt_list += [np.ones((np.random.randint(fs*utt_dur_range[0], fs*utt_dur_range[1])))*(i*100+j) for j in range(n_utt_per_spk)]
        spk_list += ['spk_'+str(i) for j in range(n_utt_per_spk)]

    simulator = OverlapSimulator(overlap_ratio=0.2, sil_range=[0.0, 3.0])
    simulator.simulate(utt_list, spk_list, required_duration=700)
    