import numpy as np
from .sampling import get_distribution_template


def gen_stft_config():
    config = {}
    config['fs'] = 16000
    config['frame_len'] = 400
    config['frame_shift'] = 160
    config['fft_size'] = 512
    config['window'] = 'hamming'
    config['dc_removal'] = False
    config['do_dither'] = True
    return config


def gen_default_config(mic_positions=[[0],[0],[0]]):

    config = {}

    config['analysis'] = gen_stft_config()

    config['num_simulated'] = get_distribution_template('distribution of how many sentences to simulate from each pair of sources', max=1,
                                                  min=1, distribution='uniform_int')
    config['global_snr'] = get_distribution_template('distribution of global signal-to-noise ratio (SNR)', max=30,
                                              min=0, distribution='uniform')
    # define the distribution of the number of speaker sources and SNRs
    config['sources'] = {}
    config['sources']['num_spk'] = get_distribution_template('distribution of how many speech sources to use in the simulation', max=3,
                                                  min=1, distribution='uniform_int')
    config['sources']['position_scheme'] = "random_coordinate"      # [minimum_angle|random_coordinate]
    config['sources']['between_source_angle'] = get_distribution_template('distribution of angles between sources in degrees',
                                                               max=180, min=20, distribution='uniform')
    config['sources']['min_dist_from_wall'] = [0.5,0.5]    # the minimum distance between speech source and a wall
    config['sources']['min_dist_from_array'] = [0.3]    # the minimum distance between speech source and array center
    config['sources']['min_dist_from_other'] = [0.5]    # the minimum distance between two speech sources
    config['sources']['height'] = get_distribution_template('distribution of the height of speech sources in meters', max=2,
        min=1, distribution='uniform')

    config['dir_noise'] = {}
    config['dir_noise']['use_dir_noise'] = True
    config['dir_noise']['snr'] = get_distribution_template('distribution of the SNR of directional noises w.r.t. total reverberant signal power', max=20,
                                              min=0, distribution='uniform')
    config['dir_noise']['num_dir_noise'] = get_distribution_template('distribution of how many directional noise sources to use in the simulation',
                                                  max=4, min=1, distribution='uniform_int')

    config['mixing'] = {}
    config['mixing']['mixed_length'] = {}
    config['mixing']['mixed_length']['scheme'] = 'longest_source'
    config['mixing']['mixed_length']['min'] = 5
    config['mixing']['positioning'] = {}
    config['mixing']['positioning']['scheme'] = 'random_start'
    config['mixing']['ref_source'] = 'first_source'
    config['mixing']['spr'] = get_distribution_template('distribution of signal power ratio (SPR) between the speech sources', max=2.5,
                                              min=-2.5, distribution='uniform')

    config['iso_noise'] = {}
    config['iso_noise']['use_iso_noise'] = True
    config['iso_noise']['use_corpus'] = False
    config['iso_noise']['type'] = 'sph'             # [sph|cyl]
    config['iso_noise']['spectrum_type'] = 'hoth'   #[white|hoth]
    config['iso_noise']['snr'] = get_distribution_template('distribution of the SNR of isotropic noises w.r.t. unit variance signal', max=30,
                                              min=10, distribution='uniform')

    config['reverb'] = {}
    config['reverb']['use_reverb'] = True
    config['reverb']['use_corpus'] = False
    config['reverb']['t60threshold_for_online_rir'] = 0.3   # if T60 is smaller than the threshold, use online rir generation
    config['reverb']['t60'] = get_distribution_template('distribution of T60 reverberation time', max=1.0,
                                              min=0.1, distribution='uniform')
    config['reverb']['max_rir_length'] = 8000

    # define geometries
    config['array'] = {}
    config['array']['mic_positions'] = mic_positions
    config['array']['position_scheme'] = 'ratio'
    config['array']['length_ratio'] = get_distribution_template('distribution of array position in length axis (percentage)', max=0.8,
                                              min=0.2, distribution='uniform')
    config['array']['width_ratio'] = get_distribution_template('distribution of array position in width axis (percentage)', max=0.8,
                                              min=0.2, distribution='uniform')
    config['array']['height_ratio'] = get_distribution_template('distribution of array position in height axis (percentage)', max=0.6,
                                              min=0.4, distribution='uniform')

    config['room'] = {}
    config['room']['length'] = get_distribution_template('distribution of room length in meters', max=20,
                                              min=2.5, distribution='uniform')
    config['room']['width'] = get_distribution_template('distribution of room width in meters', max=20,
                                              min=2.5, distribution='uniform')
    config['room']['height'] = get_distribution_template('distribution of room height in meters', max=5,
                                              min=2.5, distribution='uniform')

    return config


def single_channel_single_source_config(use_reverb=True, use_noise=True, snr_range=[0, 30], t60_range=[0.1, 0.5]):
    """
    get configuration for single channel single speech source simulation.
    :param use_reverb: whether to apply room impulse response to add reverberation
    :param use_noise: whether to add noise
    :param snr_range: the range of SNR in dB
    :param t60_range: the range to T60 reverberation time in seconds
    :return: a config dictionary
    """

    config = gen_default_config()

    config['sources']['num_spk']['max'] = 1
    config['sources']['num_spk']['min'] = 1

    assert snr_range[1] >= snr_range[0]
    config['global_snr']['min'] = snr_range[0]
    config['global_snr']['max'] = snr_range[1]

    config['dir_noise']['use_dir_noise'] = use_noise
    config['dir_noise']['num_dir_noise']['max'] = 1
    config['dir_noise']['num_dir_noise']['min'] = 1
    config['iso_noise']['use_iso_noise'] = False
    config['iso_noise']['use_corpus'] = False

    config['reverb']['use_reverb'] = use_reverb
    config['reverb']['use_corpus'] = True
    assert t60_range[1] >= t60_range[0]
    config['reverb']['t60']['min'] = t60_range[0]
    config['reverb']['t60']['max'] = t60_range[1]
    
    return config


def single_channel_multi_source_config(n_source_range=[2, 2], spr_range=[-2.5,2.5], use_reverb=True, use_noise=True,
                                       snr_range=[0, 30], t60_range=[0.1, 0.5]):
    """
    get configuration for single channel single speech source simulation.
    :param n_source_range: the range of number of speech sources
    :param use_reverb: whether to apply room impulse response to add reverberation
    :param use_noise: whether to add noise
    :param snr_range: the range of SNR in dB
    :param t60_range: the range to T60 reverberation time in seconds
    :return: a config dictionary
    """

    config = single_channel_single_source_config(use_reverb=use_reverb, use_noise=use_noise,
                                                 snr_range=snr_range, t60_range=t60_range)

    assert n_source_range[1] >= n_source_range[0]
    config['sources']['num_spk']['max'] = n_source_range[1]
    config['sources']['num_spk']['min'] = n_source_range[0]

    assert spr_range[1] >= spr_range[0]
    config['mixing']['spr']['max'] = spr_range[1]
    config['mixing']['spr']['min'] = spr_range[0]

    return config


def multi_channel_single_source_config(mic_positions, use_iso_noise=True, use_dir_noise=True, snr_range=[0, 30],
                                       t60_range=[0.1, 0.5]):
    """
    get configuration for single channel single speech source simulation.
    :param mic_positions: a 2xN or 3xN matrix of microphone coordinates, where N is the number of microphones. The first
    column is for the reference microphone and should have all 0's. If not, the first column will be subtracted from all
    columns.
    :param use_noise: whether to add noise
    :param snr_range: the range of SNR in dB
    :param t60_range: the range to T60 reverberation time in seconds
    :return: a config dictionary
    """

    config = gen_default_config()

    # verify the shape and content of mic_positions
    assert type(mic_positions) is np.ndarray
    assert mic_positions.ndim==2
    assert mic_positions.shape[0]==2 or mic_positions.shape[0]==3
    assert mic_positions.shape[1]>1

    mic_positions_list = [[float(mic_positions[i,j]) for j in range(mic_positions.shape[1])] for i in range(mic_positions.shape[0])]
    config['array']['mic_positions'] = mic_positions_list   # use list representation so we can dump it to yaml

    config['sources']['num_spk']['max'] = 1
    config['sources']['num_spk']['min'] = 1

    assert snr_range[1] >= snr_range[0]
    config['global_snr']['min'] = snr_range[0]
    config['global_snr']['max'] = snr_range[1]

    config['dir_noise']['use_dir_noise'] = use_dir_noise
    config['dir_noise']['num_dir_noise']['max'] = 2
    config['dir_noise']['num_dir_noise']['min'] = 1
    config['iso_noise']['use_iso_noise'] = use_iso_noise
    config['iso_noise']['use_corpus'] = True

    config['reverb']['use_reverb'] = True
    config['reverb']['use_corpus'] = True
    assert t60_range[1] >= t60_range[0]
    config['reverb']['t60']['min'] = snr_range[0]
    config['reverb']['t60']['max'] = snr_range[1]

    return config


def multi_channel_multi_source_config(mic_positions, n_source_range=[2, 2], spr_range=[-2.5,2.5],
                                      use_iso_noise=True, use_dir_noise=True, snr_range=[0, 30], t60_range=[0.1, 0.5]):
    """
    get configuration for single channel single speech source simulation.
    :param mic_positions: a 2xN or 3xN matrix of microphone coordinates, where N is the number of microphones. The first
    column is for the reference microphone and should have all 0's. If not, the first column will be subtracted from all
    columns.
    :param n_source_range: the range of number of speech sources
    :param use_noise: whether to add noise
    :param snr_range: the range of SNR in dB
    :param t60_range: the range to T60 reverberation time in seconds
    :return: a config dictionary
    """

    config = multi_channel_single_source_config(mic_positions, use_iso_noise=use_iso_noise,
                                                use_dir_noise=use_dir_noise, snr_range=snr_range, t60_range=t60_range)

    assert n_source_range[1] >= n_source_range[0]
    config['sources']['num_spk']['max'] = n_source_range[1]
    config['sources']['num_spk']['min'] = n_source_range[0]

    assert spr_range[1] >= spr_range[0]
    config['mixing']['spr']['max'] = spr_range[1]
    config['mixing']['spr']['min'] = spr_range[0]

    return config


if __name__ == "__main__":
    import yaml

    mic_positions = np.asarray([[0, 0.1, 0.2],
                                [0, 0, 0]])  # a linear array

    mcms_config = multi_channel_multi_source_config(mic_positions)
    with open(r'multi_channel_multi_source_config.yaml', 'w') as outfile:
        yaml.dump(mcms_config, outfile, default_flow_style=False)

    scms_config = single_channel_multi_source_config()
    with open(r'single_channel_multi_source_config.yaml', 'w') as outfile:
        yaml.dump(scms_config, outfile, default_flow_style=False)

    mcss_config = multi_channel_single_source_config(mic_positions)
    with open(r'multi_channel_single_source_config.yaml', 'w') as outfile:
        yaml.dump(mcss_config, outfile, default_flow_style=False)

    scss_config = single_channel_single_source_config()
    with open(r'single_channel_single_source_config.yaml', 'w') as outfile:
        yaml.dump(scss_config, outfile, default_flow_style=False)
