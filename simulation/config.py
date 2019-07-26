import numpy as np
from ._sampling import get_distribution_template
from ._mixer import MixerConfig
from ._geometry import SoundSourceConfig, ArrayPositionConfig, RoomConfig
from ._iso_noise_simulator import ISONoiseConfig


def _gen_stft_config():
    config = {}
    config['fs'] = 16000
    config['frame_len'] = 400
    config['frame_shift'] = 160
    config['fft_size'] = 512
    config['window'] = 'hamming'
    config['dc_removal'] = False
    config['do_dither'] = True
    return config


def _gen_default_simu_config(mic_positions=[[0],[0],[0]]):
    """
    Generate configuration for room simulation with default values.
    Users should not call this function directly. See the following configuration functions.
    """

    config = {}

    config['analysis'] = _gen_stft_config()

    config['num_simulated'] = get_distribution_template('distribution of how many sentences to simulate from each pair of sources', max=1,
                                                  min=1, distribution='uniform_int')
    config['global_snr'] = get_distribution_template('distribution of global signal-to-noise ratio (SNR)', max=30,
                                              min=0, distribution='uniform')
    # define the distribution of the number of speaker sources and SNRs
    config['sources'] = SoundSourceConfig().config

    config['dir_noise'] = {}
    config['dir_noise']['use_dir_noise'] = True
    config['dir_noise']['snr'] = get_distribution_template('distribution of the SNR of directional noises w.r.t. total reverberant signal power', max=20,
                                              min=0, distribution='uniform')
    config['dir_noise']['num_dir_noise'] = get_distribution_template('distribution of how many directional noise sources to use in the simulation',
                                                  max=4, min=1, distribution='uniform_int')

    config['mixing'] = MixerConfig().config

    config['iso_noise'] = ISONoiseConfig().config

    config['reverb'] = {}
    config['reverb']['use_reverb'] = True
    config['reverb']['use_corpus'] = False
    config['reverb']['t60threshold_for_online_rir'] = 0.3   # if T60 is smaller than the threshold, use online rir generation
    config['reverb']['t60'] = get_distribution_template('distribution of T60 reverberation time', max=1.0,
                                              min=0.1, distribution='uniform')
    config['reverb']['max_rir_length'] = 8000

    # define geometries
    config['array'] = ArrayPositionConfig(mic_positions).config
    config['room'] = RoomConfig().config

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

    config = _gen_default_simu_config()

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

    config = _gen_default_simu_config()

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


def _dump_config2file():
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
