import numpy as np
import scipy.interpolate as interp
from . import _sampling


# Hoth Noise Specifications
# For details, see p. 80 of
# http://studylib.net/doc/18787871/ieee-std-269-2001-draft-standard-methods-for-measuring
_hoth_freqs = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000]
_hoth_mag_db = [32.4, 30.9, 29.1, 27.6, 26, 24.4, 22.7, 21.1, 19.5, 17.8, 16.2, 14.6, 12.9, 11.3, 9.6, 7.8, 5.4, 2.6, -1.3, -6.6]
_hoth_index_1000_hz = 10
_hoth_index_4000_hz = 16
_speed_of_sound = 340 #m/s


class ISONoiseConfig:
    def __init__(self):
        self.config = dict()
        self.config['use_iso_noise'] = True
        self.config['use_corpus'] = False
        self.config['type'] = 'sph'  # [sph|cyl]
        self.config['spectrum_type'] = 'hoth'  # [white|hoth]
        self.config['snr'] = _sampling.get_distribution_template(
            'distribution of the SNR of isotropic noises w.r.t. unit variance signal', max=30,
            min=10, distribution='uniform')


def _sample_sphere(num_points):
    # theta = elevation
    # phi = azimuth
    radius=1
    theta = np.zeros([num_points])
    phi = np.zeros([num_points])
    for k in range(0,num_points,1):
        h = -1 + 2*k/(num_points-1)
        phi[k] = np.arccos(h)
        if k==0 or k==num_points-1:
            theta[k] = 0
        else:
            theta[k] = np.mod(theta[k-1] + 3.6 / np.sqrt(num_points*(1-h*h)), 2*np.pi)

    loc_xyz=np.zeros([3,len(theta)*len(phi)])
    for k in range(0, num_points, 1):
        p = phi[k]
        t = theta[k]
        x = radius*np.sin(p)*np.cos(t)
        y = radius*np.sin(p)*np.sin(t)
        z = radius*np.cos(p)
        loc_xyz[:,k] = [x,y,z]
    return loc_xyz


def _get_hoth_mag(samp_rate, fft_size):

    fft_size_by_2 = int(fft_size / 2)
    hoth_mag = np.asarray(_hoth_mag_db) - _hoth_mag_db[_hoth_index_1000_hz]
    hoth_mag = np.power(10, hoth_mag/20)
    hoth_w = 2 * np.pi * np.asarray(_hoth_freqs)/samp_rate

    if (samp_rate == 16000):
        f = interp.interp1d(hoth_w, hoth_mag, kind='cubic', bounds_error=False,
                            fill_value=(hoth_mag[0], hoth_mag[-1]))
    elif (samp_rate == 8000):
        f = interp.interp1d(hoth_w[0:_hoth_index_4000_hz + 1], hoth_mag[0:_hoth_index_4000_hz + 1], kind='cubic', bounds_error=False,
                            fill_value=(hoth_mag[0], hoth_mag[_hoth_index_4000_hz + 1]))
    else:
        RuntimeError('Can only generate Hoth noise for 16000 sampling rates!')

    w = 2 * np.pi * np.arange(0, fft_size_by_2 + 1, 1) / fft_size

    hoth_mag_interp = f(w)
    hoth_mag_interp[0] = 0 # skip DC (0 Hz)

    #plt.plot(hoth_w, 20*np.log10(hoth_mag))
    #plt.plot(w, 20*np.log10(hoth_mag_interp))
    #plt.show()
    # add DC
    #hoth_mag_interp = np.insert(hoth_mag_interp, 0, 0)
    return hoth_mag_interp


def _sample_circle(num_points):
    # phi = azimuth
    radius = 1
    phi = 2*np.pi*np.arange(0,1,1/num_points)
    loc_xyz = np.zeros([3, len(phi)])
    for k in range(0, num_points, 1):
        p = phi[k]
        x = radius * np.cos(p)
        y = radius * np.sin(p)
        z = 0
        loc_xyz[:,k] = [x,y,z]

    return loc_xyz


def generate_isotropic_noise(mic_xyz, N, samp_rate, type='sph', spectrum='hoth'):
    """
     Generate isotropic noise for microphone array.

     Implemented by Mike seltzer and followed:
         E.A.P. Habets and S. Gannot, Generating sensor signals in isotropic noise fields,
         Journal of the Acoustical Society of America, Vol. 122, Issue 6, pp. 3464-3470, Dec. 2007.
     Added the spectral shaping to "Hoth" noise profile

     mic_xyz: a Cx3 matrix that contains the 3D coordinates of microphones.
     N: number of samples to generate
     samp_rate: sampling rate
     type: type of isotropic noise, select from ['sph'|'cyl']
     spectrum: whether to addd spectral shaping to noise, select from ['white'|'hoth']
    """
    num_points = 512
    num_mics = mic_xyz.shape[0]
    fft_size = int(2 ** np.ceil(np.log2(N)))
    fft_size_by_2 = int(fft_size/2)

    # calculate relative microphone positions wrt mic 1
    P_rel = np.zeros([num_mics,3])
    for m in range(0,num_mics,1):
        P_rel[m,:] = mic_xyz[m,:] - mic_xyz[0,:]

    # get locations uniformly sampled on a sphere
    if (type == 'sph'):
        loc_xyz = _sample_sphere(num_points)
    elif (type == 'cyl'):
        loc_xyz = _sample_circle(num_points)
    else:
        RuntimeError('type must be \'sph\' or \'cyl\'')

    if (spectrum == 'white'):
        g = 1
    elif (spectrum == 'hoth'):
        g = _get_hoth_mag(samp_rate, fft_size)
    else:
        RuntimeError('spectrum must be \'white\' or \'hoth\'')

    # for each point, generate random noise in frequency domain and multiply by the steering vector
    w = 2*np.pi*np.arange(0,fft_size_by_2+1,1)/fft_size
    X = np.zeros([num_mics,fft_size_by_2 + 1], dtype=complex)

    for i in range(0,num_points,1):
        X_this = g * (np.random.normal(0,1,fft_size_by_2+1) + 1j*np.random.normal(0,1,fft_size_by_2+1))
        X[0,:] = X[0,:] + X_this
        for m in range(1,num_mics,1):
            delta = np.sum(P_rel[m,:]*loc_xyz[:,i])
            tau = delta * samp_rate / _speed_of_sound
            X[m,:] = X[m,:] + X_this*np.exp(-1j*tau*w)

    X = X/np.sqrt(num_points)

    # transform to time domain
    X[:, 0] = np.sqrt(fft_size) * np.real(X[:, 0])
    X[:, fft_size_by_2] = np.sqrt(fft_size) * np.real(X[:, fft_size_by_2])
    X[:, 1:fft_size_by_2] = np.sqrt(fft_size_by_2) * X[:, 1:fft_size_by_2]

    n = np.fft.irfft(X, fft_size, axis=1)

    n = n[:,0:N]

    return n


def _test_iso_noise():
    mic_xyz = np.asarray([[0, 0, 0], [0, 1, 0]])
    result = generate_isotropic_noise(mic_xyz, N=16000, samp_rate=16000, type='sph', spectrum='white')
    import matplotlib.pyplot as plt
    plt.plot(result.T)


#_test_iso_noise()