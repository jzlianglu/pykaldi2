import numpy as np
# import cupy

def t60_to_alpha(room, t60):
    V = np.prod(room)
    c = 343
    S = 2 * (room[0] * room[2] + room[1] * room[2] + room[0] * room[1])
    alpha = 24 * V * np.log(10) / (c * S * t60)
    return alpha

def min_t60_of_room(room):
    V = np.prod(room)
    c = 343
    S = 2 * (room[0] * room[2] + room[1] * room[2] + room[0] * room[1])
    min_t60 = 24 * V * np.log(10) / (c * S)
    return min_t60 * 1.1

def xp_rirgen(room, source_loc, mic_loc, c=340, fs=16000, t60=0.5,
              beta=None, nsamples=None, htw=None, hpfilt=True, habets_compat=False, method=1):
    """Generates room impulse responses corresponding to each source-microphone pair placed in a room.

    Args:
        room (numpy/cupy array) = room dimensions in meters, shape: (3, 1)
        source_loc (numpy/cupy array) = source locations in meters, shape: (3, nsrc)
        mic_loc (numpy/cupy array) = microphone locations in meters, shape: (3, nmic)
        kwargs:
            c (float) = speed of sound in meters/second (default: 340)
            fs (float) = sampling rate in Hz (default: 16000)
            t60 (float) = t60 or rt60 in seconds or None to use beta parameters (default: 0.5)
            beta (numpy/cupy array) = beta parameters of reflections for each side, shape (6,1) (default: None)
            nsamples (int) = number of output samples (default: auto from t60)
            htw (int) = half size in samples of the time window used for sinc function interpolation (default automatic)
            hpfilt (bool) = use post-generation highpass filter or not (default True)
            method (int) = 1 or 2, 2 is not tested thoroughly and is very slow, so use 1 always (default 1)

    Returns:
        room impulse responses in time-domain of shape (nsrc, nmic, nsamples)

    Notes:
        1. If input arrays are cupy arrays (on GPU), the code runs with cupy, otherwise with numpy
        2. if you do not want to install cupy or not interested in GPU processing,
            remove line "import cupy" and replace "xp=cupy.get..." with "xp=np"

    .. seealso:: :func:`pyrirgen.RirGenerator`
    .. seealso:: :url:https://github.com/ehabets/RIR-Generator/blob/master/rir_generator.cpp

    >>> ### DOCTEST ###
    >>> room = np.array([4,7,3]).reshape(3,1)
    >>> source_loc = np.random.uniform(0,1,(3,2)) * room
    >>> mic_loc =  np.random.uniform(0,1,(3,4)) * room
    >>> t60=0.3
    >>> rirs_np = xp_rirgen(room, source_loc, mic_loc, t60=t60)
    >>> #import matplotlib.pyplot as plt
    >>> #plt.plot(rirs_np[0,0,:] , label='rir for src1 and mic1')
    >>> croom = cupy.array(room)
    >>> csource_loc = cupy.array(source_loc)
    >>> cmic_loc =  cupy.array(mic_loc)
    >>> rirs_cp = xp_rirgen(croom, csource_loc, cmic_loc, t60=t60)
    >>> cupy.testing.assert_allclose(rirs_np, cupy.asnumpy(rirs_cp), atol=1e-5, rtol=1e-5)
    >>> beta = np.random.uniform(0.1, 0.9, size=6)
    >>> rirs_np = xp_rirgen(room, source_loc, mic_loc, beta=beta, t60=None)
    >>> cbeta = cupy.array(beta)
    >>> rirs_cp = xp_rirgen(croom, csource_loc, cmic_loc, beta=cbeta, t60=None)
    >>> cupy.testing.assert_allclose(rirs_np, cupy.asnumpy(rirs_cp), atol=1e-5, rtol=1e-5)
    >>> rirs_np = xp_rirgen(room, source_loc, mic_loc, t60=t60, habets_compat=True)
    """
#     xp = cupy.get_array_module(room, source_loc, mic_loc, beta)
    xp=np
    if beta is None and t60 is None:
        raise Exception('Either t60 or beta array must be provided')
    elif beta is None:
        V = xp.prod(room)
        S = 2 * (room[0] * room[2] + room[1] * room[2] + room[0] * room[1])
        alpha = 24 * V * xp.log(10) / (c * S * t60)
        if alpha < 1:
            beta = xp.ones(6, ) * xp.sqrt(1 - alpha)
        else:
            raise Exception('t60 value {} too small for the room'.format(t60))
    else:
        if xp.max(beta) >= 1.0 or xp.min(beta) <= 0.0:
            raise Exception('beta array values should be in the interval (0,1).')
        if t60 is not None:
            print('Overwriting provided t60 value using provided beta array')
        alpha = 1 - beta**2
        V = xp.prod(room)
        Se = 2 * (room[1] * room[2] * (alpha[0] + alpha[1]) + room[0] * room[2] * (alpha[2] + alpha[3]) + room[0] * room[1] * (alpha[4] + alpha[5]))
        t60 = 24 * xp.log(10.0) * V / (c * Se);

    if htw is None:
        htw = np.minimum(32, int(xp.min(room) / 10 / c * fs))
    if habets_compat:
        htw = 64
    tw_idx = xp.arange(0, 2 * htw).reshape(2 * htw, 1)
    try:
        assert(xp.all(room.T - mic_loc.T > 0) and xp.all(room.T - source_loc.T > 0))
        assert(xp.all(mic_loc.T > 0) and xp.all(source_loc.T > 0))
    except:
        raise Exception('Room dimensions and source and mic locations are not compatible.')
    cTs = c / fs
    # convert distances in meters to time-delays in samples
    room = room / cTs
    mic_loc = mic_loc / cTs
    src_loc = source_loc / cTs
    nmic = mic_loc.shape[-1]
    nsrc = source_loc.shape[-1]
    if nsamples is None:
        nsamples = int(fs * t60)

    def get_reflection_candidates():
        nxrefl = int(nsamples / (room[0]))
        nyrefl = int(nsamples / (room[1]))
        nzrefl = int(nsamples / (room[2]))
        xro = xp.arange(-nxrefl, nxrefl + 1)
        yro = xp.arange(-nyrefl, nyrefl + 1)
        zro = xp.arange(-nzrefl, nzrefl + 1)
        xr = xro.reshape(2 * nxrefl + 1, 1, 1)
        yr = yro.reshape(1, 2 * nyrefl + 1, 1)
        zr = zro.reshape(1, 1, 2 * nzrefl + 1)
        RoughDelays = xp.sqrt((2 * xr * room[0]) ** 2 +                               (2 * yr * room[1]) ** 2 +                               (2 * zr * room[2]) ** 2)
        RoughGains = (beta[0] * beta[1]) ** xp.abs(xr) *                      (beta[2] * beta[3]) ** xp.abs(yr) *                      (beta[4] * beta[5]) ** xp.abs(zr) / (
                     RoughDelays + 0.5 / c * fs)  # assume src-mic distance at least .5 metres
        maxgain = xp.max(RoughGains)
        vreflidx = xp.vstack(xp.nonzero(xp.logical_and(RoughDelays < nsamples, RoughGains > maxgain / 1.0e4)))
        nrefl = vreflidx.shape[-1]
        reflidx = xp.arange(nrefl).reshape(1, 1, nrefl, 1, 1, 1)

        xrefl = xro[vreflidx[..., reflidx][0]]
        yrefl = yro[vreflidx[..., reflidx][1]]
        zrefl = zro[vreflidx[..., reflidx][2]]
        return xrefl, yrefl, zrefl

    xrefl, yrefl, zrefl = get_reflection_candidates()

    def get_delays_and_gains():
        xside = xp.arange(0, 2).reshape(1, 1, 1, 2, 1, 1)
        yside = xp.arange(0, 2).reshape(1, 1, 1, 1, 2, 1)
        zside = xp.arange(0, 2).reshape(1, 1, 1, 1, 1, 2)
        imic = xp.arange(nmic).reshape(1, nmic, 1, 1, 1, 1)
        isrc = xp.arange(nsrc).reshape(nsrc, 1, 1, 1, 1, 1)
        Delays = xp.sqrt((2 * xrefl * room[0] - mic_loc[0, imic] + (1 - 2 * xside) * src_loc[0, isrc]) ** 2 +                          (2 * yrefl * room[1] - mic_loc[1, imic] + (1 - 2 * yside) * src_loc[1, isrc]) ** 2 +                          (2 * zrefl * room[2] - mic_loc[2, imic] + (1 - 2 * zside) * src_loc[2, isrc]) ** 2)
        Refl_x = beta[0] ** (xp.abs(xrefl - xside)) * beta[1] ** (xp.abs(xrefl))
        Refl_y = beta[2] ** (xp.abs(yrefl - yside)) * beta[3] ** (xp.abs(yrefl))
        Refl_z = beta[4] ** (xp.abs(zrefl - zside)) * beta[5] ** (xp.abs(zrefl))
        Gains = Refl_x * Refl_y * Refl_z / (4 * np.pi * Delays * cTs)
        # Gains[Delays > nsamples] = 0.0
        return Delays, Gains

    Delays, Gains = get_delays_and_gains()

    rirs = xp.zeros((nsrc, nmic, nsamples), dtype=np.float32)
    for src in xp.arange(nsrc):
        for mic in xp.arange(nmic):
            dnow = Delays[src, mic, ...].flatten()
            gnow = Gains[src, mic, ...].flatten()
            if method == 1:
                gnow = gnow[dnow < nsamples - htw - 2]
                dnow = dnow[dnow < nsamples - htw - 2]

                dnow_floor = xp.floor(dnow)
                dnow_dist = dnow - dnow_floor
                dnow_floor = dnow_floor.reshape(1, dnow.shape[0])
                dnow_dist = dnow_dist.reshape(1, dnow.shape[0])
                gnow = gnow.reshape(1, dnow.shape[0])
                dnow_ext = dnow_floor + tw_idx - htw + 1
                garg = np.pi * (-dnow_dist + 1 + tw_idx - htw)
                gnow_ext = gnow * 0.5 * (1.0 - xp.cos(np.pi + garg / htw)) * xp.where(garg == 0.0, 1.0, xp.sin(garg) / garg)
                dnow = dnow_ext.flatten().astype(np.int32)
                gnow = gnow_ext.flatten().astype(np.float32)
                dvalid = xp.logical_and(dnow >= 0, dnow < nsamples)
                gnow = gnow[dvalid]
                dnow = dnow[dvalid]
                rirnow = xp.zeros((nsamples,), dtype=np.float32)
                if xp == np:
                    np.add.at(rirnow, dnow, gnow)
                else:
                    xp.scatter_add(rirnow, dnow, gnow)
                rirs[src, mic, ...] = rirnow
            elif method == 2:  ## this is too slow and may not be accurate as well
                gnow = gnow[dnow < nsamples]
                dnow = dnow[dnow < nsamples]
                frange = xp.arange(0, 0.5 + 0.5 / nsamples, 1.0 / nsamples)
                rirfft = xp.zeros(frange.shape, dtype=np.complex128)
                for i in range(len(frange)):
                    rirfft[i] = xp.sum(gnow * xp.exp(-1j * 2 * np.pi * frange[i] * dnow))
                rirs[src, mic, :] = xp.real(xp.fft.irfft(rirfft)).astype(dtype=np.float32)

    if habets_compat:
        if xp is np:
            import scipy.signal
            W = 2*np.pi*100/fs
            R1 = np.exp(-W)
            B1 = 2*R1*np.cos(W)
            B2 = -R1 * R1
            A1 = -(1+R1)
            a = np.array([1, -B1, -B2])
            b = np.array([1, A1, R1])
            rirs = scipy.signal.lfilter(b, a, rirs, axis=-1)
        else:
            raise Exception('habets_compat not available for cupy')
    elif hpfilt:
        rirs[:, :, 1:-1] += -0.5 * rirs[:, :, 2:] -0.5 * rirs[:, : , :-2]


    return rirs


def xp_rirgen2(room, source_loc, mic_loc, c=340, fs=16000, t60=0.5,
              beta=None, nsamples=None, htw=None, hpfilt=True, method=1):
    """Generates room impulse responses corresponding to each source-microphone pair placed in a room.

    Args:
        room (numpy/cupy array) = room dimensions in meters, shape: (3, 1)
        source_loc (numpy/cupy array) = source locations in meters, shape: (3, nsrc)
        mic_loc (numpy/cupy array) = microphone locations in meters, shape: (3, nmic)
        kwargs:
            c (float) = speed of sound in meters/second (default: 340)
            fs (float) = sampling rate in Hz (default: 16000)
            t60 (float) = t60 or rt60 in seconds or None to use beta parameters (default: 0.5)
            beta (numpy/cupy array) = beta parameters of reflections for each side, shape (6,1) (default: None)
            nsamples (int) = number of output samples (default: auto from t60)
            htw (int) = half size in samples of the time window used for sinc function interpolation (default automatic)
            hpfilt (bool) = use post-generation highpass filter or not (default True)
            method (int) = 1 or 2, 2 is not tested thoroughly and is very slow, so use 1 always (default 1)

    Returns:
        room impulse responses in time-domain of shape (nsrc, nmic, nsamples)

    Notes:
        1. If input arrays are cupy arrays (on GPU), the code runs with cupy, otherwise with numpy
        2. if you do not want to install cupy or not interested in GPU processing,
            remove line "import cupy" and replace "xp=cupy.get..." with "xp=np"

    .. seealso:: :func:`pyrirgen.RirGenerator`
    .. seealso:: :url:https://github.com/ehabets/RIR-Generator/blob/master/rir_generator.cpp

    >>> ### DOCTEST ###
    >>> room = np.array([4,7,3]).reshape(3,1)
    >>> source_loc = np.random.uniform(0,1,(3,2)) * room
    >>> mic_loc =  np.random.uniform(0,1,(3,4)) * room
    >>> t60=0.3
    >>> rirs_np = xp_rirgen(room, source_loc, mic_loc, t60=t60)
    >>> #import matplotlib.pyplot as plt
    >>> #plt.plot(rirs_np[0,0,:] , label='rir for src1 and mic1')
    >>> croom = cupy.array(room)
    >>> csource_loc = cupy.array(source_loc)
    >>> cmic_loc =  cupy.array(mic_loc)
    >>> rirs_cp = xp_rirgen(croom, csource_loc, cmic_loc, t60=t60)
    >>> cupy.testing.assert_allclose(rirs_np, cupy.asnumpy(rirs_cp), atol=1e-5, rtol=1e-5)
    >>> beta = np.random.uniform(0.1, 0.9, size=6)
    >>> rirs_np = xp_rirgen(room, source_loc, mic_loc, beta=beta, t60=None)
    >>> cbeta = cupy.array(beta)
    >>> rirs_cp = xp_rirgen(croom, csource_loc, cmic_loc, beta=cbeta, t60=None)
    >>> cupy.testing.assert_allclose(rirs_np, cupy.asnumpy(rirs_cp), atol=1e-5, rtol=1e-5)
    """
#     xp = cupy.get_array_module(room, source_loc, mic_loc, beta)
    xp
    if beta is None and t60 is None:
        raise Exception('Either t60 or beta array must be provided')
    elif beta is None:
        V = xp.prod(room)
        S = 2 * (room[0] * room[2] + room[1] * room[2] + room[0] * room[1])
        alpha = 24 * V * xp.log(10) / (c * S * t60)
        if alpha < 1:
            beta = xp.ones(6, ) * xp.sqrt(1 - alpha)
        else:
            raise Exception('t60 value {} too small for the room'.format(t60))
    else:
        if xp.max(beta) >= 1.0 or xp.min(beta) <= 0.0:
            raise Exception('beta array values should be in the interval (0,1).')
        if t60 is not None:
            print('Overwriting provided t60 value using provided beta array')
        alpha = 1 - beta**2
        V = xp.prod(room)
        Se = 2 * (room[1] * room[2] * (alpha[0] + alpha[1]) + room[0] * room[2] * (alpha[2] + alpha[3]) + room[0] * room[1] * (alpha[4] + alpha[5]))
        t60 = 24 * xp.log(10.0) * V / (c * Se);

    if htw is None:
        htw = np.minimum(32, int(xp.min(room) / 10 / c * fs))
    tw_idx = xp.arange(0, 2 * htw).reshape(2 * htw, 1)
    try:
        assert(xp.all(room.T - mic_loc.T > 0) and xp.all(room.T - source_loc.T > 0))
        assert(xp.all(mic_loc.T > 0) and xp.all(source_loc.T > 0))
    except:
        raise Exception('Room dimensions and source and mic locations are not compatible.')
    cTs = c / fs
    # convert distances in meters to time-delays in samples
    room = room / cTs
    mic_loc = mic_loc / cTs
    src_loc = source_loc / cTs
    nmic = mic_loc.shape[-1]
    nsrc = source_loc.shape[-1]
    if nsamples is None:
        nsamples = int(fs * t60)

    def get_reflection_candidates():
        nxrefl = int(nsamples / (room[0]))
        nyrefl = int(nsamples / (room[1]))
        nzrefl = int(nsamples / (room[2]))
        xro = xp.arange(-nxrefl, nxrefl + 1)
        yro = xp.arange(-nyrefl, nyrefl + 1)
        zro = xp.arange(-nzrefl, nzrefl + 1)
        xr = xro.reshape(2 * nxrefl + 1, 1, 1)
        yr = yro.reshape(1, 2 * nyrefl + 1, 1)
        zr = zro.reshape(1, 1, 2 * nzrefl + 1)
        RoughDelays = xp.sqrt((2 * xr * room[0]) ** 2 +                               (2 * yr * room[1]) ** 2 +                               (2 * zr * room[2]) ** 2)
        RoughGains = (beta[0] * beta[1]) ** xp.abs(xr) *                      (beta[2] * beta[3]) ** xp.abs(yr) *                      (beta[4] * beta[5]) ** xp.abs(zr) / (
                     RoughDelays + 0.5 / c * fs)  # assume src-mic distance at least .5 metres
        maxgain = xp.max(RoughGains)
        vreflidx = xp.vstack(xp.nonzero(xp.logical_and(RoughDelays < nsamples, RoughGains > maxgain / 1.0e4)))
        nrefl = vreflidx.shape[-1]
        reflidx = xp.arange(nrefl).reshape(1, 1, nrefl, 1, 1, 1)

        xrefl = xro[vreflidx[..., reflidx][0]]
        yrefl = yro[vreflidx[..., reflidx][1]]
        zrefl = zro[vreflidx[..., reflidx][2]]
        return xrefl, yrefl, zrefl

    xrefl, yrefl, zrefl = get_reflection_candidates()

    def get_delays_and_gains():
        xside = xp.arange(0, 2).reshape(1, 1, 1, 2, 1, 1)
        yside = xp.arange(0, 2).reshape(1, 1, 1, 1, 2, 1)
        zside = xp.arange(0, 2).reshape(1, 1, 1, 1, 1, 2)
        imic = xp.arange(nmic).reshape(1, nmic, 1, 1, 1, 1)
        isrc = xp.arange(nsrc).reshape(nsrc, 1, 1, 1, 1, 1)
        Delays = xp.sqrt((2 * xrefl * room[0] - mic_loc[0, imic] + (1 - 2 * xside) * src_loc[0, isrc]) ** 2 +                          (2 * yrefl * room[1] - mic_loc[1, imic] + (1 - 2 * yside) * src_loc[1, isrc]) ** 2 +                          (2 * zrefl * room[2] - mic_loc[2, imic] + (1 - 2 * zside) * src_loc[2, isrc]) ** 2)
        Refl_x = beta[0] ** (xp.abs(xrefl - xside)) * beta[1] ** (xp.abs(xrefl))
        Refl_y = beta[2] ** (xp.abs(yrefl - yside)) * beta[3] ** (xp.abs(yrefl))
        Refl_z = beta[4] ** (xp.abs(zrefl - zside)) * beta[5] ** (xp.abs(zrefl))
        Gains = Refl_x * Refl_y * Refl_z / (4 * np.pi * Delays * cTs)
        # Gains[Delays > nsamples] = 0.0
        return Delays, Gains

    Delays, Gains = get_delays_and_gains()

    rirs = xp.zeros((nsrc, nmic, nsamples), dtype=np.float32)
    for src in xp.arange(nsrc):
        for mic in xp.arange(nmic):
            dnow = Delays[src, mic, ...].flatten()
            gnow = Gains[src, mic, ...].flatten()
            if method == 1:
                gnow = gnow[dnow < nsamples - htw - 2]
                dnow = dnow[dnow < nsamples - htw - 2]

                dnow_floor = xp.floor(dnow)
                dnow_dist = dnow - dnow_floor
                dnow_floor = dnow_floor.reshape(1, dnow.shape[0])
                dnow_dist = dnow_dist.reshape(1, dnow.shape[0])
                gnow = gnow.reshape(1, dnow.shape[0])
                dnow_ext = dnow_floor + tw_idx - htw + 1
                garg = np.pi * (-dnow_dist + 1 + tw_idx - htw)
                gnow_ext = gnow * 0.5 * (1.0 - xp.cos(np.pi + garg / htw)) * xp.where(garg == 0.0, 1.0, xp.sin(garg) / garg)
                dnow = dnow_ext.flatten().astype(np.uint32)
                gnow = gnow_ext.flatten().astype(np.float32)
                rirnow = xp.zeros((nsamples,), dtype=np.float32)
                if xp == np:
                    np.add.at(rirnow, dnow, gnow)
                else:
                    xp.scatter_add(rirnow, dnow, gnow)
                rirs[src, mic, ...] = rirnow
            elif method == 2:  ## this is too slow and may not be accurate as well
                gnow = gnow[dnow < nsamples]
                dnow = dnow[dnow < nsamples]
                frange = xp.arange(0, 0.5 + 0.5 / nsamples, 1.0 / nsamples)
                rirfft = xp.zeros(frange.shape, dtype=np.complex128)
                for i in range(len(frange)):
                    rirfft[i] = xp.sum(gnow * xp.exp(-1j * 2 * np.pi * frange[i] * dnow))
                rirs[src, mic, :] = xp.real(xp.fft.irfft(rirfft)).astype(dtype=np.float32)
    if hpfilt:
        rirs[:, :, 1:-1] += -0.5 * rirs[:, :, 2:] -0.5 * rirs[:, : , :-2]

    return rirs
