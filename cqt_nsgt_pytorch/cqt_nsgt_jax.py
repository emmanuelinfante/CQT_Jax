import jax.numpy as jnp
import jax
import haiku as hk
from jax.ops import index_update

from .fscale import LogScale, FlexLogOctScale
from .nsgfwin import nsgfwin_jax
from .nsdual import nsdual_jax
from .util import calcwinrange_jax, hannwin_jax, blackharr_jax, kaiserwin_jax

class CQTNSGT(hk.Module):
    def __init__(self, numocts, binsoct, mode="critical", window="hann", flex_Q=None, fs=44100, audio_len=44100, dtype=jnp.float32, name=None):
        super().__init__(name=name)
        self.numocts = numocts
        self.binsoct = binsoct
        self.mode = mode
        self.window = window
        self.flex_Q = flex_Q
        self.fs = fs
        self.Ls = audio_len
        self.dtype = dtype

    def __call__(self, x):
        # 1. Calculate frequencies and Q-factors (same as before)
        fmax = self.fs / 2 - 1e-6
        fmin = fmax / (2**self.numocts)
        fbins = int(self.binsoct * self.numocts)

        if self.mode == "flex_oct":
            self.scale = FlexLogOctScale(self.fs, self.numocts, self.binsoct, self.flex_Q)
        else:
            self.scale = LogScale(fmin, fmax, fbins)

        self.frqs, self.q = self.scale()

        # 2. Generate windows and other derived values
        self.g, rfbas, self.M = nsgfwin_jax(self.frqs, self.q, self.fs, self.Ls,
                                           dtype=self.dtype, min_win=4, window=self.window)
        self.wins, self.nn = calcwinrange_jax(self.g, rfbas, self.Ls)
        self.gd = nsdual_jax(self.g, self.wins, self.nn, M=self.M, dtype=self.dtype)
        # 3. Calculate filters for DC and Nyquist
        self.Hlpf = jnp.zeros(self.Ls, dtype=self.dtype)
        self.Hlpf = jax.ops.index_update(self.Hlpf, jax.ops.index[:len(self.g[0]) // 2], 
                                       self.g[0][:len(self.g[0]) // 2] *
                                       self.gd[0][:len(self.g[0]) // 2] *
                                       self.M[0])
        self.Hlpf = jax.ops.index_update(self.Hlpf, jax.ops.index[-len(self.g[0]) // 2:],
                                       self.g[0][len(self.g[0]) // 2:] *
                                       self.gd[0][len(self.g[0]) // 2:] *
                                       self.M[0])

        nyquist_idx = len(self.g) // 2
        Lg = len(self.g[nyquist_idx])
        self.Hlpf = jax.ops.index_add(self.Hlpf, jax.ops.index[self.wins[nyquist_idx][: (Lg + 1) // 2]],
                                       self.g[nyquist_idx][(Lg) // 2:] *
                                       self.gd[nyquist_idx][(Lg) // 2:] *
                                       self.M[nyquist_idx])
        self.Hlpf = jax.ops.index_add(self.Hlpf, jax.ops.index[self.wins[nyquist_idx][-(Lg - 1) // 2:]],
                                       self.g[nyquist_idx][:(Lg) // 2] *
                                       self.gd[nyquist_idx][:(Lg) // 2] *
                                       self.M[nyquist_idx])

        self.Hhpf = 1 - self.Hlpf

        # ... (Implement nsgtf, nsigtf) ...

        c = self.nsgtf(x)
        return c 
        # ... (previous code)

    def nsgtf(self, f):
        ft = jnp.fft.fft(f)
        Ls = f.shape[-1]
        assert self.nn == Ls

        sl = slice(1, len(self.g) // 2) if self.mode in ["matrix", "oct", "matrix_pow2"] else slice(0, len(self.g) // 2 + 1)

        # Pre-calculate for efficiency
        self.maxLg_enc = max(int(jnp.ceil(float(len(gii)) / mii)) * mii
                           for mii, gii in zip(self.M[sl], self.g[sl]))
    def get_ragged_giis_jax(g, wins, ms, mode, dtype=jnp.float32):
    c = jnp.zeros((len(g), len(g[0]) // 2 + 1), dtype=dtype)  # Use len(g[0]) for consistency
    ix = []

    if mode == "oct":
        for i in range(len(ms) // len(g[0])):  # Calculate numocts dynamically
            ix.append(jnp.zeros((len(g[0]), ms[i * len(g[0])]), dtype=int))
    elif mode in ["matrix", "matrix_pow2"]:
        maxLg_enc = max(int(jnp.ceil(float(len(gii)) / mii)) * mii
                       for mii, gii in zip(ms, g))
        ix.append(jnp.zeros((len(g), maxLg_enc), dtype=int))
    elif mode in ["oct_complete", "matrix_complete"]:
        ix.append(jnp.zeros((1, ms[0]), dtype=int))
        count = 0
        for i in range(1, len(g) - 1):
            if count == 0 or ms[i] == ms[i - 1]:
                count += 1
            else:
                ix.append(jnp.zeros((count, ms[i - 1]), dtype=int))
                count = 1
        ix.append(jnp.zeros((count, ms[i - 1]), dtype=int))
        ix.append(jnp.zeros((1, ms[-1]), dtype=int))

    j = 0
    k = 0
    for i, (gii, win_range) in enumerate(zip(g, wins)):
        if i > 0:
            if ms[i] != ms[i - 1] or ((mode == "oct_complete" or mode == "matrix_complete") and (j == 0 or i == len(g) - 1)):
                j += 1
                k = 0

        gii = jnp.fft.fftshift(gii).unsqueeze(0)  # Use jnp.fft.fftshift
        Lg = gii.shape[1]

        if (i == 0 or i == len(g) - 1) and (mode == "oct_complete" or mode == "matrix_complete"):
            if i == 0:
                c = index_update(c, jax.ops.index[i, win_range[Lg // 2:]], gii[..., Lg // 2:])
                ix[j] = index_update(ix[j], jax.ops.index[0, :(Lg + 1) // 2], win_range[Lg // 2:])
                ix[j] = index_update(ix[j], jax.ops.index[0, -(Lg // 2):], jnp.flip(win_range[Lg // 2:]))
            if i == len(g) - 1:
                c = index_update(c, jax.ops.index[i, win_range[:(Lg + 1) // 2]], gii[..., :(Lg + 1) // 2])
                ix[j] = index_update(ix[j], jax.ops.index[0, :(Lg + 1) // 2], jnp.flip(win_range[:(Lg + 1) // 2]))
                ix[j] = index_update(ix[j], jax.ops.index[0, -(Lg // 2):], win_range[:(Lg) // 2])
        else:
            c = index_update(c, jax.ops.index[i, win_range], gii)
            ix[j] = index_update(ix[j], jax.ops.index[k, :(Lg + 1) // 2], win_range[Lg // 2:])
            ix[j] = index_update(ix[j], jax.ops.index[k, -(Lg // 2):], win_range[:Lg // 2])

        k += 1

    return jnp.conj(c), ix

    if self.mode in ["matrix", "matrix_pow2", "matrix_complete"]:
            self.giis, self.idx_enc = get_ragged_giis_jax(self.g[sl], self.wins[sl], self.M[sl], self.mode, dtype=self.dtype)

            ft = jnp.fft.fft(f)[..., :self.Ls // 2 + 1]  # Apply FFT and slice
            t = ft.unsqueeze(-2) * self.giis 
            
            if self.mode == "matrix_complete":
                # Special handling for DC and Nyquist in "matrix_complete" mode
                ret = []
                L = self.idx_enc[0].shape[-1]
                a = jnp.take_along_axis(t[..., 0, :].unsqueeze(-2), self.idx_enc[0].unsqueeze(0).unsqueeze(0), axis=-1)
                a = index_update(a, jax.ops.index[..., (L + 1) // 2:], jnp.conj(a[..., (L + 1) // 2:]))
                ret.append(jnp.fft.ifft(a))

                a = jnp.take_along_axis(t[..., 1:-1, :], self.idx_enc[1].unsqueeze(0).unsqueeze(0), axis=-1)
                ret.append(jnp.fft.ifft(a))

                a = jnp.take_along_axis(t[..., -1, :].unsqueeze(-2), self.idx_enc[-1].unsqueeze(0).unsqueeze(0), axis=-1)
                a = index_update(a, jax.ops.index[..., :(L) // 2], jnp.conj(a[..., :(L) // 2]))
                ret.append(jnp.fft.ifft(a))

                return jnp.concatenate(ret, axis=2)
            else:
                # Common logic for "matrix" and "matrix_pow2" modes
                a = jnp.take_along_axis(t, self.idx_enc[0].unsqueeze(0).unsqueeze(0), axis=-1)
                return jnp.fft.ifft(a)
                elif self.mode == "oct":
            self.giis, self.idx_enc = get_ragged_giis_jax(self.g[sl], self.wins[sl], self.M[sl], self.mode, dtype=self.dtype)
            ft = jnp.fft.fft(f)[..., :self.Ls // 2 + 1]
            t = ft.unsqueeze(-2) * self.giis

            ret = []
            for i in range(len(self.idx_enc)):  # Iterate over octaves
                a = jnp.take_along_axis(t[..., i * self.binsoct:(i + 1) * self.binsoct, :],
                                        self.idx_enc[i].unsqueeze(0).unsqueeze(0), axis=-1)
                ret.append(jnp.fft.ifft(a))

            return ret

        elif self.mode == "oct_complete":
            self.giis, self.idx_enc = get_ragged_giis_jax(self.g[sl], self.wins[sl], self.M[sl], self.mode, dtype=self.dtype)
            ft = jnp.fft.fft(f)[..., :self.Ls // 2 + 1]
            t = ft.unsqueeze(-2) * self.giis

            ret = []
            L = self.idx_enc[0].shape[-1]
            a = jnp.take_along_axis(t[..., 0, :].unsqueeze(-2), self.idx_enc[0].unsqueeze(0).unsqueeze(0), axis=-1)
            a = index_update(a, jax.ops.index[..., (L + 1) // 2:], jnp.conj(a[..., (L + 1) // 2:]))
            ret.append(jnp.fft.ifft(a))

            for i in range(len(self.idx_enc) - 2):  # Iterate over octaves (excluding DC and Nyquist)
                a = jnp.take_along_axis(t[..., i * self.binsoct + 1:(i + 1) * self.binsoct + 1, :],
                                        self.idx_enc[i + 1].unsqueeze(0).unsqueeze(0), axis=-1)
                ret.append(jnp.fft.ifft(a))

            L = self.idx_enc[-1].shape[-1]
            a = jnp.take_along_axis(t[..., -1, :].unsqueeze(-2), self.idx_enc[-1].unsqueeze(0).unsqueeze(0), axis=-1)
            a = index_update(a, jax.ops.index[..., :(L) // 2], jnp.conj(a[..., :(L) // 2]))
            ret.append(jnp.fft.ifft(a))

            return ret
            elif self.mode == "critical":
            self.giis = jnp.conj(jnp.concatenate([jnp.pad(gii.unsqueeze(0), (0, self.maxLg_enc - gii.shape[0])) 
                                                   for gii in self.g[sl]]))
            
            block_ptr = -1
            bucketed_tensors = []
            ret = []

            for j, (mii, win_range, Lg, col) in enumerate(self.loopparams_enc):  # Note: loopparams_enc needs to be defined
                c = jnp.zeros(*f.shape[:2], 1, mii, dtype=ft.dtype)  # ft needs to be defined before the loop

                t = ft[:, :, win_range] * jnp.fft.fftshift(self.giis[j, :Lg])

                sl1 = slice(None, (Lg + 1) // 2)
                sl2 = slice(-(Lg // 2), None)

                c = index_update(c, jax.ops.index[:, :, 0, sl1], t[:, :, Lg // 2:])
                c = index_update(c, jax.ops.index[:, :, 0, sl2], t[:, :, :Lg // 2])

                if block_ptr == -1 or bucketed_tensors[block_ptr][0].shape[-1] != mii:
                    bucketed_tensors.append(c)
                    block_ptr += 1
                else:
                    bucketed_tensors[block_ptr] = jnp.concatenate([bucketed_tensors[block_ptr], c], axis=2)

            # This loop can be optimized with jax.lax.scan
            for bucketed_tensor in bucketed_tensors:
                ret.append(jnp.fft.ifft(bucketed_tensor))

            return ret

        elif self.mode == "matrix_slow":
            c = jnp.zeros(*f.shape[:2], len(self.loopparams_enc), self.maxLg_enc, dtype=jnp.fft.fft(f).dtype)  # loopparams_enc needs to be defined

            # This loop can be optimized with jax.lax.scan
            for j, (mii, win_range, Lg, col) in enumerate(self.loopparams_enc):
                t = jnp.fft.fft(f)[:, :, win_range] * jnp.fft.fftshift(self.giis[j, :Lg])

                sl1 = slice(None, (Lg + 1) // 2)
                sl2 = slice(-(Lg // 2), None)

                c = index_update(c, jax.ops.index[:, :, j, sl1], t[:, :, Lg // 2:])
                c = index_update(c, jax.ops.index[:, :, j, sl2], t[:, :, :Lg // 2])

            return jnp.fft.ifft(c)

        else:
            raise ValueError(f"Invalid mode: {self.mode}")
            # 4. Define loopparams_enc for use in nsgtf
        sl = slice(1, len(self.g) // 2) if self.mode in ["matrix", "oct", "matrix_pow2"] else slice(0, len(self.g) // 2 + 1)
        self.maxLg_enc = max(int(jnp.ceil(float(len(gii)) / mii)) * mii
                           for mii, gii in zip(self.M[sl], self.g[sl]))

        self.loopparams_enc = [(mii, win_range, len(gii), int(jnp.ceil(float(len(gii)) / mii))) 
                               for mii, gii, win_range in zip(self.M[sl], self.g[sl], self.wins[sl])]

        # ... (Implement nsgtf, nsigtf) ...

        c = self.nsgtf(x)
        return c 
elif self.mode == "critical":
            self.giis = jnp.conj(jnp.concatenate([jnp.pad(gii.unsqueeze(0), (0, self.maxLg_enc - gii.shape[0]))
                                                   for gii in self.g[sl]]))

            def critical_loop_body(carry, params):
                j, c, block_ptr, bucketed_tensors = carry
                mii, win_range, Lg, col = params

                t = ft[:, :, win_range] * jnp.fft.fftshift(self.giis[j, :Lg])
                sl1 = slice(None, (Lg + 1) // 2)
                sl2 = slice(-(Lg // 2), None)

                c = index_update(c, jax.ops.index[:, :, 0, sl1], t[:, :, Lg // 2:])
                c = index_update(c, jax.ops.index[:, :, 0, sl2], t[:, :, :Lg // 2])

                cond = (block_ptr == -1) | (bucketed_tensors[block_ptr][0].shape[-1] != mii)
                block_ptr = jax.lax.cond(cond, lambda: block_ptr + 1, lambda: block_ptr)
                bucketed_tensors = jax.lax.cond(cond, lambda: bucketed_tensors + [c],
                                              lambda: index_update(bucketed_tensors, block_ptr,
                                                                  jnp.concatenate([bucketed_tensors[block_ptr], c], axis=2)))

                return (j + 1, c, block_ptr, bucketed_tensors), None

            # Initialize carry and loop parameters
            init_carry = (0, jnp.zeros(*f.shape[:2], 1, self.M[sl][0], dtype=ft.dtype), -1, [])
            _, _, _, bucketed_tensors = jax.lax.scan(critical_loop_body, init_carry, self.loopparams_enc)

            ret = [jnp.fft.ifft(bucketed_tensor) for bucketed_tensor in bucketed_tensors]
            return ret

        # ... (Implement "matrix_slow" mode with jax.lax.scan) ...

        else:
            raise ValueError(f"Invalid mode: {self.mode}")
            def nsigtf(self, cseq):
        if self.mode not in ["matrix", "matrix_slow", "matrix_complete", "matrix_pow2"]:
            assert isinstance(cseq, list)
            nfreqs = sum(cseq_tsor.shape[2] for cseq_tsor in cseq)
            cseq_shape = (*cseq[0].shape[:2], nfreqs)  # Assume all tensors in cseq have the same shape[:2]
            cseq_dtype = cseq[0].dtype
            fc = jnp.concatenate([jnp.fft.fft(cseq_tsor) for cseq_tsor in cseq], axis=2)  # Concatenate along frequency axis
        else:
            assert isinstance(cseq, jnp.ndarray)
            cseq_shape = cseq.shape[:3]
            cseq_dtype = cseq.dtype
            fc = jnp.fft.fft(cseq)

        fbins = cseq_shape[2]
if self.mode == "matrix_slow":
            fr = jnp.zeros(*cseq_shape[:2], self.nn, dtype=cseq_dtype)
            temp0 = jnp.empty(*cseq_shape[:2], self.maxLg_dec, dtype=cseq_dtype)  # maxLg_dec needs to be defined

            def matrix_slow_loop_body(carry, i):
                fr, temp0 = carry
                wr1, wr2, Lg = self.loopparams_dec[i]  # loopparams_dec needs to be defined

                t = fc[:, :, i]  # Assuming fc is defined correctly
                r = (Lg + 1) // 2
                l = (Lg // 2)

                temp0 = index_update(temp0, jax.ops.index[:, :, :r], t[:, :, :r])
                temp0 = index_update(temp0, jax.ops.index[:, :, Lg - l:Lg], t[:, :, self.maxLg_dec - l:self.maxLg_dec])

                temp0 = temp0 * self.gdiis[i, :Lg] * self.maxLg_dec  # gdiis needs to be defined

                fr = index_add(fr, jax.ops.index[:, :, wr1], temp0[:, :, Lg - l:Lg])
                fr = index_add(fr, jax.ops.index[:, :, wr2], temp0[:, :, :r])
                return (fr, temp0), None

            # Initialize carry and loop parameters
            init_carry = (fr, temp0)
            (fr, _), _ = jax.lax.scan(matrix_slow_loop_body, init_carry, jnp.arange(fbins))

            ftr = fr[:, :, :self.nn // 2 + 1]
            sig = jnp.fft.irfft(ftr, n=self.nn)
            sig = sig[:, :, :self.Ls]
            return sig
    # 5. Define loopparams_dec for use in nsigtf
        sl = slice(1, len(self.g) // 2) if self.mode in ["matrix", "oct", "matrix_pow2"] else slice(0, len(self.g) // 2 + 1)
        self.maxLg_dec = max(len(gdii) for gdii in self.gd)
        if self.mode == "matrix_pow2":
            self.maxLg_dec = self.maxLg_enc

        self.loopparams_dec = [(wr1, wr2, len(gdii))
                               for gdii, win_range in zip(self.gd[sl], self.wins[sl])
                               for wr1, wr2 in [(win_range[:(len(gdii)) // 2], win_range[-((len(gdii) + 1) // 2):])]]

        # ... (Implement nsgtf, nsigtf) ...

        c = self.nsgtf(x)
        return c

    def nsigtf(self, cseq):
        # ... (Input handling and FFT - same as before) ...

        if self.mode == "matrix_slow":
            # ... (matrix_slow_loop_body - same as before) ...

            # Initialize carry and loop parameters
            fr = jnp.zeros(*cseq_shape[:2], self.nn, dtype=cseq_dtype)
            temp0 = jnp.empty(*cseq_shape[:2], self.maxLg_dec, dtype=cseq_dtype)
            (fr, _), _ = jax.lax.scan(matrix_slow_loop_body, (fr, temp0), jnp.arange(fbins))
    def get_ragged_gdiis_jax(gd, wins, mode, ms=None, dtype=jnp.float32):
    ragged_gdiis = []
    maxLg_dec = max(len(gdii) for gdii in gd)
    ix = jnp.full((len(gd), len(gd[0]) // 2 + 1), maxLg_dec // 2, dtype=int)  # Initialize with center index

    for i, (g, win_range) in enumerate(zip(gd, wins)):
        Lg = len(g)
        gl = g[:(Lg + 1) // 2]
        gr = g[(Lg + 1) // 2:]
        zeros = jnp.zeros(maxLg_dec - Lg, dtype=g.dtype)
        paddedg = jnp.concatenate((gl, zeros, gr), axis=0).unsqueeze(0) * maxLg_dec
        ragged_gdiis.append(paddedg)

        wr1 = win_range[:(Lg) // 2]
        wr2 = win_range[-((Lg + 1) // 2):]

        if mode == "matrix_complete" and i == 0:
            ix = index_update(ix, jax.ops.index[i, wr2], jnp.arange(len(wr2)))
        elif mode == "matrix_complete" and i == len(gd) - 1:
            ix = index_update(ix, jax.ops.index[i, wr1], maxLg_dec - (Lg // 2) + jnp.arange(len(wr1)))
        else:
            ix = index_update(ix, jax.ops.index[i, wr1], maxLg_dec - (Lg // 2) + jnp.arange(len(wr1)))
            ix = index_update(ix, jax.ops.index[i, wr2], jnp.arange(len(wr2)))

    return jnp.conj(jnp.concatenate(ragged_gdiis, axis=0)).astype(dtype), ix
        elif self.mode in ["matrix", "matrix_complete", "matrix_pow2"]:
            sl = slice(1, len(self.g) // 2) if self.mode in ["matrix", "matrix_pow2"] else slice(0, len(self.g) // 2 + 1)
            self.gdiis, self.idx_dec = get_ragged_gdiis_jax(self.gd[sl], self.wins[sl], self.mode, ms=self.M[sl], dtype=self.dtype)

            fr = jnp.zeros(*cseq_shape[:2], self.nn // 2 + 1, dtype=cseq_dtype)
            temp0 = fc * self.gdiis.unsqueeze(0).unsqueeze(0)
            fr = jnp.sum(jnp.take_along_axis(temp0, self.idx_dec.unsqueeze(0).unsqueeze(0), axis=3), axis=2)

            ftr = fr
            sig = jnp.fft.irfft(ftr, n=self.nn)
            sig = sig[:, :, :self.Ls]
            return sig
        
