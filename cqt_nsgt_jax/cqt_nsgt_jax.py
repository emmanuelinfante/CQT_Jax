import jax.numpy as jnp
import jax
import haiku as hk

from .fscale_jax import LogScale, FlexLogOctScale
from .nsgfwin_jax import nsgfwin_jax
from .nsdual_jax import nsdual_jax
from .util_jax import calcwinrange_jax, hannwin_jax, blackharr_jax, kaiserwin_jax

def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

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
        fmax = self.fs / 2 - 1e-6
        fmin = fmax / (2**self.numocts)
        fbins = int(self.binsoct * self.numocts)

        if self.mode == "flex_oct":
            self.scale = FlexLogOctScale(self.fs, self.numocts, self.binsoct, self.flex_Q)
        else:
            self.scale = LogScale(fmin, fmax, fbins)

        self.frqs, self.q = self.scale()

        self.g, rfbas, self.M = nsgfwin_jax(self.frqs, self.q, self.fs, self.Ls,
                                           dtype=self.dtype, min_win=4, window=self.window)
        self.wins, self.nn = calcwinrange_jax(self.g, rfbas, self.Ls)
        self.gd = nsdual_jax(self.g, self.wins, self.nn, M=self.M, dtype=self.dtype)

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

        sl = slice(1, len(self.g) // 2) if self.mode in ["matrix", "oct", "matrix_pow2"] else slice(0, len(self.g) // 2 + 1)
        self.maxLg_enc = max(int(jnp.ceil(float(len(gii)) / mii)) * mii
                           for mii, gii in zip(self.M[sl], self.g[sl]))

        self.loopparams_enc = [(mii, win_range, len(gii), int(jnp.ceil(float(len(gii)) / mii))) 
                               for mii, gii, win_range in zip(self.M[sl], self.g[sl], self.wins[sl])]

        self.maxLg_dec = max(len(gdii) for gdii in self.gd)
        if self.mode == "matrix_pow2":
            self.maxLg_dec = self.maxLg_enc

        self.loopparams_dec = [(wr1, wr2, len(gdii))
                               for gdii, win_range in zip(self.gd[sl], self.wins[sl])
                               for wr1, wr2 in [(win_range[:(len(gdii)) // 2], win_range[-((len(gdii) + 1) // 2):])]]

        c = self.nsgtf(x)
        return c

    def nsgtf(self, f):
        ft = jnp.fft.fft(f)
        Ls = f.shape[-1]
        assert self.nn == Ls

        sl = slice(1, len(self.g) // 2) if self.mode in ["matrix", "oct", "matrix_pow2"] else slice(0, len(self.g) // 2 + 1)

        if self.mode in ["matrix", "matrix_pow2", "matrix_complete"]:
            self.giis, self.idx_enc = get_ragged_giis_jax(self.g[sl], self.wins[sl], self.M[sl], self.mode, dtype=self.dtype)

            ft = jnp.fft.fft(f)[..., :self.Ls // 2 + 1] 
            t = ft.unsqueeze(-2) * self.giis 
            
            if self.mode == "matrix_complete":
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
                a = jnp.take_along_axis(t, self.idx_enc[0].unsqueeze(0).unsqueeze(0), axis=-1)
                return jnp.fft.ifft(a)

        elif self.mode == "oct":
            self.giis, self.idx_enc = get_ragged_giis_jax(self.g[sl], self.wins[sl], self.M[sl], self.mode, dtype=self.dtype)
            ft = jnp.fft.fft(f)[..., :self.Ls // 2 + 1]
            t = ft.unsqueeze(-2) * self.giis

            ret = []
            for i in range(len(self.idx_enc)): 
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

            for i in range(len(self.idx_enc) - 2):
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

            init_carry = (0, jnp.zeros(*f.shape[:2], 1, self.M[sl][0], dtype=ft.dtype), -1, [])
            _, _, _, bucketed_tensors = jax.lax.scan(critical_loop_body, init_carry, self.loopparams_enc)

            ret = [jnp.fft.ifft(bucketed_tensor) for bucketed_tensor in bucketed_tensors]
            return ret

        elif self.mode == "matrix_slow":
            c = jnp.zeros(*f.shape[:2], len(self.loopparams_enc), self.maxLg_enc, dtype=jnp.fft.fft(f).dtype)

            def matrix_slow_loop_body(carry, params):
                j, c = carry
                mii, win_range, Lg, col = params

                t = ft[:, :, win_range] * jnp.fft.fftshift(self.giis[j, :Lg])
                sl1 = slice(None, (Lg + 1) // 2)
                sl2 = slice(-(Lg // 2), None)

                c = index_update(c, jax.ops.index[:, :, j, sl1], t[:, :, Lg // 2:])
                c = index_update(c, jax.ops.index[:, :, j, sl2], t[:, :, :Lg // 2])

                return (j + 1, c), None

            init_carry = (0, c)
            _, c = jax.lax.scan(matrix_slow_loop_body, init_carry, self.loopparams_enc)

            return jnp.fft.ifft(c)

        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def nsigtf(self, cseq):
        if self.mode not in ["matrix", "matrix_slow", "matrix_complete", "matrix_pow2"]:
            assert isinstance(cseq, list)
            nfreqs = sum(cseq_tsor.shape[2] for cseq_tsor in cseq)
            cseq_shape = (*cseq[0].shape[:2], nfreqs)
            cseq_dtype = cseq[0].dtype
            fc = jnp.concatenate([jnp.fft.fft(cseq_tsor) for cseq_tsor in cseq], axis=2)
        else:
            assert isinstance(cseq, jnp.ndarray)
            cseq_shape = cseq.shape[:3]
            cseq_dtype = cseq.dtype
            fc = jnp.fft.fft(cseq)

        fbins = cseq_shape[2]

        if self.mode == "matrix_slow":
            fr = jnp.zeros(*cseq_shape[:2], self.nn, dtype=cseq_dtype)
            temp0 = jnp.empty(*cseq_shape[:2], self.maxLg_dec, dtype=cseq_dtype)

            def matrix_slow_loop_body(carry, i):
                fr, temp0 = carry
                wr1, wr2, Lg = self.loopparams_dec[i]

                t = fc[:, :, i]
                r = (Lg + 1) // 2
                l = (Lg // 2)

                temp0 = index_update(temp0, jax.ops.index[:, :, :r], t[:, :, :r])
                temp0 = index_update(temp0, jax.ops.index[:, :, Lg - l:Lg], t[:, :, self.maxLg_dec - l:self.maxLg_dec])

                temp0 = temp0 * self.gdiis[i, :Lg] * self.maxLg_dec

                fr = index_add(fr, jax.ops.index[:, :, wr1], temp0[:, :, Lg - l:Lg])
                fr = index_add(fr, jax.ops.index[:, :, wr2], temp0[:, :, :r])
                return (fr, temp0), None

            init_carry = (fr, temp0)
            (fr, _), _ = jax.lax.scan(matrix_slow_loop_body, init_carry, jnp.arange(fbins))

            ftr = fr[:, :, :self.nn // 2 + 1]
            sig = jnp.fft.irfft(ftr, n=self.nn)
            sig = sig[:, :, :self.Ls]
            return sig

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

        elif self.mode in ["oct", "oct_complete"]:
            sl = slice(1, len(self.g) // 2) if self.mode == "oct" else slice(0, len(self.g) // 2 + 1)

            if self.mode == "oct":
                self.gdiis, self.idx_dec = get_ragged_gdiis_oct_jax(self.gd[sl], self.M[sl], self.wins[sl], self.mode, dtype=self.dtype)
            else: # "oct_complete"
                self.gdiis, self.idx_dec = get_ragged_gdiis_oct_jax(self.gd[sl], self.M[sl], self.wins[sl], self.mode, dtype=self.dtype)

            fr = jnp.zeros(*cseq_shape[:2], self.nn // 2 + 1, dtype=cseq_dtype)

            def oct_loop_body(carry, i):
                fr, fc = carry
                gdii_j = self.gdiis[i]
                idx_dec_j = self.idx_dec[i]

                Lg_outer = fc.shape[-1]
                nb_fbins = fc.shape[2]
                temp0 = jnp.zeros(*cseq_shape[:2], nb_fbins, Lg_outer, dtype=cseq_dtype)
                temp0 = fc * gdii_j.unsqueeze(0).unsqueeze(0)
                fr += jnp.sum(jnp.take_along_axis(temp0, idx_dec_j.unsqueeze(0).unsqueeze(0), axis=3), axis=2)
                return (fr, fc), None

            init_carry = (fr, cseq) 
            (fr, _), _ = jax.lax.scan(oct_loop_body, init_carry, jnp.arange(len(self.gdiis)))

            ftr = fr
            sig = jnp.fft.irfft(ftr, n=self.nn)
            sig = sig[:, :, :self.Ls]
            return sig

        elif self.mode == "critical":
            self.gdiis = get_ragged_gdiis_critical_jax(self.gd[sl], self.M[sl], dtype=self.dtype)

            fr = jnp.zeros(*cseq_shape[:2], self.nn, dtype=cseq_dtype)
            fbin_ptr = 0

            def critical_loop_body(carry, gdii_j):
                fr, fbin_ptr, fc = carry
                Lg_outer = gdii_j.shape[-1]
                nb_fbins = fc.shape[2]
                temp0 = jnp.zeros(*cseq_shape[:2], nb_fbins, Lg_outer, dtype=cseq_dtype)

                def inner_loop_body(carry, i):
                    fr, temp0 = carry
                    wr1, wr2, Lg = self.loopparams_dec[fbin_ptr + i]
                    r = (Lg + 1) // 2
                    l = (Lg // 2)
                    fr = index_add(fr, jax.ops.index[:, :, wr1], temp0[:, :, i, Lg_outer - l:Lg_outer])
                    fr = index_add(fr, jax.ops.index[:, :, wr2], temp0[:, :, i, :r])
                    return (fr, temp0), None

                temp0 = fc * gdii_j.unsqueeze(0).unsqueeze(0)
                init_carry = (fr, temp0)
                (fr, _), _ = jax.lax.scan(inner_loop_body, init_carry, jnp.arange(nb_fbins))

                fbin_ptr += nb_fbins
                return (fr, fbin_ptr, fc), None

            init_carry = (fr, fbin_ptr, cseq)
            (fr, _, _), _ = jax.lax.scan(critical_loop_body, init_carry, self.gdiis)

            ftr = fr[:, :, :self.nn // 2 + 1]
            sig = jnp.fft.irfft(ftr, n=self.nn)
            sig = sig[:, :, :self.Ls]
            return sig

        else:
            raise ValueError(f"Invalid mode: {self.mode}")
            import jax.numpy as jnp
from jax.ops import index_update

def get_ragged_giis_jax(g, wins, ms, mode, dtype=jnp.float32):
    c = jnp.zeros((len(g), len(g[0]) // 2 + 1), dtype=dtype)
    ix = []

    if mode == "oct":
        for i in range(len(ms) // len(g[0])):
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

        def scan_body(carry, i):
            j, k, c, ix = carry
            gii, win_range, m = g[i], wins[i], ms[i]

            if i > 0:
                if ms[i] != ms[i - 1] or ((mode == "oct_complete" or mode == "matrix_complete") and (j == 0 or i == len(g) - 1)):
                    j += 1
                    k = 0

            gii = jnp.fft.fftshift(gii).unsqueeze(0)
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
            return (j, k, c, ix), None

        init_carry = (0, 0, c, ix)
        (_, _, c, ix), _ = jax.lax.scan(scan_body, init_carry, jnp.arange(len(g)))

    return jnp.conj(c), ix

import jax.numpy as jnp
from jax.ops import index_update

def get_ragged_gdiis_jax(gd, wins, mode, ms=None, dtype=jnp.float32):
    ragged_gdiis = []
    maxLg_dec = max(len(gdii) for gdii in gd)
    ix = jnp.full((len(gd), len(gd[0]) // 2 + 1), maxLg_dec // 2, dtype=int)

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

import jax.numpy as jnp

def get_ragged_gdiis_oct_jax(gd, ms, wins, mode, dtype=jnp.float32):
    seq_gdiis = []
    ragged_gdiis = []
    mprev = -1
    ix = []

    if mode == "oct_complete":
        ix.append(jnp.full((1, len(gd[0]) // 2 + 1), ms[0] // 2, dtype=int))

    size_per_oct = [next_power_of_2(jnp.max(ms[i * len(gd[0]):(i + 1) * len(gd[0])])) for i in range(len(ms) // len(gd[0]))]
    ix.extend([jnp.full((len(gd[0]), len(gd[0]) // 2 + 1), size // 2, dtype=int) for size in size_per_oct])

    if mode == "oct_complete":
        ix.append(jnp.full((1, len(gd[0]) // 2 + 1), ms[-1] // 2, dtype=int))

    j = 0
    k = 0
    for i, (g, m, win_range) in enumerate(zip(gd, ms, wins)):
        if i > 0 and (m != mprev or (mode == "oct_complete" and i == len(gd) - 1)):
            gdii = jnp.conj(jnp.concatenate(ragged_gdiis, axis=0)).astype(dtype)
            seq_gdiis.append(gdii)
            ragged_gdiis = []
            j += 1
            k = 0

        Lg = len(g)
        gl = g[:(Lg + 1) // 2]
        gr = g[(Lg + 1) // 2:]
        zeros = jnp.zeros(m - Lg, dtype=g.dtype)
        paddedg = jnp.concatenate((gl, zeros, gr), axis=0).unsqueeze(0) * m
        ragged_gdiis.append(paddedg)
        mprev = m

        wr1 = win_range[:(Lg) // 2]
        wr2 = win_range[-((Lg + 1) // 2):]

        if mode == "oct_complete" and i == 0:
            ix[0] = index_update(ix[0], jax.ops.index[k, wr2], jnp.arange(len(wr2)))
        elif mode == "oct_complete" and i == len(gd) - 1:
            ix[-1] = index_update(ix[-1], jax.ops.index[k, wr1], m - (Lg // 2) + jnp.arange(len(wr1)))
        else:
            ix[j] = index_update(ix[j], jax.ops.index[k, wr1], m - (Lg // 2) + jnp.arange(len(wr1)))
            ix[j] = index_update(ix[j], jax.ops.index[k, wr2], jnp.arange(len(wr2)))

        k += 1

    gdii = jnp.conj(jnp.concatenate(ragged_gdiis, axis=0)).astype(dtype)
    seq_gdiis.append(gdii)
    return seq_gdiis, ix

import jax.numpy as jnp

def get_ragged_gdiis_critical_jax(gd, ms, dtype=jnp.float32):
    seq_gdiis = []
    ragged_gdiis = []
    mprev = -1
    for i, (g, m) in enumerate(zip(gd, ms)):
        if i > 0 and m != mprev:
            gdii = jnp.conj(jnp.concatenate(ragged_gdiis, axis=0)).astype(dtype)
            seq_gdiis.append(gdii)
            ragged_gdiis = []

        Lg = len(g)
        gl = g[:(Lg + 1) // 2]
        gr = g[(Lg + 1) // 2:]
        zeros = jnp.zeros(m - Lg, dtype=g.dtype)
        paddedg = jnp.concatenate((gl, zeros, gr), axis=0).unsqueeze(0) * m
        ragged_gdiis.append(paddedg)
        mprev = m

    gdii = jnp.conj(jnp.concatenate(ragged_gdiis, axis=0)).astype(dtype)
    seq_gdiis.append(gdii)
    return seq_gdiis
    import jax.numpy as jnp
from jax.ops import index_update

def nsgfwin_jax(f, q, sr, Ls, min_win=4, Qvar=1, dowarn=True, dtype=jnp.float64, window="hann"):
    nf = sr / 2.0

    lim = jnp.argmax(f > 0).astype(int)
    f = f[lim:] if lim != 0 else f
    q = q[lim:] if lim != 0 else q

    lim = jnp.argmax(f >= nf).astype(int)
    f = f[:lim] if lim != 0 else f
    q = q[:lim] if lim != 0 else q

    assert len(f) == len(q)
    assert jnp.all((f[1:] - f[:-1]) > 0)
    assert jnp.all(q > 0)

    qneeded = f * (Ls / (8.0 * sr))
    # if jnp.any(q >= qneeded) and dowarn:
    #     warn("Q-factor too high for frequencies %s"%",".join("%.2f"%fi for fi in f[q >= qneeded]))

    fbas = f
    lbas = len(fbas)

    frqs = jnp.concatenate(((0.0,), fbas, (nf,)))
    fbas = jnp.concatenate((frqs, sr - frqs[-2:0:-1]))
    fbas *= float(Ls) / sr

    M = jnp.zeros(fbas.shape, dtype=int)
    M = index_update(M, 0, jnp.round(2 * fbas[1])) 
    M = index_update(M, 1, jnp.round(fbas[1] / q[0]))

    for k in range(2, lbas + 1):
        M = index_update(M, k, jnp.round(fbas[k + 1] - fbas[k - 1]))

    M = index_update(M, lbas + 1, jnp.round(fbas[lbas] / q[lbas - 1]))
    M = index_update(M, slice(lbas + 2, None), M[lbas:0:-1]) 
    M = jnp.clip(M, min_win, None)

    if window == "hann":
        g = jnp.array([hannwin_jax(m, dtype=dtype) for m in M])
    elif window == "blackharr":
        g = jnp.array([blackharr_jax(m, dtype=dtype) for m in M])
    elif window[0] == "kaiser":
        _, beta = window
        g = jnp.array([kaiserwin_jax(m, beta, dtype=dtype) for m in M])

    fbas = index_update(fbas, lbas, (fbas[lbas - 1] + fbas[lbas + 1]) / 2)
    fbas = index_update(fbas, lbas + 2, Ls - fbas[lbas])
    rfbas = jnp.round(fbas).astype(int)

    return g, rfbas, M

import jax.numpy as jnp
from jax.ops import index_add

def nsdual_jax(g, wins, nn, M=None, dtype=jnp.float32):
    x = jnp.zeros((nn,), dtype=dtype)
    
    for gi, mii, sl in zip(g, M, wins):
        xa = jnp.square(jnp.fft.fftshift(gi)) * mii
        x = index_add(x, sl, xa)

    gd = jnp.array([gi / jnp.fft.ifftshift(x[wi]) for gi, wi in zip(g, wins)])
    return gd


import jax.numpy as jnp

def hannwin_jax(l, dtype=jnp.float32):
    r = jnp.arange(l, dtype=dtype)
    r *= jnp.pi * 2.0 / l
    r = jnp.cos(r) + 1.0
    r *= 0.5
    return r

def blackharr_jax(n, l=None, mod=True, dtype=jnp.float32):
    if l is None:
        l = n
    nn = (n // 2) * 2
    k = jnp.arange(n)
    bh = (0.35872 - 0.48832 * jnp.cos(k * (2 * jnp.pi / nn)) +
          0.14128 * jnp.cos(k * (4 * jnp.pi / nn)) -
          0.01168 * jnp.cos(k * (6 * jnp.pi / nn)))
    bh = jnp.where(mod, bh, 
                  0.35875 - 0.48829 * jnp.cos(k * (2 * jnp.pi / nn)) +
                  0.14128 * jnp.cos(k * (4 * jnp.pi / nn)) -
                  0.01168 * jnp.cos(k * (6 * jnp.pi / nn)))
    bh = jnp.concatenate((bh, jnp.zeros(l - n, dtype=dtype)))
    bh = jnp.concatenate((bh[-n // 2:], bh[:-n // 2]))
    return bh

def kaiserwin_jax(l, beta, dtype=jnp.float32):
    r = jnp.arange(l, dtype=dtype)
    r *= jnp.pi * 2.0 / l
    r = jnp.cos(r) + 1.0
    r *= 0.5
    r = jnp.sqrt(r)
    r = jnp.i0(beta * jnp.sqrt(1.0 - r**2)) / (2.0 * jnp.i0(beta))
    r = jnp.roll(r, l // 2)
    return r

def calcwinrange_jax(g, rfbas, Ls):
    shift = jnp.concatenate(((jnp.mod(-rfbas[-1], Ls),), rfbas[1:] - rfbas[:-1]))
    timepos = jnp.cumsum(shift)
    nn = timepos[-1]
    timepos -= shift[0]

    wins = []
    for gii, tpii in zip(g, timepos):
        Lg = len(gii)
        win_range = jnp.arange(-(Lg // 2) + tpii, Lg - (Lg // 2) + tpii, dtype=int) % nn
        wins.append(win_range)

    return wins, nn

import jax.numpy as jnp

class Scale:
    dbnd = 1.e-8

    def __init__(self, bnds):
        self.bnds = bnds

    def __len__(self):
        return self.bnds

    def Q(self, bnd=None):
        if bnd is None:
            bnd = jnp.arange(self.bnds)
        return self.F(bnd) * self.dbnd / (self.F(bnd + self.dbnd) - self.F(bnd - self.dbnd))

    def __call__(self):
        f = jnp.array([self.F(b) for b in range(self.bnds)], dtype=float)
        q = jnp.array([self.Q(b) for b in range(self.bnds)], dtype=float)
        return f, q

    def suggested_sllen_trlen(self, sr):
        f, q = self()
        Ls = int(jnp.ceil(jnp.max((q * 8.0 * sr) / f)))
        Ls = Ls + (-Ls % 4)
        sllen = Ls
        trlen = sllen // 4
        trlen = trlen + (-trlen % 2)
        return sllen, trlen


class LogScale(Scale):
    def __init__(self, fmin, fmax, bnds, beyond=0):
        super().__init__(bnds + beyond * 2)
        lfmin = jnp.log2(fmin)
        lfmax = jnp.log2(fmax)
        odiv = (lfmax - lfmin) / (bnds - 1)
        lfmin_ = lfmin - odiv * beyond
        lfmax_ = lfmax + odiv * beyond
        self.fmin = 2**lfmin_
        self.fmax = 2**lfmax_
        self.pow2n = 2**odiv
        self.q = jnp.sqrt(self.pow2n) / (self.pow2n - 1.0) / 2.0 

    def F(self, bnd=None):
        return self.fmin * self.pow2n**(bnd if bnd is not None else jnp.arange(self.bnds))


class FlexLogOctScale(Scale):
    def __init__(self, fs, numocts, binsoct, flex_q, beyond=0):
        fmax = fs / 2
        fmin = fmax / (2**numocts)

        self.bnds = 0
        for i in range(numocts):
            self.bnds += binsoct[i]

        lfmin = jnp.log2(fmin)
        lfmax = jnp.log2(fmax)

        odiv = (lfmax - lfmin) / (self.bnds - 1)
        lfmin_ = lfmin - odiv * beyond
        lfmax_ = lfmax + odiv * beyond
        self.fmin = 2**lfmin_
        self.fmax = 2**lfmax_
        self.pow2n = 2**odiv
        self.q = jnp.sqrt(self.pow2n) / (self.pow2n - 1.0) / 2.0

        self.bnds = self.bnds 

    def F(self, bnd=None):
        return self.fmin * self.pow2n**(bnd if bnd is not None else jnp.arange(self.bnds))

    def Q(self, bnd=None):
        return self.q
