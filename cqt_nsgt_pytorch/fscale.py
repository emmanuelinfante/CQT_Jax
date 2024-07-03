import jax.numpy as jnp

class Scale:
    dbnd = 1.e-8

    def __init__(self, bnds):
        self.bnds = bnds

    def __len__(self):
        return self.bnds

    def Q(self, bnd=None):
        # Numerical differentiation (if self.Q not defined by sub-class)
        if bnd is None:
            bnd = jnp.arange(self.bnds)  # Use jnp.arange
        return self.F(bnd) * self.dbnd / (self.F(bnd + self.dbnd) - self.F(bnd - self.dbnd))

    def __call__(self):
        f = jnp.array([self.F(b) for b in range(self.bnds)], dtype=float) # Use jnp.array
        q = jnp.array([self.Q(b) for b in range(self.bnds)], dtype=float) # Use jnp.array
        return f, q

    def suggested_sllen_trlen(self, sr):
        f, q = self()
        Ls = int(jnp.ceil(jnp.max((q * 8.0 * sr) / f))) # Use jnp.ceil, jnp.max
        Ls = Ls + (-Ls % 4)
        sllen = Ls
        trlen = sllen // 4
        trlen = trlen + (-trlen % 2)
        return sllen, trlen


class LogScale(Scale):
    def __init__(self, fmin, fmax, bnds, beyond=0):
        super().__init__(bnds + beyond * 2)
        lfmin = jnp.log2(fmin)  # Use jnp.log2
        lfmax = jnp.log2(fmax)  # Use jnp.log2
        odiv = (lfmax - lfmin) / (bnds - 1)
        lfmin_ = lfmin - odiv * beyond
        lfmax_ = lfmax + odiv * beyond
        self.fmin = 2**lfmin_
        self.fmax = 2**lfmax_
        self.pow2n = 2**odiv
        self.q = jnp.sqrt(self.pow2n) / (self.pow2n - 1.0) / 2.0  # Use jnp.sqrt

    def F(self, bnd=None):
        return self.fmin * self.pow2n**(bnd if bnd is not None else jnp.arange(self.bnds))


class FlexLogOctScale(Scale):
    def __init__(self, fs, numocts, binsoct, flex_q, beyond=0):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        fmax = fs / 2
        fmin = fmax / (2**numocts)

        self.bnds = 0
        for i in range(numocts):
            self.bnds += binsoct[i]

        lfmin = jnp.log2(fmin)
        lfmax = jnp.log2(fmax)

        odiv = (lfmax - lfmin) / (self.bnds - 1) #I think this was a typo and it should be self.bnds
        lfmin_ = lfmin - odiv * beyond #This "beyon" is not defined, should it be an argument of the class?
        lfmax_ = lfmax + odiv * beyond #This "beyon" is not defined, should it be an argument of the class?
        self.fmin = 2**lfmin_
        self.fmax = 2**lfmax_
        self.pow2n = 2**odiv
        self.q = jnp.sqrt(self.pow2n) / (self.pow2n - 1.0) / 2.0

        self.bnds = self.bnds  # number of frequency bands

    def F(self, bnd=None):
        return self.fmin * self.pow2n**(bnd if bnd is not None else jnp.arange(self.bnds))

    def Q(self, bnd=None):
        return self.q
