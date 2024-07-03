import jax.numpy as jnp
from jax.ops import index_update
from .util import hannwin_jax, blackharr_jax, kaiserwin_jax  # We'll define these later

def nsgfwin_jax(f, q, sr, Ls, min_win=4, Qvar=1, dowarn=True, dtype=jnp.float64, window="hann"):
    nf = sr / 2.0

    # Ensure f is within the valid range (0, nf)
    lim = jnp.argmax(f > 0).astype(int) # Convert to int for indexing
    f = f[lim:] if lim != 0 else f
    q = q[lim:] if lim != 0 else q

    lim = jnp.argmax(f >= nf).astype(int) # Convert to int for indexing
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

    # Calculate window lengths (M)
    M = jnp.zeros(fbas.shape, dtype=int)
    M = index_update(M, 0, jnp.round(2 * fbas[1])) 
    M = index_update(M, 1, jnp.round(fbas[1] / q[0]))

    # Note: This loop could potentially be optimized with jax.lax.scan
    for k in range(2, lbas + 1):
        M = index_update(M, k, jnp.round(fbas[k + 1] - fbas[k - 1]))

    M = index_update(M, lbas + 1, jnp.round(fbas[lbas] / q[lbas - 1]))
    M = index_update(M, slice(lbas + 2, None), M[lbas:0:-1]) 
    M = jnp.clip(M, min_win, None) # Use None for the upper limit in jnp.clip

    # Generate windows (g)
    if window == "hann":
        g = jnp.array([hannwin_jax(m, dtype=dtype) for m in M])
    elif window == "blackharr":
        g = jnp.array([blackharr_jax(m, dtype=dtype) for m in M])
    elif window[0] == "kaiser":
        _, beta = window
        g = jnp.array([kaiserwin_jax(m, beta, dtype=dtype) for m in M])

    # Adjust fbas and calculate rfbas
    fbas = index_update(fbas, lbas, (fbas[lbas - 1] + fbas[lbas + 1]) / 2)
    fbas = index_update(fbas, lbas + 2, Ls - fbas[lbas])
    rfbas = jnp.round(fbas).astype(int)

    return g, rfbas, M
