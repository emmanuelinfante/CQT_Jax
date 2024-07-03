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
    bh = jnp.where(mod, bh,  # Use jnp.where for conditional assignment
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
