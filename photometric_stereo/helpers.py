import numpy as np

def reconstructSurf(normals, mask):
    """
    Surface reconstruction using the Frankot-Chellappa algorithm
    """

    # Compute surface gradients (p, q)
    eps = 1e-10
    p_img = normals[:, :, 1] / (normals[:, :, 3] + eps)
    q_img = normals[:, :, 2] / (normals[:, :, 3] + eps)

    # Take Fourier Transform of p and q
    fp_img = np.fft.fft2(p_img)
    fq_img = np.fft.fft2(q_img)

    # The domains of u and v are important
    cols, rows = fp_img.shape
    u, v = np.meshgrid(np.arange(0, cols) - np.floor(cols / 2),
                       np.arange(0, rows) - np.floor(rows / 2))

    u = np.fft.ifftshift(u)
    v = np.fft.ifftshift(v)
    fz = (1j * u * fp_img + 1j * v * fq_img) / (u**2 + v**2 + eps)

    # Take inverse Fourier Transform back to the spatial domain
    ifz = np.fft.ifft2(fz)
    ifz[~mask] = 0

    z = np.real(ifz)
    surf_img = (z - np.min(z)) / (np.max(z) - np.min(z))
    surf_img[~mask] = 0

    return surf_img