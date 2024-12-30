import torch
import matplotlib
matplotlib.use('Agg')
import numpy as np

torch.autograd.set_detect_anomaly(True)

def get_mgrid(sidelen, dim=2, normalize=False):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        if normalize:
            pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
            pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        if normalize:
            pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
            pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
            pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)
    if normalize:
        pixel_coords -= 0.5
        pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def select_coordinates(coords_all, target, N_rand, select_valid_px):
    if select_valid_px:
        coords = torch.nonzero(target)
    else: 
        select_inds = torch.randperm(coords_all.shape[0])[:N_rand]
        coords = coords_all[select_inds]
    return coords
