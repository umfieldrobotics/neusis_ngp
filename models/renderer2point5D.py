"""
Adapt to 2.5D heightmap representation; The color network is still in 3D
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
import sys, os
import pickle 
import matplotlib.pyplot as plt
import time 

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val

    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]

    return vertices, triangles
def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
class NeuSRenderer:
    def __init__(self,
                 sdf_network,
                 deviation_network,
                 color_network,
                 base_exp_dir,
                 expID,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.base_exp_dir = base_exp_dir
        self.expID = expID
    
    
    def render_altimeter(self, pts, sdf_network, compute_gradient=False):
        if compute_gradient:
            sdf_nn_output, gradients = sdf_network.forward_with_nablas(pts)
        else:
            sdf_nn_output = sdf_network(pts)
            gradients = None
        z = sdf_nn_output[:, :1]
        return {
            "z":z,# this is f(x,y)
            "gradients":gradients
        }

    def render_core_sonar(self,
                        dirs,
                        pts,
                        dists,
                        sdf_network,
                        deviation_network,
                        color_network,
                        # n_pixels,
                        # arc_n_samples,
                        # ray_n_samples,
                        cos_anneal_ratio=0.0):

        # pts_mid = pts + dirs * dists.view(-1,1)/2 #(-1,3)
        pts_mid = (pts + dirs * dists.view(-1,1)/2).contiguous() #(-1,3), for tcnn

        sdf_nn_output, gradients = sdf_network.forward_with_nablas(pts_mid[:,:2])

        sdf =  pts_mid[:,2:3]-sdf_nn_output[:, :1]

        feature_vector = sdf_nn_output[:, 1:]

        gradient_error = (torch.linalg.norm(gradients, ord=2,
                                            dim=-1) - 0.0) ** 2

        gradients = torch.concat((-gradients, torch.ones_like(gradients[:,:1])),dim=1) 
        # gradients = F.normalize(gradients,dim=1)





        sampled_color = color_network(pts_mid, gradients, dirs, feature_vector).reshape(self.n_selected_px, self.arc_n_samples, self.ray_n_samples)


        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s.expand(self.n_selected_px*self.arc_n_samples*self.ray_n_samples, 1)
        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        activation  = nn.Softplus(beta=100)


        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        # iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                    #  F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
        iter_cos = -(activation(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                    activation(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points

        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5


        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(self.n_selected_px, self.arc_n_samples, self.ray_n_samples).clip(0.0, 1.0)

        cumuProdAllPointsOnEachRay = torch.cat([torch.ones([self.n_selected_px, self.arc_n_samples, 1]), 1. - alpha + 1e-7], -1)
    
        cumuProdAllPointsOnEachRay = torch.cumprod(cumuProdAllPointsOnEachRay, -1)

        TransmittancePointsOnArc = cumuProdAllPointsOnEachRay[:, :, self.ray_n_samples-2]
        
        alphaPointsOnArc = alpha[:, :, self.ray_n_samples-1]

        weights = alphaPointsOnArc * TransmittancePointsOnArc 

        intensityPointsOnArc = sampled_color[:, :, self.ray_n_samples-1]

        summedIntensities = (intensityPointsOnArc*weights).sum(dim=1) 

        # Eikonal loss
        # gradients = gradients.reshape(n_pixels, arc_n_samples, ray_n_samples, 3)

        # gradient_error = (torch.linalg.norm(gradients, ord=2,
                                            # dim=-1) - 1.0) ** 2

        variation_error = torch.linalg.norm(alpha, ord=1, dim=-1).sum()

        return {
            'color': summedIntensities,
            'intensityPointsOnArc': intensityPointsOnArc,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients,
            's_val': 1.0 / inv_s,
            'weights': weights,
            'cdf': c.reshape(self.n_selected_px, self.arc_n_samples, self.ray_n_samples),
            'gradient_error': gradient_error,
            'variation_error': variation_error
        }

    def up_sample(self, r, theta, phi_vals, sdf, n_importance, inv_s, last=False):
        """
        Up sampling give a fixed inv_s
        """
        # print("inv_s in up_sample ", inv_s)
        # dist = torch.diff(r, dim=1)# n_selected_px, arc_n_samples-1

        prev_phi_vals, next_phi_vals = phi_vals[:, :-1], phi_vals[:, 1:]
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_phi_vals - prev_phi_vals + 1e-5)

        prev_cos_val = torch.cat([torch.zeros([self.n_selected_px, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) 
        dist = (next_phi_vals - prev_phi_vals + 1e-5).abs() * r.reshape(-1,1)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([self.n_selected_px, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        new_phi_vals = sample_pdf(phi_vals, weights, n_importance, det=True).detach()
        
        arc_n_samples = phi_vals.shape[1]
        assert n_importance == new_phi_vals.shape[1]

        phi_vals = torch.cat([phi_vals, new_phi_vals], dim=-1)
        phi_vals, index = torch.sort(phi_vals, dim=-1)
        _, pts, _ = self.get_coords_on_arc(r, theta, new_phi_vals,arc_n_samples=n_importance)

        if not last:
            pts = pts.reshape(-1, 3)
            new_sdf = pts[:,2:3] - self.sdf_network.sdf(pts[:,:2])
            
            new_sdf = new_sdf.reshape(self.n_selected_px, n_importance)
            

            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(self.n_selected_px,)[:, None].expand(self.n_selected_px, arc_n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(self.n_selected_px, arc_n_samples + n_importance)

        return phi_vals, sdf

    # def render_sonar(self, rays_d, pts, dists, n_pixels,
                    #  arc_n_samples, ray_n_samples, cos_anneal_ratio=0.0):
    def render_sonar(self, H, W, phi_min, phi_max, r_min, r_max, c2w, n_selected_px, arc_n_samples, ray_n_samples, 
            hfov, px, r_increments, randomize_points, device, cube_center, cos_anneal_ratio=0.0):
        r, theta, phi = self.get_arcs(H, W, phi_min, phi_max, r_min, r_max, c2w, n_selected_px, arc_n_samples, ray_n_samples, 
            hfov, px, r_increments, randomize_points, device, cube_center)
        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                dirs, pts_r_rand, dists = self.get_coords_on_arc(r, theta, phi,arc_n_samples=self.arc_n_samples)

                


                sdf = (pts_r_rand[:,2:3]- self.sdf_network.sdf(pts_r_rand[:,:2])).reshape(n_selected_px, self.arc_n_samples)

                pts_r_rand = pts_r_rand.reshape(n_selected_px, self.arc_n_samples, 3)# (n_selected_px, arc_n_samples, 3)

                
                for i in range(self.up_sample_steps):
                    # phi, sdf = self.up_sample(r, theta, phi, sdf, self.n_importance // self.up_sample_steps, self.deviation_network(torch.zeros([1, 3])).item()  * 2**i, last=(i + 1 == self.up_sample_steps))
                    phi, sdf = self.up_sample(r, theta, phi, sdf, self.n_importance // self.up_sample_steps, 64 * 2**i, last=(i + 1 == self.up_sample_steps))
                    # print("here ", i)

            self.arc_n_samples = self.arc_n_samples + self.n_importance
        dirs, pts_r_rand, dists = self.get_coords(r, theta, phi)

        
        ret_fine = self.render_core_sonar(dirs,
                                        pts_r_rand,
                                        dists,
                                        self.sdf_network,
                                        self.deviation_network,
                                        self.color_network,
                                        # n_pixels,
                                        # arc_n_samples,
                                        # ray_n_samples,
                                        cos_anneal_ratio=cos_anneal_ratio)
        
        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        #s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'color_fine': color_fine,
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'intensityPointsOnArc': ret_fine["intensityPointsOnArc"],
            'gradient_error': ret_fine['gradient_error'],
            'variation_error': ret_fine['variation_error']
        }

    def get_arcs(self, H, W, phi_min, phi_max, r_min, r_max, c2w, n_selected_px, arc_n_samples, ray_n_samples, 
            hfov, px, r_increments, randomize_points, device, cube_center):

        self.n_selected_px = n_selected_px
        self.arc_n_samples = arc_n_samples
        self.ray_n_samples = ray_n_samples
        self.device = device
        self.c2w = c2w 
        self.cube_center = cube_center
        self.r_increments = r_increments
        self.randomize_points = randomize_points


        i = px[:, 0]
        j = px[:, 1]

        self.i = i

        # sample angle phi
        phi = torch.linspace(phi_min, phi_max, arc_n_samples).float().repeat(n_selected_px).reshape(n_selected_px, -1)

        dphi = (phi_max - phi_min) / arc_n_samples
        rnd = -dphi + torch.rand(n_selected_px, arc_n_samples)*2*dphi

        sonar_resolution = (r_max-r_min)/H
        self.sonar_resolution = sonar_resolution
        if randomize_points:
            phi =  torch.clip(phi + rnd, min=phi_min, max=phi_max)

        # compute radius at each pixel
        r = i*sonar_resolution + r_min
        # compute bearing angle at each pixel
        theta = -hfov/2 + j*hfov/W

        return r, theta, phi    

    def cal_pts(self, r_samples, theta_samples, phi_samples, coords, ray_n_samples):
        pts = torch.stack((r_samples, theta_samples, phi_samples), dim=-1).reshape(-1, 3)

        dists = torch.diff(r_samples, dim=1)
        dists = torch.cat([dists, torch.Tensor([self.sonar_resolution]).expand(dists[..., :1].shape)], -1)

        #r_samples_mid = r_samples + dists/2

        X_r_rand = pts[:,0]*torch.cos(pts[:,1])*torch.cos(pts[:,2])
        Y_r_rand = pts[:,0]*torch.sin(pts[:,1])*torch.cos(pts[:,2])
        Z_r_rand = pts[:,0]*torch.sin(pts[:,2])
        pts_r_rand = torch.stack((X_r_rand, Y_r_rand, Z_r_rand, torch.ones_like(X_r_rand)))


        pts_r_rand = torch.matmul(self.c2w, pts_r_rand)

        pts_r_rand = torch.stack((pts_r_rand[0,:], pts_r_rand[1,:], pts_r_rand[2,:]))

        # Centering step 
        pts_r_rand = pts_r_rand.T - self.cube_center

        # Transform to cartesian to apply pose transformation and get the direction
        # transformation as described in https://www.ri.cmu.edu/pub_files/2016/5/thuang_mastersthesis.pdf
        X = coords[:,0]*torch.cos(coords[:,1])*torch.cos(coords[:,2])
        Y = coords[:,0]*torch.sin(coords[:,1])*torch.cos(coords[:,2])
        Z = coords[:,0]*torch.sin(coords[:,2])

        dirs = torch.stack((X,Y,Z, torch.ones_like(X))).T
        if ray_n_samples>1:
            dirs = dirs.repeat_interleave(ray_n_samples, 0)
        dirs = torch.matmul(self.c2w, dirs.T).T
        origin = torch.matmul(self.c2w, torch.tensor([0., 0., 0., 1.])).unsqueeze(dim=0)
        dirs = dirs - origin
        dirs = dirs[:, 0:3]
        dirs = torch.nn.functional.normalize(dirs, dim=1)

        return dirs, pts_r_rand, dists

    def get_coords_on_arc(self, r, theta, phi, arc_n_samples):
        coords = torch.stack((r.repeat_interleave(arc_n_samples).reshape(self.n_selected_px, -1), 
                            theta.repeat_interleave(arc_n_samples).reshape(self.n_selected_px, -1), 
                            phi), dim = -1)
        coords = coords.reshape(-1, 3)
        r_samples = torch.index_select(self.r_increments, 0, self.i).repeat(arc_n_samples)

        rnd = torch.rand((self.n_selected_px*arc_n_samples))*self.sonar_resolution
        
        if self.randomize_points:
            r_samples = r_samples + rnd
        
        

        theta_samples = coords[:, 1]
        phi_samples = coords[:, 2]
        dirs, pts_r_rand, dists = self.cal_pts(r_samples[...,None], theta_samples[...,None], phi_samples[...,None], coords,ray_n_samples=1)
        return dirs, pts_r_rand, dists

    def get_coords(self, r, theta, phi):
        

        # Need to calculate coords to figure out the ray direction 
        # the following operations mimick the cartesian product between the two lists [r, theta] and phi
        # coords is of size: n_selected_px x n_arc_n_samples x 3
        coords = torch.stack((r.repeat_interleave(self.arc_n_samples).reshape(self.n_selected_px, -1), 
                            theta.repeat_interleave(self.arc_n_samples).reshape(self.n_selected_px, -1), 
                            phi), dim = -1)
        coords = coords.reshape(-1, 3)

        holder = torch.empty(self.n_selected_px, self.arc_n_samples*self.ray_n_samples, dtype=torch.long).to(self.device)
        bitmask = torch.zeros(self.ray_n_samples, dtype=torch.bool)
        bitmask[self.ray_n_samples - 1] = True
        bitmask = bitmask.repeat(self.arc_n_samples)


        for n_px in range(self.n_selected_px):
            holder[n_px, :] = torch.randint(0, self.i[n_px]-1, (self.arc_n_samples*self.ray_n_samples,))
            holder[n_px, bitmask] = self.i[n_px] 
        
        holder = holder.reshape(self.n_selected_px, self.arc_n_samples, self.ray_n_samples)
        
        holder, _ = torch.sort(holder, dim=-1)

        holder = holder.reshape(-1)
            

        r_samples = torch.index_select(self.r_increments, 0, holder).reshape(self.n_selected_px, 
                                                                        self.arc_n_samples, 
                                                                        self.ray_n_samples)
        
        rnd = torch.rand((self.n_selected_px, self.arc_n_samples, self.ray_n_samples))*self.sonar_resolution
        
        if self.randomize_points:
            r_samples = r_samples + rnd

        rs = r_samples[:, :, -1]
        r_samples = r_samples.reshape(self.n_selected_px*self.arc_n_samples, self.ray_n_samples)

        theta_samples = coords[:, 1].repeat_interleave(self.ray_n_samples).reshape(-1, self.ray_n_samples)
        phi_samples = coords[:, 2].repeat_interleave(self.ray_n_samples).reshape(-1, self.ray_n_samples)

        dirs, pts_r_rand, dists = self.cal_pts(r_samples, theta_samples, phi_samples, coords, ray_n_samples=self.ray_n_samples)

        
        return dirs, pts_r_rand, dists

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
