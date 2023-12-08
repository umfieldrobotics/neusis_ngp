import os, sys
import numpy as np
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import scipy.io
import matplotlib.pyplot as plt
from helpers import *
from MLP import *
#from PIL import Image
# import cv2 as cv
import time
import random
import string 
from pyhocon import ConfigFactory
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF, SDFNetworkTcnn
from models.renderer2point5D import NeuSRenderer
import trimesh
from itertools import groupby
from operator import itemgetter
from load_data import *
import logging
import argparse 

from math import ceil

from torch.utils.tensorboard import SummaryWriter

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()



class Runner:
    def __init__(self, args, write_config=True):
        conf_path = args.conf
        f = open(conf_path)
        conf_text = f.read()
        self.is_continue = args.is_continue
        self.conf = ConfigFactory.parse_string(conf_text)
        self.write_config = write_config
        self.PC_name = args.PC_name
        self.heightmap_name = args.heightmap_name
        self.slurm_id = args.slurm_id
        self.load_slurm_id=args.load_slurm_id
        self.base_dir = args.base_dir

        print(self.conf)

    def set_params(self):
        self.expID = self.conf.get_string('conf.expID') + "_" + self.slurm_id
        self.load_expID = self.conf.get_string('conf.expID') + "_" + self.load_slurm_id

        dataset = self.conf.get_string('conf.dataset')
        self.image_setkeyname =  self.conf.get_string('conf.image_setkeyname') 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.N_rand = self.conf.get_int('train.num_select_pixels') #H*W 
        self.BN_rand = self.conf.get_int('train.num_select_beams') #H* BN_rand

        self.arc_n_samples = self.conf.get_int('train.arc_n_samples')
        self.arc_n_samples_copy = self.arc_n_samples
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.learning_rate_decay_mode = self.conf.get_string('train.learning_rate_decay_mode')

        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.percent_select_true = self.conf.get_float('train.percent_select_true', default=0.5)
        # self.r_div = self.conf.get_bool('train.r_div')
        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.variation_reg_weight = self.conf.get_float('train.variation_reg_weight')
        self.px_sample_min_weight = self.conf.get_float('train.px_sample_min_weight')
        self.bathy_weight = self.conf.get_float('train.bathy_weight')
        self.intensity_weight = self.conf.get_float('train.intensity_weight')

        self.ray_n_samples = self.conf['model.neus_renderer']['n_samples']
        self.ray_n_samples_copy = self.ray_n_samples
        self.base_exp_dir = self.base_dir + 'experiments/{}'.format(self.expID)
        self.base_exp_dir_init = self.base_dir + 'experiments/{}'.format(self.load_expID)

        self.randomize_points = self.conf.get_float('train.randomize_points')
        self.select_px_method = self.conf.get_string('train.select_px_method')
        self.select_valid_px = self.conf.get_bool('train.select_valid_px')        
        self.x_max = self.conf.get_float('mesh.x_max')
        self.x_min = self.conf.get_float('mesh.x_min')
        self.y_max = self.conf.get_float('mesh.y_max')
        self.y_min = self.conf.get_float('mesh.y_min')
        self.z_max = self.conf.get_float('mesh.z_max')
        self.z_min = self.conf.get_float('mesh.z_min')
        self.level_set = self.conf.get_float('mesh.level_set')
        self.res = self.conf.get_float('mesh.res')

        self.data = load_data(dataset,self.base_dir,self.PC_name,self.heightmap_name)


        self.H, self.W = self.data[self.image_setkeyname][0].shape

        self.r_min = self.data["min_range"]
        self.r_max = self.data["max_range"]
        self.phi_min = self.data["min_phi"]# to deal with when sonar tilts down
        self.phi_max = self.data["max_phi"]
        self.vfov = self.data["vfov"]
        self.hfov = self.data["hfov"]
        print("self.phi_min self.phi_max", self.phi_min*180/math.pi, self.phi_max*180/math.pi)



        self.cube_center = torch.Tensor([(self.x_max + self.x_min)/2, (self.y_max + self.y_min)/2, (self.z_max + self.z_min)/2])

        self.timef = self.conf.get_bool('conf.timef')
        self.end_iter = self.conf.get_int('train.end_iter')
        self.start_iter = self.conf.get_int('train.start_iter')
         
        self.object_bbox_min = self.conf.get_list('mesh.object_bbox_min')
        self.object_bbox_max = self.conf.get_list('mesh.object_bbox_max')

        r_increments = []
        self.sonar_resolution = (self.r_max-self.r_min)/self.H
        for i in range(self.H):
            r_increments.append(i*self.sonar_resolution + self.r_min)

        self.r_increments = torch.FloatTensor(r_increments).to(self.device)

        extrapath = self.base_dir + 'experiments/{}'.format(self.expID)
        if not os.path.exists(extrapath):
            os.makedirs(extrapath)

        extrapath = self.base_dir +'experiments/{}/checkpoints'.format(self.expID)
        if not os.path.exists(extrapath):
            os.makedirs(extrapath)

        extrapath = self.base_dir +'experiments/{}/model'.format(self.expID)
        if not os.path.exists(extrapath):
            os.makedirs(extrapath)

        if self.write_config:
            with open(self.base_dir +'experiments/{}/config.json'.format(self.expID), 'w') as f:
                json.dump(self.conf.__dict__, f, indent = 2)

        # Create all image tensors beforehand to speed up process

        self.i_train = np.arange(len(self.data[self.image_setkeyname]))

        self.coords_all_ls = [(x, y) for x in np.arange(self.H) for y in np.arange(self.W)]
        self.coords_all_set = set(self.coords_all_ls)

        #self.coords_all = torch.from_numpy(np.array(self.coords_all_ls)).to(self.device)

        self.del_coords = []
        for y in np.arange(self.W):
            tmp = [(x, y) for x in np.arange(0, self.ray_n_samples)]
            self.del_coords.extend(tmp)

        self.coords_all = list(self.coords_all_set - set(self.del_coords))
        self.coords_all = torch.LongTensor(self.coords_all).to(self.device)

        # self.criterion = torch.nn.L1Loss(reduction='sum')
        self.criterion = torch.nn.HuberLoss(reduction='sum')
        
        self.model_list = []
        # self.writer = None
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))


        # Networks
        params_to_train = []
        # self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.sdf_network = SDFNetworkTcnn(**self.conf['model.sdf_network']).to(self.device)

        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate, betas=(0.9, 0.99), eps=1e-15)


        self.iter_step = 0
        self.renderer = NeuSRenderer(self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    self.base_exp_dir,
                                    self.expID,
                                    **self.conf['model.neus_renderer'])  

        latest_model_name = None
        if self.is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth': #and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)
    
    def getRandomImgCoordsAllBins(self, target,idx_x_min=10, step=1):
        idx_y = np.arange(self.W)
        np.random.shuffle(idx_y)
        idx_y = torch.Tensor(idx_y[:step].reshape(-1,1)).long().view(-1) # step,1
        idx_x = torch.arange(idx_x_min, self.H, dtype=torch.long).view(-1)# 1,self.H
        coords = torch.cartesian_prod(idx_x,idx_y)# fix bugs

        target = torch.Tensor(target).to(self.device)
        return coords, target

    def getSerialImgCoordsAllBins(self, target,idx_y_start, idx_x_min=10, step=1):
        idx_y = torch.arange(idx_y_start, idx_y_start+step,dtype=torch.long).view(-1)
        idx_x = torch.arange(idx_x_min, self.H, dtype=torch.long).reshape(-1)# 1,self.H
        coords = torch.cartesian_prod(idx_x,idx_y)
        target = torch.Tensor(target).to(self.device)
        return coords, target

    def getRandomImgCoordsAllBeams(self, target, idx_x_start=10, idx_y_min=0, step=1):
        idx_x = np.arange(idx_x_start, self.H)
        np.random.shuffle(idx_x)
        idx_x = torch.Tensor(idx_x[:step].reshape(-1,1)).long().view(-1) # step,
        idx_y = torch.arange(idx_y_min, self.W, dtype=torch.long).view(-1)# self.W,

        coords = torch.cartesian_prod(idx_x,idx_y)
        target = torch.Tensor(target).to(self.device)
        return coords, target  

    def getSerialImgCoordsAllBeams(self, target, idx_x_start=10, idx_y_min=0, step=1):
        idx_x = torch.arange(idx_x_start, idx_x_start+step,dtype=torch.long).view(-1) # range
        idx_y = torch.arange(idx_y_min, self.W, dtype=torch.long).reshape(-1)# 1,self.W
        coords = torch.cartesian_prod(idx_x,idx_y)
        target = torch.Tensor(target).to(self.device)
        return coords, target

    def getRandomImgCoordsByPercentage(self, target):
        true_coords = []
        for y in np.arange(self.W):
            col = target[:, y]
            gt0 = col > 0
            indTrue = np.where(gt0)[0]
            if len(indTrue) > 0:
                true_coords.extend([(x, y) for x in indTrue])

        sampling_perc = int(self.percent_select_true*len(true_coords))
        true_coords = random.sample(true_coords, sampling_perc)
        true_coords = list(set(true_coords) - set(self.del_coords))
        true_coords = torch.LongTensor(true_coords).to(self.device)
        target = torch.Tensor(target).to(self.device)
        if self.iter_step%len(self.data[self.image_setkeyname]) !=0:
            N_rand = 0
        else:
            N_rand = self.N_rand
        N_rand = self.N_rand
        coords = select_coordinates(self.coords_all, target, N_rand, self.select_valid_px)
        
        coords = torch.cat((coords, true_coords), dim=0)
            
        return coords, target
    
    def export_heightmap_mae(self, no_grad=True, mask_out=30, compute_mae=False):
        heightmap_H = int((self.x_max-self.x_min)/self.res)
        coords = get_mgrid(heightmap_H) # cuda
        x = coords[:,0] * self.res + self.x_min
        y = coords[:,1] * self.res + self.y_min
        
        pts=torch.concat((x.view(-1,1),y.view(-1,1)),dim=1)- self.cube_center[:2].view(1,2) # (-1,2)
        
        if no_grad:
            with torch.no_grad():
                render_out = self.renderer.render_altimeter(pts,self.sdf_network)
        else:


            render_out = self.renderer.render_altimeter(pts,self.sdf_network)
        
        z = render_out["z"][:,0] + self.cube_center[2]
        
        heightmap_nn = z.view(heightmap_H,heightmap_H).detach().cpu().numpy()
        if compute_mae:
            heightmap_gt = self.data["heightmap_init"]

            mask = np.ones_like(heightmap_gt,dtype=bool) # invalid mask
            # mask_out = 30 # meter
            mask[:int(mask_out/self.res),:]=0
            mask[heightmap_H-int(mask_out/self.res):,:]=0
            mask[:,:int(mask_out/self.res)]=0
            mask[:,heightmap_H-int(mask_out/self.res):]=0
            heightmap_gt_plot = heightmap_gt.copy()
            heightmap_nn_plot = heightmap_nn.copy()


            heightmap_gt_plot[~mask]=np.nan
            heightmap_nn_plot[~mask]=np.nan
            diff = heightmap_gt_plot - heightmap_nn_plot
            mae = np.abs(diff)[~np.isnan(diff)].mean()


            return heightmap_nn, mae
        else:
            return heightmap_nn, 0
        # return heightmap_nn, mae

    def train(self):
        loss_arr = []
        self.inv_s_up_sample_copy = self.renderer.inv_s_up_sample
        for i in trange(self.start_iter, self.end_iter, len(self.data[self.image_setkeyname])):
            i_train = np.arange(len(self.data[self.image_setkeyname]))
            np.random.shuffle(i_train)
            loss_total = 0
            sum_intensity_loss = 0
            sum_eikonal_loss = 0
            sum_total_variational = 0
            sum_bathymetric_loss = 0

            # corse to fine:
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            # self.ray_n_samples = min(3*self.ray_n_samples_copy, int(6**progress * self.ray_n_samples_copy))
            # self.arc_n_samples = min(4*self.arc_n_samples_copy, int(8**progress * self.arc_n_samples_copy))
            
            self.renderer.inv_s_up_sample = min(4*self.inv_s_up_sample_copy, (8**progress * self.inv_s_up_sample_copy))

            for j in trange(0, len(i_train)):
                img_i = i_train[j]
                target = self.data[self.image_setkeyname][img_i]

                
                pose = self.data["sensor_poses"][img_i]  
                
                if self.select_px_method == "byprob":
                    coords, target = self.getRandomImgCoordsByProbability(target)
                elif self.select_px_method == "allbins":
                    coords, target = self.getRandomImgCoordsAllBins(target,idx_x_min=100,step=self.BN_rand)
                elif self.select_px_method == "allbeams":
                    # equal spaced sampling
                    coords, target = self.getRandomImgCoordsAllBeams(target,idx_x_start=100,step=self.BN_rand)
                    self.ray_n_samples = int(self.ray_n_samples_copy * coords[0][0].item()/self.H)

                else:
                    coords, target = self.getRandomImgCoordsByPercentage(target)

                n_pixels = len(coords)
                
                
                target_s = target[coords[:, 0], coords[:, 1]]
                render_out = self.renderer.render_sonar(self.H, self.W, self.phi_min, self.phi_max, self.r_min, self.r_max,  torch.Tensor(pose), n_pixels,
                                                        self.arc_n_samples, self.ray_n_samples, self.hfov, coords, self.r_increments, 
                                                        self.randomize_points, self.device, self.cube_center,cos_anneal_ratio=self.get_cos_anneal_ratio())

                

                gradient_error = render_out['gradient_error'] #.reshape(n_pixels, self.arc_n_samples, -1)

                eikonal_loss = gradient_error.sum()*(1/(self.arc_n_samples*self.ray_n_samples*n_pixels))

                variation_regularization = render_out['variation_error']*(1/(self.arc_n_samples*self.ray_n_samples*n_pixels))


                intensity_fine = render_out['color_fine']

                intensity_error = self.criterion(intensity_fine, target_s)*(1/n_pixels)

                # altimeter measurements
                pts = torch.from_numpy(self.data["PC"]).to(torch.float32).to(self.device) - self.cube_center.view(1,3)           
                render_out = self.renderer.render_altimeter(pts[:,:2], self.sdf_network)
                sdf_err =  torch.abs(pts[:,2]-render_out["z"][:,0]).sum()*(1/pts.shape[0])    
            
                loss = intensity_error*self.intensity_weight + eikonal_loss * self.igr_weight  + variation_regularization*self.variation_reg_weight+ sdf_err*self.bathy_weight

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    lossNG = intensity_error*self.intensity_weight  + eikonal_loss * self.igr_weight  + sdf_err*self.bathy_weight

                    loss_total += lossNG.cpu().numpy().item()
                    sum_intensity_loss += intensity_error.cpu().numpy().item()
                    sum_eikonal_loss += eikonal_loss.cpu().numpy().item()
                    sum_total_variational +=  variation_regularization.cpu().numpy().item()
                    sum_bathymetric_loss+=sdf_err.cpu().numpy().item()

                self.iter_step += 1
                self.update_learning_rate(self.learning_rate_decay_mode)

                del(target)
                del(target_s)
                # del(rays_d)
                del(pts)
                # del(dists)
                del(render_out)
                del(coords)
                
            with torch.no_grad():
                l = loss_total/len(i_train)
                iL =  sum_intensity_loss/len(i_train)
                eikL =  sum_eikonal_loss/len(i_train)
                varL =  sum_total_variational/len(i_train)
                bL = sum_bathymetric_loss/len(i_train)
                loss_arr.append(l)

            if i ==0 or i % self.save_freq == 0:
                logging.info('iter:{} ********************* SAVING CHECKPOINT ****************'.format(self.optimizer.param_groups[0]['lr']))
                self.save_checkpoint()

            if i % self.report_freq == 0:
                print('iter:{:8>d} "Loss={} | intensity Loss={} " | bathymetric loss={} | eikonal loss={} | total variation loss = {} | lr={} | inv_s={} | ray_n_samples={} | arc_n_samples={}'.format(
                    self.iter_step, l, iL, bL, eikL, varL, self.optimizer.param_groups[0]['lr'], np.exp(1*self.deviation_network.variance.item()), self.ray_n_samples, self.arc_n_samples) )

            if i == 0 or i % self.val_mesh_freq == 0:
                # self.validate_mesh(threshold = self.level_set)
                heightmap_nn, mae = self.export_heightmap_mae(mask_out=15)  # (-1,2)# numpy array 
                os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
                np.save(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.npy'.format(self.iter_step)), heightmap_nn)
                print('iter:{:8>d} "mae={} | '.format(
                    self.iter_step, mae) )

            # update hash encoding mask
            level = int(self.iter_step/self.sdf_network.steps_per_level) +1
            level = max(level, self.sdf_network.level_init)
            self.sdf_network.update_mask(level)

            self.writer.add_scalar('Loss/loss', l, self.iter_step)
            self.writer.add_scalar('Loss/intensity_loss', iL, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikL, self.iter_step)
            self.writer.add_scalar('Loss/bathymetric_loss', bL, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', varL, self.iter_step)
            self.writer.add_scalar('Statistics/inv_s', torch.exp(self.deviation_network.variance*1).item(), self.iter_step)
            self.writer.add_scalar('Statistics/lr', self.optimizer.param_groups[0]["lr"], self.iter_step)
            self.writer.add_scalar('Statistics/mae', mae, self.iter_step)
            self.writer.add_scalar('Statistics/level', level, self.iter_step)
            
            for beamform_k_azimuth_i in range(self.color_network.beamform_k_azimuth.shape[0]):
                self.writer.add_scalar('Statistics/beamform_k_azimuth'+str(beamform_k_azimuth_i), self.color_network.beamform_k_azimuth[beamform_k_azimuth_i], self.iter_step)
            for beamform_k_elevation_i in range(self.color_network.beamform_k_elevation.shape[0]):
                self.writer.add_scalar('Statistics/beamform_k_elevation'+str(beamform_k_elevation_i), self.color_network.beamform_k_elevation[beamform_k_elevation_i], self.iter_step)


    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

    def update_learning_rate(self, mode="cos"):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            if mode == "cos":
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
            elif mode=="linear":
                learning_factor = (1-alpha)** (progress*100)# (1-0.03)**100 ~ 0.05
            else:
                raise NotImplementedError

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    # def validate_heightmap(self):

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)

        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default="./confs/conf.conf")
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--PC_name', type=str, default="PC.npy")
    parser.add_argument('--heightmap_name', type=str, default="heightmap_gt.npy")
    parser.add_argument('--slurm_id', type=str, default="")
    parser.add_argument('--load_slurm_id', type=str, default="")
    parser.add_argument('--base_dir', type=str, default="./")



    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    # runner = Runner(args.conf, args.is_continue)
    runner = Runner(args)

    runner.set_params()
    runner.train()
