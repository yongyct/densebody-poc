from .smpl_util import SMPLModel
from .uv_map_util import UV_Map_Generator
import os
from cv2 import imwrite
import torch
import numpy as np


class Visualizer():
    def __init__(self, conf):
        os.chdir('./densebody_poc/resources/')
        self.UV_sampler = UV_Map_Generator(
            UV_height=conf.IM_SIZE,
            UV_pickle=conf.UV_MAP + '.pickle'
        )
        # Only use save obj
        self.model = SMPLModel(
            device=None,
            model_path='model_lsp.pkl',
        )
        if conf.PHASE == 'train':
            self.save_root = './output/{}/{}'.format(conf.CHECKPOINT_DIR, conf.NAME)
        elif conf.PHASE == 'test':
            self.save_root = './output/{}/{}'.format(conf.RESULTS_DIR, conf.NAME)
        else:
            self.save_root = './output/{}/{}'.format(conf.RESULTS_DIR, conf.NAME)
        if not os.path.isdir(self.save_root):
            os.makedirs(self.save_root)
        os.chdir('./../../')

    @staticmethod
    def tensor2im(tensor):
        # input: cuda tensor (CHW) [-1,1]; output: numpy uint8 [0,255] (HWC)
        return ((tensor.detach().cpu().numpy().transpose(1, 2, 0) + 1.) * 127.5).astype(np.uint8)

    @staticmethod
    def tensor2numpy(tensor):
        # input: cuda tensor (CHW) [-1,1]; output: numpy float [0,1] (HWC)
        return (tensor.detach().cpu().numpy().transpose(1, 2, 0) + 1.) / 2.

    def save_results(self, visual_dict, epoch, batch):
        img_name = self.save_root + '{:03d}_{:05d}.png'.format(epoch, batch)
        obj_name = self.save_root + '{:03d}_{:05d}.obj'.format(epoch, batch)
        ply_name = self.save_root + '{:03d}_{:05d}.ply'.format(epoch, batch)
        imwrite(img_name,
                self.tensor2im(torch.cat([im for im in visual_dict.values()], dim=2))
                )
        fake_UV = visual_dict['fake_UV']
        # fake_UV = (fake_UV + 1) / 2 * 255
        resampled_verts = self.UV_sampler.resample(self.tensor2numpy(fake_UV))
        self.UV_sampler.write_ply(ply_name, resampled_verts)
        self.model.write_obj(resampled_verts, obj_name)

