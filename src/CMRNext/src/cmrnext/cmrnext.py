import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rospy
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import visibility
from cmrnext.camera_model import CameraModel
from cmrnext.model.get_model import get_model
from cmrnext.utils import (
    depth_to_3D,
    downsample_depth,
    overlay_imgs,
    quat2mat,
    quaternion_from_matrix,
    rotate_back,
    rotate_forward,
    show_or_save_plt,
    tvector2mat,
)
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image
from torch import nn


class CMRNext():

    def __init__(self,
                 weights_paths: List[str],
                 cam_params: List[float],
                 target_shape: Tuple[int, int],
                 downsample: bool = True) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cam_params = cam_params
        self.cam_model = CameraModel()
        self.cam_model.focal_length = cam_params[:2]
        self.cam_model.principal_point = cam_params[2:]
        self.downsample = downsample
        self.show = True  # ONLY FOR DEBUG
        self.pub_init = rospy.Publisher('cmrnext/visualization_init',
                                        numpy_msg(Image),
                                        queue_size=10)
        self.pub_final = rospy.Publisher('cmrnext/visualization_final',
                                         numpy_msg(Image),
                                         queue_size=10)
        self.cv_bridge = CvBridge()
        checkpoint = torch.load(os.path.join(weights_paths[0]), map_location='cpu')
        _config = checkpoint['config']
        _config['reverse'] = _config['al_contrario']  # Sorry for the Italian name of the parameter
        _config['pnp'] = 'cuda_cv'

        self._config = _config
        self.mean_torch = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std_torch = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self.target_shape = target_shape

        self.models = []
        print(f'Loading weights from {weights_paths}')
        for weights_path in weights_paths:
            checkpoint = torch.load(weights_path, map_location='cpu')
            if 'der_type' not in _config:
                _config['der_type'] = 'NLL'
            if 'unc_freeze' not in _config:
                _config['unc_freeze'] = False
            if 'context_encoder' not in _config:
                _config['context_encoder'] = 'rgb'
            model = get_model(_config)
            saved_state_dict = checkpoint['state_dict']
            model.load_state_dict(saved_state_dict, strict=True)
            model = model.to(self.device)
            model.eval()
            self.models.append(model)


    # pylint: disable-next=inconsistent-return-statements
    def process_pair(
        self, image: torch.Tensor, point_cloud: torch.Tensor, initial_calib: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Process and convert inputs
        real_shape = [image.shape[0], image.shape[1], image.shape[2]]
        point_cloud = point_cloud.clone().t().to(self.device)
        initial_calib = initial_calib.to(self.device)
        # Convert point cloud based on initial calibration
        point_cloud = rotate_forward(point_cloud, initial_calib)
        pc_rotated = point_cloud
        image = image.to(self.device)

        # Project point cloud into virtual image plane placed at 'initial_calib'
        RTs = [torch.eye(4).float().to(self.device)]
        uv, depth, _, _ = self.cam_model.project_pytorch(pc_rotated, real_shape, None)
        uv = uv.t().int().contiguous()
        depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
        depth_img += 1000.
        depth_img = visibility.depth_image(uv, depth, depth_img, uv.shape[0], real_shape[1],
                                           real_shape[0])
        depth_img[depth_img == 1000.] = 0.
        # Here add occlusion filter if using map instead of single scan
        depth_img_no_occlusion = depth_img
        uv = uv.long()

        # Check valid indexes: as multiple points MIGHT be projected into the same pixel, keep only
        # the points that are actually projected
        indexes = depth_img_no_occlusion[uv[:, 1], uv[:, 0]] == depth
        depth_img_no_occlusion /= self._config['max_depth']
        depth_img_no_occlusion = depth_img_no_occlusion.unsqueeze(0)
        uv = uv[indexes]

        # Convert depth map into 3D point cloud, to make sure that we have 1-to-1 correspondence
        # between 2d and 3d, might not be necessary
        points_3D = depth_to_3D(uv.float(),
                                depth[indexes],
                                self.cam_model)

        # Normalize image
        rgb = image
        rgb = rgb / 255.
        if self._config['normalize_images']:
            rgb = (rgb - self.mean_torch) / self.std_torch
        rgb = rgb.permute(2, 0, 1)
        image = rgb

        # flow_mask containts 1 in pixels that have a point projected
        flow_mask = torch.zeros((real_shape[0], real_shape[1]), device=self.device, dtype=torch.int)
        flow_mask[uv[:, 1], uv[:, 0]] = 1
        points_3D = points_3D.clone()
        depth = depth[indexes].clone()
        shape_pad = [0, 0, 0, 0]

        if self.show:
            std = [0.229, 0.224, 0.225]
            mean = [0.485, 0.456, 0.406]

            rgbshow = rgb.clone().cpu().permute(1, 2, 0).numpy()
            rgbshow = rgbshow * std + mean
            rgbshow = rgbshow.clip(0, 1)

            valid_indexes = flow_mask[uv[:, 1], uv[:, 0]] == 1

            overlay = overlay_imgs(rgb, depth_img_no_occlusion.unsqueeze(0))
            overlay = overlay * 255.
            overlay = np.uint8(overlay)
            overlay_msg = self.cv_bridge.cv2_to_imgmsg(overlay, encoding='rgb8')
            if self.pub_init.get_num_connections() > 0:
                self.pub_init.publish(overlay_msg)
                rospy.logwarn('Published initial estimate (overlay image).')

        if self.downsample:
            original_img = rgb.clone()
            rgb = nn.functional.interpolate(rgb.unsqueeze(0), scale_factor=0.5)[0]
            depth_img_no_occlusion = downsample_depth(
                depth_img_no_occlusion.permute(1, 2, 0).contiguous(), 2)
            depth_img_no_occlusion = depth_img_no_occlusion.permute(2, 0, 1)

            shape_pad[3] = (self.target_shape[0] - real_shape[0] // 2)
            shape_pad[1] = (self.target_shape[1] - real_shape[1] // 2)

            rgb = F.pad(rgb, shape_pad)
            depth_img_no_occlusion = F.pad(depth_img_no_occlusion, shape_pad)

            shape_pad[3] = (self.target_shape[0] * 2 - real_shape[0])
            shape_pad[1] = (self.target_shape[1] * 2 - real_shape[1])
            flow_mask = F.pad(flow_mask, shape_pad)
            original_img = F.pad(original_img, shape_pad)

        else:
            shape_pad[3] = (self.target_shape[0] - real_shape[0])
            shape_pad[1] = (self.target_shape[1] - real_shape[1])

            rgb = F.pad(rgb, shape_pad)
            depth_img_no_occlusion = F.pad(depth_img_no_occlusion, shape_pad)
            flow_mask = F.pad(flow_mask, shape_pad)
            original_img = rgb.clone()

        # Convert depth into fourier frequencies, similar to the positional encoding used in NERF
        if self._config['fourier_levels'] >= 0:
            depth_img_no_occlusion = depth_img_no_occlusion.squeeze()
            mask = (depth_img_no_occlusion > 0).clone()
            fourier_feats = []
            for L in range(self._config['fourier_levels']):
                fourier_feat = depth_img_no_occlusion * np.pi * 2**L
                fourier_feats.append(fourier_feat.sin())
                fourier_feats.append(fourier_feat.cos())
            depth_img_no_occlusion = torch.stack(fourier_feats + [depth_img_no_occlusion])
            depth_img_no_occlusion = depth_img_no_occlusion * mask.unsqueeze(0)

        rgb_input = rgb.unsqueeze(0)
        lidar_input = depth_img_no_occlusion.unsqueeze(0)

        for iteration, model in enumerate(self.models):
            # Predict 'flow': dense lidar depth map to rgb pixel displacements
            with torch.no_grad():
                predicted_flow = model(rgb_input, lidar_input)
                predicted_flow, predicted_uncertainty = predicted_flow
                # Upsample if necessary
                if self.downsample:
                    predicted_flow = list(predicted_flow)
                    # pylint: disable-next=consider-using-enumerate
                    for scale in range(len(predicted_flow)):
                        predicted_flow[scale] *= 2
                        predicted_flow[scale] = F.interpolate(predicted_flow[scale],
                                                              scale_factor=2,
                                                              mode='bilinear')
                        if self._config['uncertainty']:
                            predicted_uncertainty[scale] = F.interpolate(
                                predicted_uncertainty[scale], scale_factor=2, mode='bilinear')
            up_flow = predicted_flow[-1]
            up_flow = up_flow[0].permute(1, 2, 0)

            if self._config['reverse']:
                new_uv = uv.float() - up_flow[uv[:, 1], uv[:, 0]]
            else:
                new_uv = uv.float() + up_flow[uv[:, 1], uv[:, 0]]

            valid_indexes = flow_mask[uv[:, 1], uv[:, 0]] == 1

            # Check only pixels that are within the image border
            valid_indexes = valid_indexes & (new_uv[:, 0] < flow_mask.shape[1])
            valid_indexes = valid_indexes & (new_uv[:, 1] < flow_mask.shape[0])
            valid_indexes = valid_indexes & (new_uv[:, 0] >= 0)
            valid_indexes = valid_indexes & (new_uv[:, 1] >= 0)
            new_uv = new_uv[valid_indexes]

            
            points_2d = new_uv.cpu().numpy()
            obj_coord = points_3D[valid_indexes][:, :3].cpu().numpy()
            cam_mat = self.cam_model.get_matrix()

            # If last iteration of CMRNext, compute the final correspondences
            if iteration == len(self.models) - 1:

                # convert points_3D (which are in the reference frame of the previous iteration)
                # to the original lidar frame
                points_3D_orig = torch.mm(RTs[-1], points_3D[valid_indexes].T).T
                points_3D_orig = rotate_back(points_3D_orig, initial_calib)

                #Get correspondences uncertainty
                if self._config['uncertainty']:
                    uncertainties = predicted_uncertainty[-1][0].permute(1, 2, 0)
                    uncertainties = uncertainties[[uv[:, 1], uv[:, 0]]][valid_indexes]
                    uncertainties = uncertainties.sum(axis=1)
                else:
                    uncertainties = None

                # save final 2d-3d correspondences
                final_correspondences = new_uv, points_3D_orig

                # ==============================================================================
                # Compute Final prediction by CMRNext, not needed for optimization,
                # but can be used for comparison

                cuda_pnp_final = cv2.pythoncuda.cudaPnP(
                    final_correspondences[1][:, :3].cpu().numpy().astype(np.float32).copy(),
                    final_correspondences[0].cpu().numpy().astype(np.float32).copy(),
                    points_3D_orig.shape[0], 200, 2., cam_mat.astype(np.float32))
                transl = cuda_pnp_final[0, :3]
                rot_mat = cuda_pnp_final[:, 3:6].T
                rot_mat, _ = cv2.Rodrigues(rot_mat)
                rot_mat = torch.tensor(rot_mat)
                T_predicted = tvector2mat(torch.tensor(transl))
                R_predicted = torch.eye(4)
                R_predicted[:3, :3] = rot_mat.clone().detach()
                # Final prediction by CMRNext
                RT_predicted_final = torch.mm(T_predicted, R_predicted)
                # ==============================================================================

            # Predict relative transformation based on CMRNext correspondences
            # for iterative refinement
            if self._config['pnp'] == 'cpu':
                res = cv2.solvePnPRansac(obj_coord.astype(np.float32).copy(),
                                         points_2d.astype(np.float32).copy(),
                                         cam_mat,
                                         np.array([0., 0., 0., 0.]),
                                         iterationsCount=1000,
                                         reprojectionError=2)
                transl = res[2][[0, 1, 2], :]
                rot_mat, _ = cv2.Rodrigues(res[1])
            elif self._config['pnp'] == 'cuda_cv':
                cuda_pnp = cv2.pythoncuda.cudaPnP(
                    obj_coord.astype(np.float32).copy(),
                    points_2d.astype(np.float32).copy(), obj_coord.shape[0], 200, 2.,
                    cam_mat.astype(np.float32))

                transl = cuda_pnp[0, [0, 1, 2]]
                rot_mat = cuda_pnp[:, 3:6].T
                rot_mat, _ = cv2.Rodrigues(rot_mat)
            else:
                raise NotImplementedError('Not implemented')

            transl = torch.tensor(transl).float().squeeze().cuda()
            rot_mat = torch.tensor(rot_mat)
            pred_quaternion = quaternion_from_matrix(rot_mat)

            R_predicted = quat2mat(pred_quaternion).cuda()
            T_predicted = tvector2mat(transl)
            RT_predicted = torch.mm(T_predicted, R_predicted)
            composed = torch.mm(RTs[iteration], RT_predicted.inverse())
            RTs.append(composed)

            # Rotate point cloud based on predicted pose, and generate new
            # inputs for the next iteration
            rotated_point_cloud = rotate_forward(point_cloud, RTs[-1])

            uv, depth, _, _, proj_indexes = self.cam_model.project_pytorch(rotated_point_cloud,
                                                                           real_shape,
                                                                           None,
                                                                           return_indexes=True)
            uv = uv.t().int().contiguous()
            depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
            depth_img += 1000.
            depth_img = visibility.depth_image(uv, depth, depth_img, uv.shape[0], real_shape[1],
                                               real_shape[0])
            depth_img[depth_img == 1000.] = 0.

            # Here add occlusion filter if using map instead of single scan
            depth_img_no_occlusion = depth_img

            uv = uv.long()
            indexes = depth_img_no_occlusion[uv[:, 1], uv[:, 0]] == depth

            depth_img_no_occlusion /= self._config['max_depth']
            depth_img_no_occlusion = depth_img_no_occlusion.unsqueeze(0)

            uv = uv[indexes]

            points_3D = depth_to_3D(uv.float(),
                                    depth[indexes],
                                    self.cam_model)

            rgb = image
            flow_mask = torch.zeros((real_shape[0], real_shape[1]), device='cuda', dtype=torch.int)
            flow_mask[uv[:, 1], uv[:, 0]] = 1

            depth = depth[indexes].clone()
            shape_pad = [0, 0, 0, 0]

            if self.downsample:
                original_img = rgb.clone()
                rgb = nn.functional.interpolate(rgb.unsqueeze(0), scale_factor=0.5)[0]
                depth_img_no_occlusion = downsample_depth(
                    depth_img_no_occlusion.permute(1, 2, 0).contiguous(), 2)
                depth_img_no_occlusion = depth_img_no_occlusion.permute(2, 0, 1)

                shape_pad[3] = (self.target_shape[0] - real_shape[0] // 2)
                shape_pad[1] = (self.target_shape[1] - real_shape[1] // 2)

                rgb = F.pad(rgb, shape_pad)
                depth_img_no_occlusion = F.pad(depth_img_no_occlusion, shape_pad)

                shape_pad[3] = (self.target_shape[0] * 2 - real_shape[0])
                shape_pad[1] = (self.target_shape[1] * 2 - real_shape[1])
                flow_mask = F.pad(flow_mask, shape_pad)
                original_img = F.pad(original_img, shape_pad)

            else:
                shape_pad[3] = (self.target_shape[0] - real_shape[0])
                shape_pad[1] = (self.target_shape[1] - real_shape[1])

                rgb = F.pad(rgb, shape_pad)
                depth_img_no_occlusion = F.pad(depth_img_no_occlusion, shape_pad)
                flow_mask = F.pad(flow_mask, shape_pad)
                original_img = rgb.clone()

            if self._config['fourier_levels'] >= 0:
                depth_img_no_occlusion = depth_img_no_occlusion.squeeze()
                mask = (depth_img_no_occlusion > 0).clone()
                fourier_feats = []
                for L in range(self._config['fourier_levels']):
                    fourier_feat = depth_img_no_occlusion * np.pi * 2**L
                    fourier_feats.append(fourier_feat.sin())
                    fourier_feats.append(fourier_feat.cos())
                depth_img_no_occlusion = torch.stack(fourier_feats + [depth_img_no_occlusion])
                depth_img_no_occlusion = depth_img_no_occlusion * mask.unsqueeze(0)

            rgb_input = rgb.unsqueeze(0)
            lidar_input = depth_img_no_occlusion.unsqueeze(0)

            # Show results, for debug only
            if iteration == len(self.models) - 1:
                gt_uv, gt_depth, _, _ = self.cam_model.project_pytorch(
                    rotated_point_cloud, real_shape, None)
                gt_uv = gt_uv.t().int().contiguous()

                new_depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                new_depth_img += 1000.
                new_depth_img = visibility.depth_image(gt_uv.int().contiguous(), gt_depth,
                                                       new_depth_img, gt_uv.shape[0], real_shape[1],
                                                       real_shape[0])
                new_depth_img[new_depth_img == 1000.] = 0.

                new_depth_img_no_occlusion = new_depth_img
                lidar_flow = new_depth_img_no_occlusion.unsqueeze(0).unsqueeze(0)
                lidar_flow /= self._config['max_depth']
                overlay = overlay_imgs(image, lidar_flow)
                if self.show:
                    show_or_save_plt(overlay, '/data/overlay.png', save_images=False)
                overlay = overlay * 255.
                overlay = np.uint8(overlay)
                overlay_msg = self.cv_bridge.cv2_to_imgmsg(overlay, encoding='rgb8')
                self.pub_final.publish(overlay_msg)

            # Return final correspondences, uncertainties, and final extrinsic
            # calibration predicted by CMRNext (not used for optimization)
            if iteration == len(self.models) - 1:
                return final_correspondences, uncertainties, RT_predicted_final
