import torch
import torch.nn as nn
from torch.autograd import Variable

class post_refine(nn.Module):
    def __init__(self, opt, fc_unit=1024):
        super().__init__()

        if opt.refine:
           fc_unit = 1536

        fc_in = 3 * 2 *opt.n_joints
        fc_out = 2 * opt.n_joints

        self.post_refine = nn.Sequential(
            nn.Linear(fc_in, fc_unit),
            nn.ReLU(),
            nn.Dropout(0.5,inplace=True),
            nn.Linear(fc_unit, fc_out),
            nn.Sigmoid()
        )

    def forward(self, x, x_1):
        N, T, V,_ = x.size()
        x_in = torch.cat((x, x_1), -1)
        x_in = x_in.view(N, -1)

        score = self.post_refine(x_in).view(N,T,V,2)
        score_cm = Variable(torch.ones(score.size()), requires_grad=False).cuda() - score
        x_out = x.clone()
        x_out[:, :, :, :2] = score * x[:, :, :, :2] + score_cm * x_1[:, :, :, :2]

        return x_out


def get_uvd2xyz(uvd, gt_3D, cam):
    N, T, V,_ = uvd.size()

    dec_out_all = uvd.view(-1, T, V, 3).clone()
    root = gt_3D[:, :, 0, :].unsqueeze(-2).repeat(1, 1, V, 1).clone()
    enc_in_all = uvd[:, :, :, :2].view(-1, T, V, 2).clone()

    cam_f_all = cam[..., :2].view(-1,1,1,2).repeat(1,T,V,1)
    cam_c_all = cam[..., 2:4].view(-1,1,1,2).repeat(1,T,V,1)

    z_global = dec_out_all[:, :, :, 2]
    z_global[:, :, 0] = root[:, :, 0, 2]
    z_global[:, :, 1:] = dec_out_all[:, :, 1:, 2] + root[:, :, 1:, 2]
    z_global = z_global.unsqueeze(-1)
    
    uv = enc_in_all - cam_c_all
    xy = uv * z_global.repeat(1, 1, 1, 2) / cam_f_all
    xyz_global = torch.cat((xy, z_global), -1)
    xyz_offset = (xyz_global - xyz_global[:, :, 0, :].unsqueeze(-2).repeat(1, 1, V, 1))

    return xyz_offset


def refine_model(model_refine, output_3D, input_2D, gt_3D, batch_cam, pad, root_joint):
    input_2D_single = input_2D[:, pad, :, :].unsqueeze(1)
    
    if output_3D.size(1) > 1:
        output_3D_single = output_3D[:, pad, :, :].unsqueeze(1)
    else:
        output_3D_single = output_3D

    if gt_3D.size(1) > 1:
        gt_3D_single = gt_3D[:, pad, :, :].unsqueeze(1)
    else:
        gt_3D_single = gt_3D

    uvd = torch.cat((input_2D_single, output_3D_single[:, :, :, 2].unsqueeze(-1)), -1)
    xyz = get_uvd2xyz(uvd, gt_3D_single, batch_cam)
    xyz[:, :, root_joint, :] = 0
    refine_out = model_refine(output_3D_single, xyz)

    return refine_out

