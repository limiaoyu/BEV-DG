import torch
import torch.nn as nn
import torch.nn.functional as F

from xmuda.models.resnet34_unet import UNetResNet34
from xmuda.models.scn_unet import UNetSCN
from xmuda.data.utils.bev_fusion import BEV_fusion

class Net2DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_2d,
                 backbone_2d_kwargs
                 ):
        super(Net2DSeg, self).__init__()

        # 2D image network
        if backbone_2d == 'UNetResNet34':
            self.net_2d = UNetResNet34(**backbone_2d_kwargs)
            feat_channels = 64
        else:
            raise NotImplementedError('2D backbone {} not supported'.format(backbone_2d))

        # segmentation head
        self.linear = nn.Linear(feat_channels, num_classes)
        # fusion
        self.linear1 = nn.Linear(80,feat_channels)
        self.linear3 = nn.Linear(feat_channels*2,feat_channels)
        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(feat_channels, num_classes)

    def forward(self, data_batch, feats_2d=None, batch_bev_feats=None):
        if feats_2d==None and batch_bev_feats==None:
            # (batch_size, 3, H, W)
            img = data_batch['img']
            img_indices = data_batch['img_indices']

            # 2D network
            x = self.net_2d(img)

            # 2D-3D feature lifting
            img_feats = []
            for i in range(x.shape[0]):
                img_feats.append(x.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
            img_feats = torch.cat(img_feats, 0)

            return img_feats
        else:

            batch_bev_feats = F.relu(self.linear1(batch_bev_feats))
            fusion_feats = torch.cat([feats_2d, batch_bev_feats], dim=1)
            fusion_feats = self.linear3(fusion_feats)
            # linear
            x = self.linear(fusion_feats)

            preds = {
                'feats': feats_2d,
                'seg_logit': x,
            }

            if self.dual_head:
                preds['seg_logit2'] = self.linear2(feats_2d)

            return preds


class Net3DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_3d,
                 backbone_3d_kwargs,
                 ):
        super(Net3DSeg, self).__init__()

        # 3D network
        if backbone_3d == 'SCN':
            self.net_3d = UNetSCN(**backbone_3d_kwargs)
        else:
            raise NotImplementedError('3D backbone {} not supported'.format(backbone_3d))

        # segmentation head
        self.linear = nn.Linear(self.net_3d.out_channels, num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(self.net_3d.out_channels, num_classes)

        # linear_fusion 80-16
        self.linear3 = nn.Linear(80,self.net_3d.out_channels)

        self.linear4 = nn.Linear(self.net_3d.out_channels*2,self.net_3d.out_channels)

        self.linear5 = nn.Linear(80,32)

    def forward(self, data_batch, feats_3d=None, batch_bev_feats=None, batch_vector=None):
        if feats_3d==None and batch_bev_feats==None and batch_vector==None:
            feats = self.net_3d(data_batch['x'])
            return feats
        else:

            batch_bev_feats = F.relu(self.linear3(batch_bev_feats))
            feats_fusion = torch.cat([feats_3d,batch_bev_feats],dim=1)
            feats_fusion = self.linear4(feats_fusion)
            x = self.linear(feats_fusion)
            preds = {
                'feats': feats_3d,
                'seg_logit': x,
            }

            if self.dual_head:
                preds['seg_logit2'] = self.linear2(feats_3d)
            return preds


def test_Net2DSeg():
    # 2D
    batch_size = 2
    img_width = 400
    img_height = 225

    # 3D
    num_coords = 2000
    num_classes = 11

    # 2D
    img = torch.rand(batch_size, 3, img_height, img_width)
    u = torch.randint(high=img_height, size=(batch_size, num_coords // batch_size, 1))
    v = torch.randint(high=img_width, size=(batch_size, num_coords // batch_size, 1))
    img_indices = torch.cat([u, v], 2)

    # to cuda
    img = img.cuda()
    img_indices = img_indices.cuda()

    net_2d = Net2DSeg(num_classes,
                      backbone_2d='UNetResNet34',
                      backbone_2d_kwargs={},
                      dual_head=True)

    net_2d.cuda()
    out_dict = net_2d({
        'img': img,
        'img_indices': img_indices,
    })
    for k, v in out_dict.items():
        print('Net2DSeg:', k, v.shape)


def test_Net3DSeg():
    in_channels = 1
    num_coords = 2000
    full_scale = 4096
    num_seg_classes = 11

    coords = torch.randint(high=full_scale, size=(num_coords, 3))
    feats = torch.rand(num_coords, in_channels)

    feats = feats.cuda()

    net_3d = Net3DSeg(num_seg_classes,
                      dual_head=True,
                      backbone_3d='SCN',
                      backbone_3d_kwargs={'in_channels': in_channels})

    net_3d.cuda()
    out_dict = net_3d({
        'x': [coords, feats],
    })
    for k, v in out_dict.items():
        print('Net3DSeg:', k, v.shape)


if __name__ == '__main__':
    test_Net2DSeg()
    test_Net3DSeg()
