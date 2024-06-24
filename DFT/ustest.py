# -*- encoding: utf-8 -*-
import os
import glob
import numpy as np
import nibabel as nib
import torch
from networks import FNestedUnet


data_base_dir = r"D:\NMDF\norm01"
out_feat_dir = r"F:\dfeats_ext\datasets\ACDC"

n_patients = 100
deep_supervision = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FNestedUnet(in_shape=[192,192,16], in_channel=1, out_channel=1, deep_supervision=deep_supervision).to(device)
model.load_state_dict(torch.load(r"F:\dfeats_ext\fold1\20.ckpt", map_location=lambda storage, loc: storage))
model.eval()

for pat in range(n_patients):
    if (pat+1) <= 20:
        pat_num = str('%03d' % (pat+1))
        os.mkdir(os.path.join(out_feat_dir, 'p'+pat_num))
        pat_data = glob.glob(os.path.join(data_base_dir, 'pat'+pat_num, '*.nii.gz'))

        for pha in range(2):
            pat_img_name = os.path.basename(pat_data[pha]).split('.')[0]
            pat_img = nib.load(pat_data[pha])
            pat_header = pat_img.header.copy()
            pat_v = pat_img.get_fdata()[np.newaxis, ..., np.newaxis]
            input_vt = torch.from_numpy(pat_v).to(device).float()
            input_vt = input_vt.permute(0, 4, 1, 2, 3)

            with torch.no_grad():
                pred0_vt, pred1_vt, pred_vt = model(input_vt)
                s0 = torch.squeeze(pred0_vt).detach().cpu().numpy()
                s1 = torch.squeeze(pred1_vt).detach().cpu().numpy()
                recon = torch.squeeze(pred_vt).detach().cpu().numpy()

                out_recon = nib.nifti1.Nifti1Image(recon, None, pat_header)
                nib.save(out_recon, os.path.join(out_feat_dir, 'p'+pat_num, pat_img_name+".nii.gz"))

                for fno in range(s0.shape[0]):
                    out_s0 = nib.nifti1.Nifti1Image(s0[fno,:,:,:], None, pat_header)
                    feat_no = str('%02d' % (fno+1))
                    nib.save(out_s0, os.path.join(out_feat_dir, 'p'+pat_num, pat_img_name+"_"+feat_no+".nii.gz"))

                for fno in range(s1.shape[0]):
                    out_s1 = nib.nifti1.Nifti1Image(s1[fno,:,:,:], None, pat_header)
                    feat_no = str('%02d' % (s0.shape[0]+fno+1))
                    nib.save(out_s1, os.path.join(out_feat_dir, 'p'+pat_num, pat_img_name+"_"+feat_no+".nii.gz"))



