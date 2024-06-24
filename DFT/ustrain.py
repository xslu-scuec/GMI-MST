# -*- encoding: utf-8 -*-
import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.optim import Adam

from logger import get_logger
from networks import FNestedUnet


data_base_dir = r"D:\NMDF\norm01"
model_out_dir = r"F:\dfeats_ext\fold1"

n_epochs = 20
n_patients = 100
deep_supervision = False
arry_pnt = np.arange(n_patients)
train_pnt = np.array(arry_pnt[20:])
#train_pnt = np.append(train_pnt, arry_pnt[100:])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FNestedUnet(in_shape=[192,192,16], in_channel=1, out_channel=1, deep_supervision=deep_supervision).to(device)
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

logger = get_logger(os.path.join(model_out_dir, 'dfext_train.log'))
for ep in range(n_epochs):
    model.train()
    np.random.shuffle(train_pnt)

    for pat in range(len(train_pnt)):
        pat_num = str('%03d' % (train_pnt[pat]+1))
        pat_data = glob.glob(os.path.join(data_base_dir, 'pat'+pat_num, '*.nii.gz'))

        for pha in range(2):
            pat_img = nib.load(pat_data[pha])
            pat_v = pat_img.get_fdata()[np.newaxis, ..., np.newaxis]
            input_vt = torch.from_numpy(pat_v).to(device).float()
            input_vt = input_vt.permute(0, 4, 1, 2, 3)

            optimizer.zero_grad()
            _, _, _, _, pred_vt = model(input_vt)
            loss = criterion(pred_vt, input_vt)

            loss.backward()
            optimizer.step()

            logger.info("Epoch:[{:0>5d}/{:0>5d}/{:0>5d}]\t train loss={:.5f}\t".format(ep+1, train_pnt[pat]+1, pha+1, loss.item()))

    # Save model checkpoint
    if (ep+1) % 5 == 0:
        save_file_name = os.path.join(model_out_dir, '%d.ckpt' % (ep+1))
        torch.save(model.state_dict(), save_file_name)

logger.info("train finished!")
