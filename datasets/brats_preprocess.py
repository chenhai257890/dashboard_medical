'this file is for transfering 3D BRaTs MRI to 2D Slices for training'
import os
import argparse

import numpy as np
import nibabel as nib


def nii2np_test(img_root, img_name, upper_per=0.9, lower_per=0.02, output_root_test='./datasets/data/test', modality=['t1'], slice_id=70):
    '''generate image for each modality'''
    for mod_num in range(len(modality)):
        img_file = os.path.join(img_root, img_name)
        img = nib.load(img_file)
        img = (img.get_fdata())
        img_original = img

        '''normalize the [lower_per, lower_per] of the brain to [-3,3]'''
        perc_upper = ((img > 0).sum() * (1 - upper_per)) / (img.shape[0] * img.shape[1] * img.shape[
            2])  # find the proportion of top (upper_per)% intensity of the brain within the whole 3D image
        perc_lower = ((img > 0).sum() * lower_per) / (img.shape[0] * img.shape[1] * img.shape[2])
        upper_value = np.percentile(img, (1 - perc_upper) * 100)
        lower_value = np.percentile(img, perc_lower * 100)
        img_half = (upper_value - lower_value) / 2
        img = (img - img_half) / (upper_value - lower_value) * 6  # normalize the [lower_per, upper_per] of the brain to [-3,3]

        dirs_mod = os.path.join(output_root_test, modality[mod_num])
        if not os.path.exists(dirs_mod):
            os.makedirs(dirs_mod)
        filename = os.path.join(dirs_mod, modality[mod_num])
        img_slice = img[:, :, slice_id]
        np.save(filename, img_slice)

        dirs_brainmask = os.path.join(output_root_test, 'brainmask')
        if not os.path.exists(dirs_brainmask):
            os.makedirs(dirs_brainmask)
        filename_brainmask = os.path.join(dirs_brainmask, 'brainmask')
        img_brainmask = (img_original > 0).astype(int)
        img_slice_brainmask = img_brainmask[:, :, slice_id]
        np.save(filename_brainmask, img_slice_brainmask)
