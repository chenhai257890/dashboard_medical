import os
import sys
import argparse
import random

import numpy as np
import torch as th
import blobfile as bf
from pathlib import Path
from datasets.brats_preprocess import nii2np_test
from datasets import loader
from configs import get_config
from utils import logger
from utils.script_util import create_model
from utils.metrics import sensitivity_metric, precision_metric, dice_score
sys.path.append(str(Path.cwd()))
from torchmetrics.functional import structural_similarity_index_measure
from models.test_model import iter_mask_refinement, validation_thres, iter_mask_refinement_bestthres
import matplotlib.pyplot as plt
import streamlit as st
from scipy.ndimage import zoom
import nibabel as nib
def normalize(img, _min=None, _max=None):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def main():
    #use_gpus = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpus)
    config = get_config.file_from_dataset('brats')

    experiment_name_first_iter = "first_iter_brats_t1"
    experiment_name_masked_autoencoder = "masked_autoencoder_brats_t1"

    input_mod = 't1'
    test_loader = loader.get_data_loader('brats', '/mount/src/dashboard_medical/datasets/data', config, input_mod, split_set='test', generator=False)

    model_first_iter = create_model(config, image_level_cond=False)
    model_masked_autoencoder = create_model(config, image_level_cond=True)


    filename = "model090000.pt"
    path_first_iter = bf.join('/mount/src/dashboard_medical/model_save', experiment_name_first_iter, filename)
    path_masked_autoencoder = bf.join('/mount/src/dashboard_medical/model_save', experiment_name_masked_autoencoder, filename)

    model_first_iter.load_state_dict(
        th.load(path_first_iter, map_location=th.device('cpu'))
    )
    model_first_iter.to(th.device('cpu'))
    model_masked_autoencoder.load_state_dict(
        th.load(path_masked_autoencoder, map_location=th.device('cpu'))
    )
    model_masked_autoencoder.to(th.device('cpu'))

    if config.model.use_fp16:
        model_first_iter.convert_to_fp16()

    model_first_iter.eval()
    model_masked_autoencoder.eval()


    num_sample = 0
    img_pred_mask_all = np.zeros(
        (len(test_loader.dataset), config.model.num_input_channels * 1, config.model.image_size,
         config.model.image_size))


    num_iter = 0
    for test_data_dict in enumerate(test_loader):
        test_data_input = test_data_dict[1].pop('input').cpu()
        brain_mask = test_data_dict[1].pop('brainmask').cpu()
        """
        final_mask, final_reconstruction = iter_mask_refinement(
            model_masked_autoencoder, model_first_iter, test_data_input,
            brain_mask, experiment_name_masked_autoencoder
        )
        """
        #print(test_data_dict[0], type(test_data_dict[0]))
        final_mask, final_reconstruction = iter_mask_refinement(
            model_masked_autoencoder, model_first_iter, test_data_input,
            brain_mask, experiment_name_masked_autoencoder, test_data_dict[0]
        )
        img_pred_mask_all[num_sample:num_sample + test_data_input.shape[0]] = final_mask.cpu().numpy()
        num_sample += test_data_input.shape[0]
        num_iter += 1
    return final_mask
        






def reseed_random(seed):
    random.seed(seed)  # python random generator
    np.random.seed(seed)  # numpy random generator
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
st.set_page_config(page_title="Brain Lesion Detection Dashboard", layout="wide")

# ---------------------------- UI -----------------------------
st.title("🧠 Brain MRI Lesion Detection Dashboard")
# 创建两列布局：左边为上传区域，右边为展示区域
col1, col2 = st.columns([1, 3])

# ---------------------------- 左侧区域 ----------------------------
with col1:
    # 文件上传控件
    uploaded_file = st.file_uploader("上传 .nii.gz 文件", type=["nii.gz"])
    slice_num = st.slider("Slice", 0, 154, 77, key="slider1")
    # 切片位置滑块
    if uploaded_file:
        # 读取 NIfTI 文件
        file_path = os.path.join("./data", uploaded_file.name)
        nii_image = nib.load(file_path)
        img_data = nii_image.get_fdata()  # 获取图像数据

        # 获取图像的维度
        depth, height, width = img_data.shape
        st.write(f"图像维度: {depth} x {height} x {width}")
        slice_data = img_data[:, :, slice_num]


       

# ---------------------------- 右侧区域 ----------------------------
with col2:
    st.subheader("病灶检测区")
    # -------------------- 上半部分：展示病灶掩膜 --------------------
    left, middle1, middle2, right = st.columns(4, vertical_alignment="center")
    left.subheader("当前切片")
    b = middle1.button("Detect")
    middle2.subheader("病灶检测结果")
    right.subheader("叠加图")
    if uploaded_file:
        plt.figure(figsize=(5, 5))
        plt.imshow(slice_data.T, cmap="gray")
        left.pyplot(plt)
   
        if b:
        # 假设你有一个封装好的病灶检测函数 `detect_lesion` 
            nii2np_test('/mount/src/dashboard_medical/data', uploaded_file.name, slice_id=slice_num)
            lesion_mask = main()  # 你需要提供该函数

        # 显示病灶掩膜
            plt.figure(figsize=(5, 5))
            plt.imshow(lesion_mask.squeeze().T, cmap="gray")  # 热力图表示病灶掩膜
            middle2.pyplot(plt)
        
        # 显示原图与病灶掩膜叠加图

            overlay = np.copy(slice_data)
            overlay = zoom(overlay, (128 / overlay.shape[0], 128 / overlay.shape[1]))
            overlay[lesion_mask.squeeze() == 1] = 255  # 将病灶区域标记为 255
            plt.figure(figsize=(5, 5))
            plt.imshow(overlay.squeeze().T, cmap="hot")
            right.pyplot(plt)

    else:
        st.write("请先上传医学图像文件")
    # -------------------- 下半部分：展示不同方向的切片 --------------------
    st.empty()
    st.empty()
    st.empty()

    st.subheader("方向切片展示")
    col21, col22, col23 = st.columns(3, border=True)
    with col21:
        st.subheader("Axial", text_alignment="center")
        slice_num1 = st.slider("Loc", 0, 154, 77, key="slider2")
        if uploaded_file:
            plt.imshow(img_data[:, :, slice_num1].T, cmap="gray")
            st.pyplot(plt)
    with col22:
        st.subheader("Coronal", text_alignment="center")
        slice_num2 = st.slider("Loc", 0, 239, 120, key="slider3")
        if uploaded_file:
            plt.imshow(img_data[ :,slice_num2, :].T, cmap="gray")
            st.pyplot(plt)
    with col23:
        st.subheader("Sagittal", text_alignment="center")
        slice_num3 = st.slider("Loc", 0, 239, 120, key="slider4")
        if uploaded_file:
            plt.imshow(img_data[:, :, slice_num3].T, cmap="gray")
            st.pyplot(plt)
