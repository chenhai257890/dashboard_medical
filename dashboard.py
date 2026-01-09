import streamlit as st
import numpy as np
import pandas as pd
import os
import random
import monai.transforms as transforms
from os.path import join
from pathlib import Path
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset, DataLoader
import torch as th
import ml_collections
import unet
import blobfile as bf
from test_model import iter_mask_refinement
import nibabel as nib
import matplotlib.pyplot as plt
import io
# ---------------------------- 函数功能区 -----------------------------
# 1、主函数
def normalize(img, _min=None, _max=None):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img
def get_default_configs():
    config = ml_collections.ConfigDict()
    config.device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    config.seed = 1
    config.data = data = ml_collections.ConfigDict()
    data.path = "/home/some5338/Documents/data/brats"
    data.sequence_translation = False
    data.healthy_data_percentage = None

    ## model config
    config.model = model = ml_collections.ConfigDict()
    model.image_size = 128
    model.num_input_channels = 1 
    model.num_channels = 32
    model.num_res_blocks = 2
    model.num_heads = 1
    model.num_heads_upsample = -1
    model.num_head_channels = -1
    model.attention_resolutions = "32,16,8"

    attention_ds = []
    if model.attention_resolutions != "":
        for res in model.attention_resolutions.split(","):
            attention_ds.append(model.image_size // int(res))
    model.attention_ds = attention_ds

    model.channel_mult = {64:(1, 2, 3, 4), 128:(1, 1, 2, 3, 4)}[model.image_size]
    model.dropout = 0.1
    model.use_checkpoint = False
    model.use_scale_shift_norm = True
    model.resblock_updown = True
    model.use_fp16 = False
    model.use_new_attention_order = False
    model.dims = 2

    # score model training
    config.model.training = training_model = ml_collections.ConfigDict()
    training_model.lr = 1e-4
    training_model.weight_decay = 0.00
    training_model.lr_decay_steps = 150000
    training_model.lr_decay_factor = 0.1
    training_model.batch_size = 32
    training_model.ema_rate = "0.9999"  # comma-separated list of EMA values
    training_model.log_interval = 100
    training_model.save_interval = 5000
    training_model.use_fp16 = model.use_fp16
    training_model.fp16_scale_growth = 1e-3
    training_model.iterations = 150000

    config.testing = testing = ml_collections.ConfigDict()
    testing.batch_size = 32
    testing.task = 'inpainting'

    return config
# datasets

def get_brats2021_train_transform_abnormalty(image_size):
    base_transform = [
        transforms.EnsureChannelFirstd(
            keys=['input', 'brainmask', 'seg', 'gauss_mask'], channel_dim='no_channel'),
        transforms.Resized(
            keys=['input', 'brainmask', 'seg', 'gauss_mask'],
            spatial_size=(image_size, image_size)),
    ]
    return transforms.Compose(base_transform)


class BraTS2021Dataset(Dataset):
    def __init__(self, images, transforms=None):
        super(BraTS2021Dataset, self).__init__()
        self.images = images
        self.transforms = transforms

    def __getitem__(self, index: int) -> tuple:
        image = self.images[index]
        brain_mask = (image > 0).astype(np.uint8)
        item = self.transforms(
            {'input': input, 'brainmask': brain_mask})

        return item

    def __len__(self):
        return len(self.images)


# dataloader
def seed_worker(worker_id):
    np.random.seed(worker_id)
    random.seed(0)

g = th.Generator()
g.manual_seed(0)
def get_data_loader_brats(mod, path, batch_size, image_size, split_set: str = 'train'):

    assert split_set in ["train", "val", "test"]
    default_kwargs = {"drop_last": True, "batch_size": batch_size, "pin_memory": False, "num_workers": 0,
                    "worker_init_fn": seed_worker, "generator": g, }
    if split_set == "test":
        patient_dir = os.path.join(path, 'test')
        default_kwargs["shuffle"] = False
    elif split_set == "val":
        patient_dir = os.path.join(path, 'val')
        default_kwargs["shuffle"] = False
    else:
        patient_dir = os.path.join(path, 'train')
        default_kwargs["shuffle"] = True
        default_kwargs["num_workers"] = 0
    transforms = get_brats2021_train_transform_abnormalty(image_size)
    dataset = BraTS2021Dataset(
        data_root=patient_dir,
        mode='train',
        input_modality=mod,
        transforms=transforms)

    #print(f"dataset lenght: {len(dataset)}")
    return th.utils.data.DataLoader(dataset, **default_kwargs)
# Model
def create_model(config: ml_collections.ConfigDict, image_level_cond):
    return unet.UNetModel(
        in_channels=config.model.num_input_channels,
        model_channels=config.model.num_channels,
        out_channels=config.model.num_input_channels,
        num_res_blocks=config.model.num_res_blocks,
        attention_resolutions=tuple(config.model.attention_ds),
        dropout=config.model.dropout,
        channel_mult=config.model.channel_mult,
        dims=config.model.dims,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=config.model.num_heads,
        num_head_channels=config.model.num_head_channels,
        num_heads_upsample=config.model.num_heads_upsample,
        use_scale_shift_norm=config.model.use_scale_shift_norm,
        resblock_updown=config.model.resblock_updown,
        image_level_cond=image_level_cond,
    )
def main(images):
    use_gpus = 0
    config = get_default_configs()
    experiment_name_first_iter = "first_iter_brats_t1"
    experiment_name_masked_autoencoder = "masked_autoencoder_brats_t1"
    input_mod = 't1'
    default_kwargs = {"drop_last": True, "batch_size": 1, "pin_memory": False, "num_workers": 0, "shuffle": False,
                    "worker_init_fn": seed_worker, "generator": g, }
    transforms = get_brats2021_train_transform_abnormalty(config.model.image_size)
    dataset = BraTS2021Dataset(
        images=images,
        transforms=transforms)
    test_loader = th.utils.data.DataLoader(dataset, **default_kwargs)
    model_first_iter = create_model(config, image_level_cond=False)
    model_masked_autoencoder = create_model(config, image_level_cond=True)

    filename = "model090000.pt"
    path_first_iter = bf.join('./model_save', experiment_name_first_iter, filename)
    path_masked_autoencoder = bf.join('./model_save', experiment_name_masked_autoencoder, filename)
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
    brain_mask_all = np.zeros((len(test_loader.dataset), config.model.num_input_channels, config.model.image_size, config.model.image_size))


    num_iter = 0
    for test_data_dict in enumerate(test_loader):
        test_data_input = test_data_dict[1].pop('input').cpu()
        brain_mask = test_data_dict[1].pop('brainmask').cpu()
        
        final_mask, final_reconstruction = iter_mask_refinement(
                model_masked_autoencoder, model_first_iter, test_data_input,
                brain_mask, experiment_name_masked_autoencoder
            )
        img_pred_mask_all[num_sample:num_sample + test_data_input.shape[0]] = final_mask.cpu().numpy()
        num_sample += test_data_input.shape[0]

    return final_mask






st.set_page_config(page_title="Brain Lesion Detection Dashboard", layout="wide")

# ---------------------------- UI -----------------------------
st.title("🧠 Brain MRI Lesion Detection Dashboard")
# 创建两列布局：左边为上传区域，右边为展示区域
col1, col2 = st.columns([1, 3])

# ---------------------------- 左侧区域 ----------------------------
with col1:
    # 文件上传控件
    uploaded_file = st.file_uploader("上传 .nii.gz 文件", type=["nii.gz"])
    slider1 = st.slider

    # 切片位置滑块
    if uploaded_file:
        # 读取 NIfTI 文件
        file_path = os.path.join("./data", uploaded_file.name)
        nii_image = nib.load(file_path)
        img_data = nii_image.get_fdata()  # 获取图像数据

        # 获取图像的维度
        depth, height, width = img_data.shape
        st.write(f"图像维度: {depth} x {height} x {width}")

        # 切片位置滑块（控制当前选择的切片）
        direction = st.radio("选择切片方向", ("横截面", "纵截面", "冠状面"))
        
        if direction == "横截面":
            slice_num = st.slider("选择横截面位置", 0, depth - 1, depth // 2)
            slice_data = img_data[slice_num, :, :]
        elif direction == "纵截面":
            slice_num = st.slider("选择纵截面位置", 0, height - 1, height // 2)
            slice_data = img_data[:, slice_num, :]
        else:  # 冠状面
            slice_num = st.slider("选择冠状面位置", 0, width - 1, width // 2)
            slice_data = img_data[:, :, slice_num]

        # 显示切片信息
        st.write(f"当前 {direction} 切片位置: {slice_num}")
        plt.imshow(slice_data.T, cmap="gray")  # 转置显示
        st.pyplot(plt)

# ---------------------------- 右侧区域 ----------------------------
with col2:
    # -------------------- 上半部分：展示病灶掩膜 --------------------
    st.subheader("病灶检测")
    
    # 病灶检测按钮
    if uploaded_file:
        if st.button("生成病灶掩膜"):
            # 假设你有一个封装好的病灶检测函数 `detect_lesion` 
            lesion_mask = main(slice_data)  # 你需要提供该函数

            # 显示病灶掩膜
            st.subheader("病灶检测结果")
            plt.imshow(lesion_mask.T, cmap="hot")  # 热力图表示病灶掩膜
            st.pyplot(plt)
            
            # 显示原图与病灶掩膜叠加图
            st.subheader("原图与病灶掩膜叠加")
            overlay = np.copy(slice_data)
            overlay[lesion_mask == 1] = 255  # 将病灶区域标记为 255
            plt.imshow(overlay.T, cmap="hot")
            st.pyplot(plt)

    else:
        st.write("请先上传医学图像文件")

    # -------------------- 下半部分：展示不同方向的切片 --------------------
    if uploaded_file:
        st.subheader(f"显示 {direction} 切片位置: {slice_num} 的三个方向切片")

        # 显示由滑块控制的切片位置的三个方向图
        if direction == "横截面":
            coronal_slice = img_data[:, slice_num, :]
            sagittal_slice = img_data[slice_num, :, :]
        elif direction == "纵截面":
            coronal_slice = img_data[:, slice_num, :]
            sagittal_slice = img_data[slice_num, :, :]
        else:  # 冠状面
            coronal_slice = img_data[:, :, slice_num]
            sagittal_slice = img_data[slice_num, :, :]

        # 显示其他方向的切片
        st.subheader("横截面")
        plt.imshow(sagittal_slice.T, cmap="gray")
        st.pyplot(plt)
        
        st.subheader("纵截面")
        plt.imshow(coronal_slice.T, cmap="gray")
        st.pyplot(plt)
        
        st.subheader("冠状面")
        plt.imshow(img_data[:, :, slice_num].T, cmap="gray")
        st.pyplot(plt)






