from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Bicubic
bicubic = Path("outputs/bicubic/bicubic.csv")
bicubic01 = Path("outputs/bicubic/bicubic_noise_std=0.1.csv")
bicubic02 = Path("outputs/bicubic/bicubic_noise_std=0.2.csv")
bicubic03 = Path("outputs/bicubic/bicubic_noise_std=0.3.csv")
bicubic04 = Path("outputs/bicubic/bicubic_noise_std=0.4.csv")
bicubic05 = Path("outputs/bicubic/bicubic_noise_std=0.5.csv")
bicubic16x = Path("outputs/bicubic/bicubic_x16.csv")
# DIP
dip = Path("outputs/DIP/dip.csv")
dip_noise_01 = Path("outputs/DIP/dip_noisy_std=0.1.csv")
dip_noise_02 = Path("outputs/DIP/dip_noisy_std=0.2.csv")
dip_noise_03 = Path("outputs/DIP/dip_noisy_std=0.3.csv")
dip_noise_04 = Path("outputs/DIP/dip_noisy_std=0.4.csv")
dip_noise_05 = Path("outputs/DIP/dip_noisy_std=0.5.csv")
dip_x16 = Path("outputs/DIP/x16_dip.csv")
# INR
inr = Path("outputs/INR/inr.csv")
inr_noise_01 = Path("outputs/INR/inr_noise_std=0.1.csv")
inr_noise_02 = Path("outputs/INR/inr_noise_std=0.2.csv")
inr_noise_03 = Path("outputs/INR/inr_noise_std=0.3.csv")
inr_noise_04 = Path("outputs/INR/inr_noise_std=0.4.csv")
inr_noise_05 = Path("outputs/INR/inr_noise_std=0.5.csv")
inr_x16 = Path("outputs/INR/x16_inr.csv")

dip_noise_01 = pd.read_csv(dip_noise_01)
dip_noise_01.drop(columns=["sample", "num_patches", "baseline_psnr", "baseline_ssim", "baseline_lpips"], inplace=True)
inr_noise_01 = pd.read_csv(inr_noise_01)
inr_noise_01.drop(columns=["image", "loss", "baseline_psnr", "baseline_ssim", "baseline_lpips"], inplace=True)
inr_noise_01.rename(columns={"psnr": "inr_psnr", "ssim": "inr_ssim", "lpips": "inr_lpips"}, inplace=True)
bicubic_noise_01 = pd.read_csv(bicubic01)
bicubic_noise_01.drop(columns=["image_idx"], inplace=True)
bicubic_noise_01.rename(columns={"psnr": "bicubic_psnr", "ssim": "bicubic_ssim", "lpips": "bicubic_lpips"}, inplace=True)
noise_01 = dip_noise_01.join(inr_noise_01).join(bicubic_noise_01)

dip_noise_02 = pd.read_csv(dip_noise_02)
dip_noise_02.drop(columns=["sample", "num_patches", "baseline_psnr", "baseline_ssim", "baseline_lpips"], inplace=True)
inr_noise_02 = pd.read_csv(inr_noise_02)
inr_noise_02.drop(columns=["image", "loss", "baseline_psnr", "baseline_ssim", "baseline_lpips"], inplace=True)
inr_noise_02.rename(columns={"psnr": "inr_psnr", "ssim": "inr_ssim", "lpips": "inr_lpips"}, inplace=True)
bicubic_noise_02 = pd.read_csv(bicubic02)
bicubic_noise_02.drop(columns=["image_idx"], inplace=True)
bicubic_noise_02.rename(columns={"psnr": "bicubic_psnr", "ssim": "bicubic_ssim", "lpips": "bicubic_lpips"}, inplace=True)
noise_02 = dip_noise_02.join(inr_noise_02).join(bicubic_noise_02)

dip_noise_03 = pd.read_csv(dip_noise_03)
dip_noise_03.drop(columns=["sample", "num_patches", "baseline_psnr", "baseline_ssim", "baseline_lpips"], inplace=True)
inr_noise_03 = pd.read_csv(inr_noise_03)
inr_noise_03.drop(columns=["image", "loss", "baseline_psnr", "baseline_ssim", "baseline_lpips"], inplace=True)
inr_noise_03.rename(columns={"psnr": "inr_psnr", "ssim": "inr_ssim", "lpips": "inr_lpips"}, inplace=True)
bicubic_noise_03 = pd.read_csv(bicubic03)
bicubic_noise_03.drop(columns=["image_idx"], inplace=True)
bicubic_noise_03.rename(columns={"psnr": "bicubic_psnr", "ssim": "bicubic_ssim", "lpips": "bicubic_lpips"}, inplace=True)
noise_03 = dip_noise_03.join(inr_noise_03).join(bicubic_noise_03)

dip_noise_04 = pd.read_csv(dip_noise_04)
dip_noise_04.drop(columns=["sample", "num_patches", "baseline_psnr", "baseline_ssim", "baseline_lpips"], inplace=True)
inr_noise_04 = pd.read_csv(inr_noise_04)
inr_noise_04.drop(columns=["image", "loss", "baseline_psnr", "baseline_ssim", "baseline_lpips"], inplace=True)
inr_noise_04.rename(columns={"psnr": "inr_psnr", "ssim": "inr_ssim", "lpips": "inr_lpips"}, inplace=True)
bicubic_noise_04 = pd.read_csv(bicubic04)
bicubic_noise_04.drop(columns=["image_idx"], inplace=True)
bicubic_noise_04.rename(columns={"psnr": "bicubic_psnr", "ssim": "bicubic_ssim", "lpips": "bicubic_lpips"}, inplace=True)
noise_04 = dip_noise_04.join(inr_noise_04).join(bicubic_noise_04)

dip_noise_05 = pd.read_csv(dip_noise_05)
dip_noise_05.drop(columns=["sample", "num_patches", "baseline_psnr", "baseline_ssim", "baseline_lpips"], inplace=True)
inr_noise_05 = pd.read_csv(inr_noise_05)
inr_noise_05.drop(columns=["image", "loss", "baseline_psnr", "baseline_ssim", "baseline_lpips"], inplace=True)
inr_noise_05.rename(columns={"psnr": "inr_psnr", "ssim": "inr_ssim", "lpips": "inr_lpips"}, inplace=True)
bicubic_noise_05 = pd.read_csv(bicubic05)
bicubic_noise_05.drop(columns=["image_idx"], inplace=True)
bicubic_noise_05.rename(columns={"psnr": "bicubic_psnr", "ssim": "bicubic_ssim", "lpips": "bicubic_lpips"}, inplace=True)
noise_05 = dip_noise_05.join(inr_noise_05).join(bicubic_noise_05)

dip_x16_pd = pd.read_csv(dip_x16)
dip_x16_pd.drop(columns=["sample", "num_patches", "baseline_psnr", "baseline_ssim", "baseline_lpips"], inplace=True)
inr_x16_pd = pd.read_csv(inr_x16)
inr_x16_pd.drop(columns=["image", "loss", "baseline_psnr", "baseline_ssim", "baseline_lpips"], inplace=True)
inr_x16_pd.rename(columns={"psnr": "inr_psnr", "ssim": "inr_ssim", "lpips": "inr_lpips"}, inplace=True)
bicubic_x16_pd = pd.read_csv(bicubic16x)
bicubic_x16_pd.drop(columns=["image"], inplace=True)
bicubic_x16_pd.rename(columns={"baseline_psnr": "bicubic_psnr", "baseline_ssim": "bicubic_ssim", "baseline_lpips": "bicubic_lpips"}, inplace=True)
x16 = dip_x16_pd.join(inr_x16_pd).join(bicubic_x16_pd)

dip_pd = pd.read_csv(dip)
dip_pd.drop(columns=["sample", "num_patches", "baseline_psnr", "baseline_ssim", "baseline_lpips"], inplace=True)
inr_pd = pd.read_csv(inr)
inr_pd.drop(columns=["image", "loss", "baseline_psnr", "baseline_ssim", "baseline_lpips"], inplace=True)
inr_pd.rename(columns={"psnr": "inr_psnr", "ssim": "inr_ssim", "lpips": "inr_lpips"}, inplace=True)
bicubic_pd = pd.read_csv(bicubic)
bicubic_pd.drop(columns=["image"], inplace=True)
bicubic_pd.rename(columns={"baseline_psnr": "bicubic_psnr", "baseline_ssim": "bicubic_ssim", "baseline_lpips": "bicubic_lpips"}, inplace=True)
regular = dip_pd.join(inr_pd).join(bicubic_pd)

labels_map = {
    'bicubic_psnr': 'Bicubic', 'dip_psnr': 'DIP', 'inr_psnr': 'INR',
    'bicubic_ssim': 'Bicubic', 'dip_ssim': 'DIP', 'inr_ssim': 'INR',
    'bicubic_lpips': 'Bicubic', 'dip_lpips': 'DIP', 'inr_lpips': 'INR'
}

sns.set_style(style="whitegrid")

def create_plot(data, title_suffix, filename):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    data_psnr = data[['bicubic_psnr', 'dip_psnr', 'inr_psnr']].rename(columns=labels_map)
    data_ssim = data[['bicubic_ssim', 'dip_ssim', 'inr_ssim']].rename(columns=labels_map)
    data_lpips = data[['bicubic_lpips', 'dip_lpips', 'inr_lpips']].rename(columns=labels_map)
    
    sns.boxplot(data=data_psnr, ax=axes[0], orient='h', width=0.5)
    axes[0].set_title(f"{title_suffix} PSNR (dB) (Higher is better)")
    
    sns.boxplot(data=data_ssim, ax=axes[1], orient='h', width=0.5)
    axes[1].set_title(f"{title_suffix} SSIM (Higher is better)")
    
    sns.boxplot(data=data_lpips, ax=axes[2], orient='h', width=0.5)
    axes[2].set_title(f"{title_suffix} LPIPS (Lower is better)")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

create_plot(noise_01, "AWGN σ=0.1: Bicubic vs DIP vs INR", "outputs/plots/comparison_noise_01.png")
create_plot(noise_02, "AWGN σ=0.2: Bicubic vs DIP vs INR", "outputs/plots/comparison_noise_02.png")
create_plot(noise_03, "AWGN σ=0.3: Bicubic vs DIP vs INR", "outputs/plots/comparison_noise_03.png")
create_plot(noise_04, "AWGN σ=0.4: Bicubic vs DIP vs INR", "outputs/plots/comparison_noise_04.png")
create_plot(noise_05, "AWGN σ=0.5: Bicubic vs DIP vs INR", "outputs/plots/comparison_noise_05.png")
create_plot(x16, "x16 Super-Resolution: Bicubic vs DIP vs INR", "outputs/plots/comparison_x16.png")
create_plot(regular, "Super-Resolution: Bicubic vs DIP vs INR", "outputs/plots/comparison_regular.png")