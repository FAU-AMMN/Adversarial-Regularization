import glob
import matplotlib.pyplot as plt

import pandas as pd
import os
import re


def plot_PSNR_SSI(data, plot_folder, metric, approximator):

    fig, ax = plt.subplots()
    ax.plot(data[metric+"_recon_to_ground_truth"].to_numpy(), color="green", label="Recon "+metric)
    ax.plot(data[metric+"_noisy_to_ground_truth"].to_numpy(),color="red", label="Noisy "+metric)
    ax.set_xticks(range(0,len(data["Step"].to_numpy())))
    ax.set_xticklabels(data["Step"].to_numpy())
    ax.set_title(metric+" values (training on " +train_image_size+" images)")
    ax.set_ylabel("metric")
    ax.set_xlabel("Step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_folder+approximator+"Recon_quality_training_"+metric+".png")


def plot_training_loss(data, plot_folder, approximator):

    fig, ax = plt.subplots()
    ax.plot(data["Data_Difference"].to_numpy(), color="green", label="Data diff")
    ax.set_xticks(range(0,len(data["Step"].to_numpy())))
    ax.set_xticklabels(data["Step"].to_numpy())
    ax.set_title(" Data Difference (training on " +train_image_size+" images)")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_folder+approximator+"_Network_Opt_training_Data_Difference.png")


    fig, ax = plt.subplots()
    ax.plot(data["Lipschitz_Regulariser"].to_numpy(), color="green", label="Lipschitz")
    ax.set_xticks(range(0,len(data["Step"].to_numpy())))
    ax.set_xticklabels(data["Step"].to_numpy())
    ax.set_title(" Lipschitz Regularuzer (training on " +train_image_size+" images)")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_folder+approximator+"_Network_Opt_training_Lipschitz.png")


    fig, ax = plt.subplots()
    ax.plot(data["Overall_Net_Loss"].to_numpy(), color="green", label="Net Loss")
    ax.set_xticks(range(0,len(data["Step"].to_numpy())))
    ax.set_xticklabels(data["Step"].to_numpy())
    ax.set_title(" Net Loss (training on " +train_image_size+" images)")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_folder+approximator+"_Network_Opt_training_Net_loss.png")


logs_folder = "Saves/Denoising/ellipses/Adversarial_Regulariser/"
data_set_name = "ellipses"
train_image_size = "128x128"
prepend_name = ""
plots_save_root = "Plots/"
noise_level = 0.05

plots_save_folder = plots_save_root + data_set_name + "/" + train_image_size + "_" + str(noise_level).replace(".", "") + "/"

files_all_noise = glob.glob(logs_folder + "*/*/*/*.csv")

files = [x for x in files_all_noise if "Noise"+str(noise_level) in x]

if not os.path.exists(plots_save_folder):
   os.makedirs(plots_save_folder)

spectral_withResize_psnr = {}
Convo_NoStride_psnr = {}
CNN_to_FNO_psnr = {}

spectral_withResize_ssi = {}
Convo_NoStride_ssi = {}
CNN_to_FNO_ssi = {}


Convo_NoStride_RQ = None
Convo_NoStride_No = None

spectral_withResize_RQ = None
spectral_withResize_No = None

noisy_psnr = {}
noisy_ssi = {}
#noisy_val = {"PSNR": [], "SSI": []}
for f in files:
    split_f = re.split('/', f)
    if "Picture_Opt" in split_f[-2]:
        data = pd.read_csv(f)
        size_val = re.split('_', split_f[-2])[-1]
        if "ConvNetClassifier" in split_f[4]:
            if "CNN_to_FNO" in split_f[-2]:
                CNN_to_FNO_psnr[size_val] = data["PSNR_recon_to_ground_truth"].to_numpy()[-1]
                CNN_to_FNO_ssi[size_val] = data["SSI_recon_to_ground_truth"].to_numpy()[-1]
            else:
                Convo_NoStride_psnr[size_val] = data["PSNR_recon_to_ground_truth"].to_numpy()[-1]
                Convo_NoStride_ssi[size_val] = data["SSI_recon_to_ground_truth"].to_numpy()[-1]
                noisy_psnr[size_val] = data["PSNR_noisy_to_ground_truth"].to_numpy()[-1]
                noisy_ssi[size_val] = data["SSI_noisy_to_ground_truth"].to_numpy()[-1]
                #noisy_val["PSNR"] = noisy_val["PSNR"]+ [data["PSNR_noisy_to_ground_truth"].to_numpy()[-1]]
                #noisy_val["SSI"] = noisy_val["SSI"]+ [data["SSI_noisy_to_ground_truth"].to_numpy()[-1]]
        elif "Spectral_withResize" in split_f[4]:
            spectral_withResize_psnr[size_val] = data["PSNR_recon_to_ground_truth"].to_numpy()[-1]
            spectral_withResize_ssi[size_val] = data["SSI_recon_to_ground_truth"].to_numpy()[-1]
    
    elif "Network_Optimization" in split_f[-2]:
        if "ConvNetClassifier" in split_f[4]:
            if "reconstruction_quality" in split_f[-1]:
                Convo_NoStride_RQ = pd.read_csv(f)
            elif "network_optimization_loss" in split_f[-1]:
                Convo_NoStride_No = pd.read_csv(f)
        elif "spectral_withResize" in split_f[4]:
            if "reconstruction_quality" in split_f[-1]:
                spectral_withResize_RQ = pd.read_csv(f)
            elif "network_optimization_loss" in split_f[-1]:
                spectral_withResize_No = pd.read_csv(f)
                

spectral_withResize_psnr = pd.DataFrame.from_dict(spectral_withResize_psnr, orient="index", columns=["Value"])
spectral_withResize_psnr = spectral_withResize_psnr.reindex(sorted(spectral_withResize_psnr.index, key=lambda x: x.zfill(3)))

Convo_NoStride_psnr = pd.DataFrame.from_dict(Convo_NoStride_psnr, orient="index", columns=["Value"])
Convo_NoStride_psnr = Convo_NoStride_psnr.reindex(sorted(Convo_NoStride_psnr.index, key=lambda x: x.zfill(3)))

CNN_to_FNO_psnr = pd.DataFrame.from_dict(CNN_to_FNO_psnr, orient="index", columns=["Value"])
CNN_to_FNO_psnr = CNN_to_FNO_psnr.reindex(sorted(CNN_to_FNO_psnr.index, key=lambda x: x.zfill(3)))

spectral_withResize_ssi = pd.DataFrame.from_dict(spectral_withResize_ssi, orient="index", columns=["Value"])
spectral_withResize_ssi = spectral_withResize_ssi.reindex(sorted(spectral_withResize_ssi.index, key=lambda x: x.zfill(3)))

Convo_NoStride_ssi = pd.DataFrame.from_dict(Convo_NoStride_ssi, orient="index", columns=["Value"])
Convo_NoStride_ssi = Convo_NoStride_ssi.reindex(sorted(Convo_NoStride_ssi.index, key=lambda x: x.zfill(3)))

CNN_to_FNO_ssi = pd.DataFrame.from_dict(CNN_to_FNO_ssi, orient="index", columns=["Value"])
CNN_to_FNO_ssi = CNN_to_FNO_ssi.reindex(sorted(CNN_to_FNO_ssi.index, key=lambda x: x.zfill(3)))

noisy_psnr = pd.DataFrame.from_dict(noisy_psnr, orient="index", columns=["Value"])
noisy_psnr = noisy_psnr.reindex(sorted(noisy_psnr.index, key=lambda x: x.zfill(3)))

noisy_ssi = pd.DataFrame.from_dict(noisy_ssi, orient="index", columns=["Value"])
noisy_ssi = noisy_ssi.reindex(sorted(noisy_ssi.index, key=lambda x: x.zfill(3)))

fig, ax = plt.subplots()
ax.plot(CNN_to_FNO_psnr["Value"].to_numpy(), color="green", label="CNN to FNO")
ax.plot(spectral_withResize_psnr["Value"].to_numpy(), color="blue", label="Trained FNO")
ax.plot(Convo_NoStride_psnr["Value"].to_numpy(), color="orange", label="Trained CNN")
ax.plot(noisy_psnr["Value"].to_numpy(), color="red", label="Noisy PSNR")
#check = CNN_to_FNO_psnr.index
ax.set_xticks(range(0,len(Convo_NoStride_psnr.index)))
ax.set_xticklabels(Convo_NoStride_psnr.index)
ax.set_title("PSNR values (trained on " +train_image_size+" images)")
ax.set_ylabel("PSNR")
ax.set_xlabel("Image Dimension")


psnr_csv_dict = {"image dim":Convo_NoStride_psnr.index,
                 "CNN to FNO": CNN_to_FNO_psnr["Value"].to_numpy(),
                 "Trained FNO": spectral_withResize_psnr["Value"].to_numpy(),
                 "Trained CNN": Convo_NoStride_psnr["Value"].to_numpy(),
                 "Noisy PSNR": noisy_psnr["Value"].to_numpy()}


psnr_csv_df = pd.DataFrame.from_dict(data=psnr_csv_dict)
psnr_csv_df.to_csv(plots_save_folder+data_set_name+"_Picture_opt_PSNR_"+prepend_name+".csv", sep=",")

plt.legend()
plt.tight_layout()
plt.savefig(plots_save_folder+data_set_name+"_Picture_opt_PSNR_"+prepend_name+".png")


fig, ax = plt.subplots()
ax.plot(CNN_to_FNO_ssi["Value"].to_numpy(), color="green", label="CNN to FNO")
ax.plot(spectral_withResize_ssi["Value"].to_numpy(), color="blue", label="Trained FNO")
ax.plot(Convo_NoStride_ssi["Value"].to_numpy(), color="orange", label="Trained CNN")
ax.plot(noisy_ssi["Value"].to_numpy(), color="red", label="Noisy SSI")
ax.set_xticks(range(0,len(Convo_NoStride_ssi.index)))
ax.set_xticklabels(Convo_NoStride_ssi.index)
ax.set_title("SSI values (trained on " +train_image_size+" images)")
ax.set_ylabel("SSI")
ax.set_xlabel("Image Dimension")



ssi_csv_dict = {"image dim":Convo_NoStride_ssi.index,
                "CNN to FNO": CNN_to_FNO_ssi["Value"].to_numpy(),
                 "Trained FNO": spectral_withResize_ssi["Value"].to_numpy(),
                 "Trained CNN": Convo_NoStride_ssi["Value"].to_numpy(),
                 "Noisy SSI": noisy_ssi["Value"].to_numpy()}


ssi_csv_df = pd.DataFrame.from_dict(data=ssi_csv_dict)
ssi_csv_df.to_csv(plots_save_folder+data_set_name+"_Picture_opt_SSI_"+prepend_name+".csv", sep=",")



plt.legend()
plt.tight_layout()
plt.savefig(plots_save_folder+data_set_name+"_Picture_opt_SSI_"+prepend_name+".png")


if type(Convo_NoStride_No) != type(None):

    plot_PSNR_SSI(data=Convo_NoStride_RQ, plot_folder=plots_save_folder, metric="PSNR", approximator = "Convo_NoStride")
    plot_PSNR_SSI(data=Convo_NoStride_RQ, plot_folder=plots_save_folder, metric="SSI", approximator = "Convo_NoStride")

    plot_training_loss(data=Convo_NoStride_No, plot_folder=plots_save_folder, approximator = "Convo_NoStride")


if type(spectral_withResize_No) != type(None):

    plot_PSNR_SSI(data=spectral_withResize_RQ, plot_folder=plots_save_folder, metric="PSNR", approximator = "spectral_withResize")
    plot_PSNR_SSI(data=spectral_withResize_RQ, plot_folder=plots_save_folder, metric="SSI", approximator = "spectral_withResize")

    plot_training_loss(data=spectral_withResize_No, plot_folder=plots_save_folder, approximator = "spectral_withResize")