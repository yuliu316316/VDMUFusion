import time
import json
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import random
from dataset import MFI_Dataset
from Diffusion import GaussianDiffusion
from Condition_Noise_Predictor.UNet import NoisePred
from utils import tensorboard_writer, logger, save_model

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def train(config_path):
    timestr = time.strftime('%Y%m%d_%H%M%S')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # ********************* load train dataset *************************
    train_datasePath = config["dataset"]["train"]["path"]
    train_phase = config["dataset"]["train"]["phase"]
    train_batch_size = config["dataset"]["train"]["batch_size"]
    train_use_dataTransform = config["dataset"]["train"]["use_dataTransform"]
    train_resize = config["dataset"]["train"]["resize"]
    train_imgSize = config["dataset"]["train"]["imgSize"]
    train_shuffle = config["dataset"]["train"]["shuffle"]
    train_drop_last = config["dataset"]["train"]["drop_last"]

    # Dataset of various fusion tasks
    # ************** Medical ****************
    medical_dataset_path = "Dataset/Medical/Train/PET_MR"
    train_medical_dataset = MFI_Dataset(medical_dataset_path, phase=train_phase, use_dataTransform=train_use_dataTransform,
                                resize=train_resize, imgSzie=train_imgSize, fusion_type="medical")
    train_medical_dataloader = DataLoader(train_medical_dataset, batch_size=train_batch_size, shuffle=train_shuffle,
                                  drop_last=train_drop_last)

    # ************** Multi-focus ****************
    multi_focus_dataset_path = "Dataset/Multi-Focus-Images/train/NYU-D-100"
    train_multi_focus_dataset = MFI_Dataset(multi_focus_dataset_path, phase=train_phase,
                                        use_dataTransform=train_use_dataTransform,
                                        resize=train_resize, imgSzie=train_imgSize, fusion_type="multi-focus")
    train_multi_focus_dataloader = DataLoader(train_multi_focus_dataset, batch_size=train_batch_size, shuffle=train_shuffle,
                                          drop_last=train_drop_last)

    # ************** Multi-exposure ****************
    MEF_dataset_path = "Dataset/MEF_data/train"
    train_MEF_dataset = MFI_Dataset(MEF_dataset_path, phase=train_phase,
                                            use_dataTransform=train_use_dataTransform,
                                            resize=train_resize, imgSzie=train_imgSize, fusion_type="multi-exposure")
    train_MEF_dataloader = DataLoader(train_MEF_dataset, batch_size=train_batch_size, shuffle=train_shuffle,
                                              drop_last=train_drop_last)

    # ************** VIF-exposure ****************
    VIF_dataset_path = "Dataset/VIF_data/M3FD_Fusion/train"
    train_VIF_dataset = MFI_Dataset(VIF_dataset_path, phase=train_phase,
                                    use_dataTransform=train_use_dataTransform,
                                    resize=train_resize, imgSzie=train_imgSize, fusion_type="vis-inf")
    train_VIF_dataloader = DataLoader(train_VIF_dataset, batch_size=train_batch_size, shuffle=train_shuffle,
                                      drop_last=train_drop_last)


    # Condition Noise Predictor
    in_channels = config["Condition_Noise_Predictor"]["UNet"]["in_channels"]
    out_channels = config["Condition_Noise_Predictor"]["UNet"]["out_channels"]
    model_channels = config["Condition_Noise_Predictor"]["UNet"]["model_channels"]
    num_res_blocks = config["Condition_Noise_Predictor"]["UNet"]["num_res_blocks"]
    dropout = config["Condition_Noise_Predictor"]["UNet"]["dropout"]
    time_embed_dim_mult = config["Condition_Noise_Predictor"]["UNet"]["time_embed_dim_mult"]
    down_sample_mult = config["Condition_Noise_Predictor"]["UNet"]["down_sample_mult"]
    model = NoisePred(in_channels, out_channels, model_channels, num_res_blocks, dropout, time_embed_dim_mult,
                      down_sample_mult)

    # whether to use the pre-training model
    use_preTrain_model = config["Condition_Noise_Predictor"]["use_preTrain_model"]
    if use_preTrain_model:
        preTrain_Model_path = config["Condition_Noise_Predictor"]["preTrain_Model_path"]
        model.load_state_dict(torch.load(preTrain_Model_path, map_location=device))
        print(f"using pre-trained model：{preTrain_Model_path}")
    model = model.to(device)

    # optimizer
    init_lr = config["optimizer"]["init_lr"]
    use_lr_scheduler = config["optimizer"]["use_lr_scheduler"]
    StepLR_size = config["optimizer"]["StepLR_size"]
    StepLR_gamma = config["optimizer"]["StepLR_gamma"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)
    if use_lr_scheduler:
        learningRate_scheduler = lr_scheduler.StepLR(optimizer, step_size=StepLR_size, gamma=StepLR_gamma)

    # diffusion model
    T = config["diffusion_model"]["T"]
    beta_schedule_type = config["diffusion_model"]["beta_schedule_type"]
    loss_scale = config["diffusion_model"]["loss_scale"]
    diffusion = GaussianDiffusion(T, beta_schedule_type)

    # log
    writer = tensorboard_writer(timestr)
    log = logger(timestr)
    print(f"time: {timestr}")
    log.write(f"time: {timestr} \n")
    # print(f"using {len(train_dataset)} images for train")
    # log.write(f"using {len(train_dataset)} images for train  \n\n")
    log.write(f"config:  \n")
    log.write(json.dumps(config, ensure_ascii=False, indent=4))
    if use_lr_scheduler:
        log.write(
            f"\n learningRate_scheduler = lr_scheduler.StepLR(optimizer, step_size={StepLR_size}, gamma={StepLR_gamma})  \n\n")

    # hyper-parameter
    epochs = config["hyperParameter"]["epochs"]
    start_epoch = config["hyperParameter"]["start_epoch"]
    loss_step = config["hyperParameter"]["loss_step"]
    save_model_epoch_step = config["hyperParameter"]["save_model_epoch_step"]
    num_train_step = 0

    fusion_task = ["medical", "multi-exposure", "vis-inf", "multi-focus"]

    for epoch in range(start_epoch, epochs):
        # train
        model.train()
        loss_sum = 0
        writer.add_scalar('lr_epoch: ', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        fusion_type = random.choice(fusion_task)
        print("This epoch conducts the fusion task:", fusion_type)
        if fusion_type == "medical":
            dataloader = train_medical_dataloader
        elif fusion_type == "multi-focus":
            dataloader =  train_multi_focus_dataloader
        elif fusion_type == "multi-exposure":
            dataloader = train_MEF_dataloader
        else:
            dataloader = train_VIF_dataloader

        train_step_sum = len(dataloader)
        log.write(f"This epoch conducts the {fusion_type} images fusion task  \n\n")

        for train_step, train_images in tqdm(enumerate(dataloader), desc="train step"):
            optimizer.zero_grad()
            train_sourceImg1 = train_images[0].to(device)
            train_sourceImg2 = train_images[1].to(device)

            t = torch.randint(0, T, (train_batch_size,), device=device).long()
            noise_loss, fusion_loss = diffusion.train_losses(model, train_sourceImg1, train_sourceImg2, t, loss_scale)
            loss_total = noise_loss + 10 * fusion_loss
            writer.add_scalar('loss_step: ', noise_loss, num_train_step)

            if train_step % loss_step == 0:
                print(
                    f" [epoch] {epoch}/{epochs}    "
                    f"[epoch_step] {train_step}/{train_step_sum}     "
                    f"[train_step] {num_train_step}     "
                    f"[noise_loss] {noise_loss.item() :.6f}     "
                    f"[fusion_loss] {10 * fusion_loss.item() :.6f}     "
                    # f"[loss_total] {loss_total.item() :.6f}     "
                    f"[lr] {optimizer.state_dict()['param_groups'][0]['lr'] :.6f}     "
                    f"[t] {t.cpu().numpy()}")

                log.write(f" [epoch] {epoch}/{epochs}    "
                          f"[epoch_step] {train_step}/{train_step_sum}     "
                          f"[train_step] {num_train_step}     "
                          f"[noise_loss] {noise_loss.item() :.6f}     "
                          f"[fusion_loss] {10 *fusion_loss.item() :.6f}     "
                          f"[lr] {optimizer.state_dict()['param_groups'][0]['lr'] :.6f}     "
                          f"[t] {t.cpu().numpy()}"
                          f"\n")

            loss_total.backward()
            optimizer.step()

            loss_sum += noise_loss
            num_train_step += 1

        aver_loss = loss_sum / train_step_sum
        print('aver_loss', aver_loss)

        if epoch % save_model_epoch_step == 0:
            save_model(model, epoch, timestr)
        if epoch == epochs - 1:
            save_model(model, epoch, timestr)

        # update learning rate
        if use_lr_scheduler:
            learningRate_scheduler.step()
        writer.add_scalar('aver_loss_epoch: ', aver_loss, epoch)
        log.write("\n")

    print("End of training")
    log.write("End of training \n")
    writer.close()


if __name__ == '__main__':
    config_path = "config.json"
    train(config_path)