import os
import json

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim, ScheduledOptim_adv, EmoSphere, EmoSphere_disc
from model.multi_window_disc_concat_3discto2_lin import Discriminator


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


# get_discriminator, get_emosphere 내부에 use_dic 추가
def get_discriminator(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    disc_win_num = 3
    h = 128
    mel_disc = Discriminator(
        time_lengths=[32, 64, 96][:disc_win_num],
        freq_length=80,
        hidden_size=h,
        kernel=(3, 3),
    ).to(device)

    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        mel_disc.load_state_dict(ckpt["model_disc"])

    if train:
        scheduled_optim = ScheduledOptim_adv(
            mel_disc, train_config["optimizer"]["disc"], model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer_disc"])
        mel_disc.train()
        return mel_disc, scheduled_optim

    mel_disc.eval()
    mel_disc.requires_grad_ = False
    return mel_disc


def get_emosphere_adv(args, configs, device, train=False, use_discriminator=False):
    (preprocess_config, model_config, train_config) = configs

    model = EmoSphere_disc(
        preprocess_config, model_config, use_discriminator=use_discriminator
    ).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim_adv(
            model, train_config["optimizer"]["gen"], model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_emosphere(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = EmoSphere(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]

    if name == "MelGAN":
        speaker = config["vocoder"]["speaker"]
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        speaker = config["vocoder"]["speaker"]
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)
    elif name == "HIFIGAN_speechbrain":
        import sys
        sys.path.append("../")
        from speechbrain.inference.vocoders import HIFIGAN
        hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="pretrained_models/tts-hifigan-libritts-16kHz")
        vocoder = hifi_gan
    elif name == "BigVGAN":
        import sys
        sys.path.append("../")
        from speech_resynth.src.bigvgan.bigvgan import BigVGan
        vocoder = BigVGan.from_pretrained("ryota-komatsu/bigvgan")
        vocoder.to(device)
    
    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    # mel: torch.Size([8, 80, 173]), torch.float32
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1) # torch.Size([8, 44288]), torch.float32
        elif name == "HIFIGAN_speechbrain":
            wavs = vocoder.decode_batch(mels).squeeze(1) # torch.Size([8, 44288]), torch.float32
        elif name == "BigVGAN":
            mels = mels.transpose(1, 2)
            wavs = vocoder(mels) # torch.Size([8, 44288]), torch.float32

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
