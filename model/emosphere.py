import os
import json

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths

def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


class EmoSphere(nn.Module):
    """FastSpeech2"""

    def __init__(self, preprocess_config, model_config, use_spk_lookup=True, use_emo_lookup=True):
        super(EmoSphere, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

        self.emotion_emb = None
        if model_config["multi_emotion"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "emotions.json"
                ),
                "r",
            ) as f:
                json_raw = json.load(f)
                n_emotion = len(json_raw["emotion_dict"])
            self.emotion_emb = nn.Embedding(
                n_emotion,
                model_config["transformer"]["encoder_hidden"],
            )
        # Sphere
        self.hidden_size = model_config["transformer"]["encoder_hidden"]

        self.use_spk_lookup = use_spk_lookup
        if self.use_spk_lookup:
            self.spk_id_proj = Embedding(n_speaker, self.hidden_size // 2)

        self.use_emo_lookup = use_emo_lookup
        if self.use_emo_lookup:
            self.emo_id_proj = Embedding(n_emotion, self.hidden_size // 4)

        self.emo_VAD_inten_proj = nn.Linear(1, self.hidden_size // 2, bias=True)
        self.emosty_layer_norm = nn.LayerNorm(self.hidden_size // 2)

        self.sty_proj = nn.Linear(
            self.hidden_size // 4, self.hidden_size // 4, bias=True
        )

        self.azimuth_bins = nn.Parameter(
            torch.linspace(-np.pi / 2, np.pi, 4), requires_grad=False
        )
        self.azimuth_emb = Embedding(4, self.hidden_size // 8)
        self.elevation_bins = nn.Parameter(
            torch.linspace(np.pi / 2, np.pi, 2), requires_grad=False
        )
        self.elevation_emb = Embedding(2, self.hidden_size // 8)

        self.emo_proj = nn.Linear(
            self.hidden_size // 4, self.hidden_size // 4, bias=True
        )
        self.azi_proj = nn.Linear(
            self.hidden_size // 4, self.hidden_size // 4, bias=True
        )
        self.ele_proj = nn.Linear(
            self.hidden_size // 4, self.hidden_size // 4, bias=True
        )

    def forward(
        self,
        speakers,
        emotions,
        arousals,
        valences,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        polar_coordinates=None,  # Sphere
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        # Sphere
        spherical_emotion_vector = polar_coordinates.squeeze(1)
        emo_VAD_inten = spherical_emotion_vector[:, 0:1]
        emo_VAD_style = spherical_emotion_vector[:, 1:3]

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)  # torch.Size([16, 37, 256])
        
        if (
            self.speaker_emb is not None
        ):  # 256 차원 -> model_config["transformer"]["encoder_hidden"]
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        # emotion embedding 값 추가
        if self.emotion_emb is not None:
            output = output + self.emotion_emb(emotions).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        # Sphere 

        #########################
        #   Reference Encoder   #
        #########################

        spks_embed = 0
        if self.use_spk_lookup:
            # spk_embed = self.forward_style_embed(speakers)
            spk_embed = spks_embed + self.spk_id_proj(speakers)
        spks_embed = spks_embed + spk_embed[:, None, :]

        emos_embed = 0
        if self.use_emo_lookup:
            emos_embed = emos_embed + self.emo_id_proj(
                emotions
            )  #  tensor([2], device='cuda:0')
        # torch.Size([1, 64]), torch.Size([1, 64])
        emos_proj_embed = self.emo_proj(emos_embed)

        intens_embed = 0
        # if self.hparams["emo_inten"] != None:
        #     emo_VAD_inten[0, 0] = self.hparams["emo_inten"]
        #     emo_VAD_inten = torch.clamp(emo_VAD_inten, min=0, max=1)
        intens_embed = intens_embed + self.emo_VAD_inten_proj(emo_VAD_inten)
        # torch.Size([1, 128]), torch.Size([1, 128]), torch.Size([1, 1])

        ele_embed = 0
        elevation = emo_VAD_style[:,0:1]  
        # torch.Size([1, 1]),  tensor([[0.6973]], device='cuda:0')
        elevation_index = torch.bucketize(elevation, self.elevation_bins)
        elevation_index = elevation_index.squeeze(1)
        elevation_embed = self.elevation_emb(elevation_index)
        ele_embed = elevation_embed + ele_embed  # torch.Size([1, 32])

        azi_embed = 0
        azimuth = emo_VAD_style[:, 1:2]  # torch.Size([1, 1])
        azimuth_index = torch.bucketize(azimuth, self.azimuth_bins)
        azimuth_index = azimuth_index.squeeze(1)
        azimuth_embed = self.azimuth_emb(azimuth_index)
        azi_embed = azimuth_embed + azi_embed  #  torch.Size([1, 32])

        style_embed = torch.cat((ele_embed, azi_embed), dim=-1)  # torch.Size([1, 64])
        style_proj_embed = self.sty_proj(style_embed)  # torch.Size([1, 64])

        # emo_all_emb
        # Softplus
        combined_embedding = torch.cat(
            (emos_proj_embed, style_proj_embed), dim=-1
        )  # torch.Size([1, 128])
        emotion_embedding = F.softplus(combined_embedding)  # torch.Size([1, 128])
        emosty_embed = self.emosty_layer_norm(emotion_embedding)  # torch.Size([1, 128])
        emo_all_emb = (intens_embed + emosty_embed)[:, None, :]  # torch.Size([batch_size, 1, 128])

        # out_embed를 만들어야 한다.
        out_embed = torch.cat(
            (spks_embed, emo_all_emb), dim=-1
        )  #  torch.Size([16, 1, 256])

        output = output + out_embed


        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
