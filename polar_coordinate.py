import os

import torch
import torch.nn as nn
from torch.nn.functional import pairwise_distance

from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

import numpy as np

from tqdm import tqdm

from glob import glob


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
        self,
        input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states_all = outputs[0]
        hidden_states = torch.mean(hidden_states_all, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states_all, logits


def load_filelist_dict(filelist_dir):
    filelist_dict, emotion_dict, arousal_dict, valence_dict = (
        dict(),
        dict(),
        dict(),
        dict(),
    )
    emotions, arousals, valences = set(), set(), set()
    with open(filelist_dir, "r", encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()):
            basename, aux_data = line.split("|")[0], line.split("|")[3:]
            filelist_dict[basename] = "|".join(aux_data).strip("\n")
            emotions.add(aux_data[-3])
            arousals.add(aux_data[-2])
            valences.add(aux_data[-1].strip("\n"))
    for i, emotion in enumerate(list(emotions)):
        emotion_dict[emotion] = i
    for i, arousal in enumerate(list(arousals)):
        arousal_dict[arousal] = i
    for i, valence in enumerate(list(valences)):
        valence_dict[valence] = i
    emotion_dict = {
        "emotion_dict": emotion_dict,
        "arousal_dict": arousal_dict,
        "valence_dict": valence_dict,
    }
    return filelist_dict, emotion_dict


def convert_tensor_to_polar_coordinates(tensor, max_distance):

    # tensor : d_a(arousal, 각성 정도), d_v(valence, 불쾌정도), d_d(dominance, 통제 정도)
    # radial distance r = sqrt(x^2 + y^2 + z^2)
    r = torch.sqrt(torch.sum(tensor**2, dim=1))

    # normalize를 감정값 마다 진행하는 듯하다.
    r_norm = torch.sqrt(torch.sum(tensor**2, dim=1)) / max_distance  # 반지름 계산

    # \theta : arccos(d_d/r)
    theta = torch.acos(tensor[:, 2] / r)  # 경사각 계산

    # \phi : arctan(d_v/d_a)
    phi = torch.atan2(tensor[:, 1], tensor[:, 0])  # 방위각 계산

    polar_coordinates_tensor = torch.stack(
        (r_norm, theta, phi), dim=1
    )  # 변환된 좌표 텐서

    # torch.Size([1, 3])
    return polar_coordinates_tensor


if __name__ == "__main__":

    out_dir = ""
    model_name = ""
    wav_path = ""

    filelist, emotions = load_filelist_dict(
        ""
    )

    os.makedirs((os.path.join(out_dir, "polar_coordinate")), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = EmotionModel.from_pretrained(model_name)
    model.to(device)

    items = glob(os.path.join(wav_path, "*.npy"))


    VAD_Neu_center = torch.tensor([0.4135, 0.5169, 0.3620]).to(device)
    all_distances = []

    VAD_values = {0: [], 1: [], 2: [], 3: [], 4: []}
    VAD_values2 = {0: [], 1: [], 2: [], 3: [], 4: []}
    max_distance = {}
    # wav ->
    for item in tqdm(items, desc="Processing items"):
        filename = os.path.basename(item)
        speaker = filename.split("-")[0]
        basename = filename.split("-")[2].split(".")[0]
        aux_data = filelist[basename]
        emotion = aux_data.split("|")[1]
        emo_id = emotions["emotion_dict"][emotion]
        wav = np.load(item)
        wav_gt = torch.from_numpy(wav)
        wav_gt = wav_gt.unsqueeze(0)  # torch.Size([1, 40960])
        pro_wav = processor(wav_gt, sampling_rate=16000)  

        pro_wav = pro_wav["input_values"][0]
        pro_wav = torch.from_numpy(pro_wav)  # (1, 40960)

        with torch.no_grad():
            _, emo_VAD = model(pro_wav.to(device))  # torch.Size([1, 3])
            VAD_values[emo_id].append(emo_VAD)
            VAD_values2[emo_id].append((emo_VAD, basename))

    for emo_id in tqdm(VAD_values):
        print(f"Calculating statistics for emo_id: {emo_id}")

        all_distances = []
        for emo_vad, item_name in VAD_values2[emo_id]:
            distance_to_neutral = pairwise_distance(
                emo_vad.unsqueeze(0), VAD_Neu_center, p=2
            )
            all_distances.append(distance_to_neutral.item())
        max_distance[emo_id] = max(all_distances)

    # else:
    VAD_Neu_center = torch.tensor([0.4135, 0.5169, 0.3620]).to(device)
    for item in tqdm(items, desc="Processing items"):

        filename = os.path.basename(item)
        speaker = filename.split("-")[0]
        basename = filename.split("-")[2].split(".")[0]
        aux_data = filelist[basename]
        emotion = aux_data.split("|")[1]
        emo_id = emotions["emotion_dict"][emotion]
        wav = np.load(item)
        wav_gt = torch.from_numpy(wav)
        wav_gt = wav_gt.unsqueeze(0)  # torch.Size([1, 40960])
        pro_wav = processor(wav_gt, sampling_rate=16000)  # 여기서 wav 파일 읽고

        pro_wav = pro_wav["input_values"][0]
        pro_wav = torch.from_numpy(pro_wav)  # (1, 40960)

        with torch.no_grad():
            hidden_emotion_vector, emo_VAD = model(pro_wav.to(device))

        re_emo_VAD = emo_VAD - VAD_Neu_center


        spherical_emotion_vector = convert_tensor_to_polar_coordinates(
            re_emo_VAD, max_distance[emo_id]
        ).to("cpu")

        # sphere
        polar_coordinate_filename = "{}-polar_coordinate-{}.npy".format(
            speaker, basename
        )
        np.save(
            os.path.join(out_dir, "polar_coordinate", polar_coordinate_filename),
            spherical_emotion_vector,
        )
