import torch
import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    """FastSpeech2 Loss"""

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[9:15]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions

        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, : mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )

class FastSpeech2AdversarialLoss(nn.Module):
    """FastSpeech2 Loss"""

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2AdversarialLoss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward_gen(self, inputs, predictions, predictions_disc=None):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[9:15]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
            emo_cond_embed,
            spk_cond_embed,
        ) = predictions
        if predictions_disc is not None:
            (
                e_p_cond,
                s_p_cond,
            ) = predictions_disc

        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, : mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        if predictions_disc is not None:
            e_p_cond_loss = self.mse_loss(e_p_cond, e_p_cond.new_ones(e_p_cond.size()))
            s_p_cond_loss = self.mse_loss(s_p_cond, s_p_cond.new_ones(s_p_cond.size()))

            # TODO 
            # lambda_mel_adv = 0.05
            lambda_mel_adv = 1.0
                    
            total_loss = (
                mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + e_p_cond_loss*lambda_mel_adv + s_p_cond_loss*lambda_mel_adv
            )

            return (
                total_loss,
                mel_loss,
                postnet_mel_loss,
                pitch_loss,
                energy_loss,
                duration_loss,
                e_p_cond_loss,
                s_p_cond_loss,
            )
        else:
            total_loss = (
                mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
            )

            return (
                total_loss,
                mel_loss,
                postnet_mel_loss,
                pitch_loss,
                energy_loss,
                duration_loss,
            )

    def forward_disc(self, predictions_disc_gt, predictions_disc_pred):
        (
            e_g_cond,
            s_g_cond,
        ) = predictions_disc_gt
        (
            e_p_cond,
            s_p_cond,
        ) = predictions_disc_pred

        e_g_cond_loss = self.mse_loss(e_g_cond, e_g_cond.new_ones(e_g_cond.size()))
        s_g_cond_loss = self.mse_loss(s_g_cond, s_g_cond.new_ones(s_g_cond.size()))
        disc_real_loss = e_g_cond_loss + s_g_cond_loss

        e_p_cond_loss = self.mse_loss(e_p_cond, e_p_cond.new_zeros(e_p_cond.size()))
        s_p_cond_loss = self.mse_loss(s_p_cond, s_p_cond.new_zeros(s_p_cond.size()))
        disc_fake_loss = e_p_cond_loss + s_p_cond_loss

        # TODO 
        lambda_mel_adv = 1.0
        
        total_loss = (
            (disc_real_loss*0.5)*lambda_mel_adv + (disc_fake_loss*0.5)*lambda_mel_adv
        )

        return (
            total_loss,
            disc_real_loss,
            disc_fake_loss,
            e_g_cond_loss,
            s_g_cond_loss,
            e_p_cond_loss,
            s_p_cond_loss,
        )
