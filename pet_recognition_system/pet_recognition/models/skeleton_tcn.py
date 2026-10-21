"""Motion-aware temporal convolution baseline for dog behaviour."""
from __future__ import annotations

import torch
from torch import nn


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels), nn.Dropout(dropout),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.network(x))


class DogBehaviorTCN(nn.Module):
    """Fuse pose, explicit joint motion and box motion with dilated TCNs."""

    def __init__(self, num_classes=5, channels=64, dropout=0.35):
        super().__init__()
        # 51 pose values + 51 first differences + 6 global motion values.
        self.input_projection = nn.Sequential(
            nn.Conv1d(17 * 3 * 2 + 6, channels, 1),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[
            TemporalResidualBlock(channels, dilation, dropout)
            for dilation in (1, 2, 4)
        ])
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None,
                global_motion: torch.Tensor | None = None) -> torch.Tensor:
        sequence = x[..., 0].permute(0, 2, 3, 1).flatten(2)
        velocity = torch.zeros_like(sequence)
        velocity[:, 1:] = sequence[:, 1:] - sequence[:, :-1]
        if global_motion is None:
            global_motion = sequence.new_zeros((*sequence.shape[:2], 6))
        features = torch.cat((sequence, velocity, global_motion), dim=2).transpose(1, 2)
        features = self.blocks(self.input_projection(features))
        if frame_mask is None:
            pooled = features.mean(dim=2)
        else:
            mask = frame_mask[:, None, :]
            pooled = (features * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
        return self.classifier(pooled)


class DogBehaviorMotionAuxTCN(DogBehaviorTCN):
    """Confidence-aware TCN with an auxiliary moving-versus-static objective."""

    auxiliary_loss_weight = 0.2

    def __init__(self, num_classes=5, channels=64, dropout=0.35):
        super().__init__(num_classes=num_classes, channels=channels, dropout=dropout)
        self.movement_head = nn.Linear(channels, 1)

    def _encode(self, x: torch.Tensor, frame_mask: torch.Tensor | None,
                global_motion: torch.Tensor | None) -> torch.Tensor:
        sequence = x[..., 0].permute(0, 2, 3, 1).flatten(2)
        velocity = torch.zeros_like(sequence)
        velocity[:, 1:] = sequence[:, 1:] - sequence[:, :-1]
        if global_motion is None:
            global_motion = sequence.new_zeros((*sequence.shape[:2], 6))
        features = torch.cat((sequence, velocity, global_motion), dim=2).transpose(1, 2)
        encoded = self.blocks(self.input_projection(features))

        # Joint confidence is part of the input stream. Use it again as an
        # explicit temporal reliability weight so distant/occluded frames do
        # not contribute as much as clean frames.
        confidence = x[:, 2, :, :, 0].mean(dim=2).clamp(0, 1)
        if frame_mask is not None:
            confidence = confidence * frame_mask
        weight = confidence[:, None, :]
        return (encoded * weight).sum(2) / weight.sum(2).clamp_min(1e-4)

    def forward_with_aux(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None,
                         global_motion: torch.Tensor | None = None):
        pooled = self._encode(x, frame_mask, global_motion)
        return self.classifier(pooled), self.movement_head(pooled).squeeze(1)

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None,
                global_motion: torch.Tensor | None = None) -> torch.Tensor:
        logits, _ = self.forward_with_aux(x, frame_mask, global_motion)
        return logits


class DogBehaviorMotionSummaryTCN(DogBehaviorTCN):
    """TCN with explicit clip-level displacement and action-amplitude cues."""

    def __init__(self, num_classes=5, channels=64, dropout=0.35):
        super().__init__(num_classes=num_classes, channels=channels, dropout=dropout)
        # Six motion signals × (mean, maximum, standard deviation), plus
        # cumulative centre/joint paths, final displacement and x/y spans.
        self.motion_summary = nn.Sequential(
            nn.Linear(23, 32), nn.LayerNorm(32), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(channels + 32, num_classes)

    @staticmethod
    def _masked_statistics(signals: torch.Tensor,
                           frame_mask: torch.Tensor | None) -> torch.Tensor:
        if frame_mask is None:
            mean = signals.mean(1)
            maximum = signals.max(1).values
            std = signals.std(1, unbiased=False)
            return torch.cat((mean, maximum, std), dim=1)
        mask = frame_mask[:, :, None]
        count = mask.sum(1).clamp_min(1)
        mean = (signals * mask).sum(1) / count
        variance = ((signals - mean[:, None]) ** 2 * mask).sum(1) / count
        maximum = signals.masked_fill(mask <= 0, float("-inf")).max(1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum, variance.sqrt()), dim=1)

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None,
                global_motion: torch.Tensor | None = None) -> torch.Tensor:
        sequence = x[..., 0].permute(0, 2, 3, 1).flatten(2)
        velocity = torch.zeros_like(sequence)
        velocity[:, 1:] = sequence[:, 1:] - sequence[:, :-1]
        if global_motion is None:
            global_motion = sequence.new_zeros((*sequence.shape[:2], 6))
        features = torch.cat((sequence, velocity, global_motion), dim=2).transpose(1, 2)
        encoded = self.blocks(self.input_projection(features))
        if frame_mask is None:
            pooled = encoded.mean(dim=2)
            valid_count = sequence.new_full((sequence.shape[0], 1), sequence.shape[1])
            last_index = sequence.new_full((sequence.shape[0],), sequence.shape[1] - 1,
                                           dtype=torch.long)
        else:
            mask = frame_mask[:, None, :]
            pooled = (encoded * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
            valid_count = frame_mask.sum(1, keepdim=True).clamp_min(1)
            last_index = (valid_count[:, 0].long() - 1).clamp_min(0)

        joints = sequence.view(sequence.shape[0], sequence.shape[1], 17, 3)
        joint_velocity = velocity.view(velocity.shape[0], velocity.shape[1], 17, 3)[..., :2]
        joint_speed = torch.linalg.vector_norm(joint_velocity, dim=3)
        joint_mean_speed = joint_speed.mean(2)
        joint_max_speed = joint_speed.max(2).values
        centre_displacement = torch.linalg.vector_norm(global_motion[..., :2], dim=2)
        centre_speed = torch.linalg.vector_norm(global_motion[..., 2:4], dim=2)
        scale_change = global_motion[..., 4].abs()
        scale_speed = global_motion[..., 5].abs()
        signals = torch.stack((centre_displacement, centre_speed,
                               joint_mean_speed, joint_max_speed,
                               scale_change, scale_speed), dim=2)
        statistics = self._masked_statistics(signals, frame_mask)
        if frame_mask is None:
            valid = torch.ones_like(centre_speed)
        else:
            valid = frame_mask
        centre_path = (centre_speed * valid).sum(1, keepdim=True)
        joint_path = (joint_mean_speed * valid).sum(1, keepdim=True)
        batch = torch.arange(sequence.shape[0], device=sequence.device)
        final_displacement = centre_displacement[batch, last_index, None]
        displacement_xy = global_motion[..., :2]
        if frame_mask is not None:
            mask_xy = frame_mask[:, :, None]
            maximum_xy = displacement_xy.masked_fill(mask_xy <= 0, float("-inf")).max(1).values
            minimum_xy = displacement_xy.masked_fill(mask_xy <= 0, float("inf")).min(1).values
        else:
            maximum_xy, minimum_xy = displacement_xy.max(1).values, displacement_xy.min(1).values
        span_xy = maximum_xy - minimum_xy
        summary = torch.cat((statistics, centre_path, joint_path,
                             final_displacement, span_xy), dim=1)
        summary = self.motion_summary(summary)
        return self.classifier(torch.cat((pooled, summary), dim=1))
