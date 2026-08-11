import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinResolvedSpectralEncoder(nn.Module):
    def __init__(
        self,
        c_in: int,
        d_model: int,
        input_length: int,
        periods: Sequence[int] = (24, 168),
        num_frequencies: int = 13,
        use_periodic: bool = True,
        use_context: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_length = input_length
        self.periods = tuple(periods)
        self.num_frequencies = num_frequencies
        self.use_periodic = use_periodic
        self.use_context = use_context
        self.input_projection = nn.Linear(c_in, d_model)
        periodic_dim = d_model // 2
        n_periods = len(self.periods)
        base = periodic_dim // n_periods
        rem = periodic_dim - base * n_periods
        periodic_parts = [base + (1 if i < rem else 0) for i in range(n_periods)]
        self.periodic_projections = nn.ModuleList(
            [nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d)) for d in periodic_parts]
        )
        self.context_projection = nn.Linear(c_in, d_model // 2)
        self.bin_processors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )
                for _ in range(num_frequencies)
            ]
        )
        self.frequency_combiner = nn.Sequential(
            nn.Linear(d_model * num_frequencies, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.temporal_pool = nn.Parameter(torch.randn(input_length, 1))

    def _periodic_features(self, time_idx: torch.Tensor) -> torch.Tensor:
        outputs = []
        for period, projection in zip(self.periods, self.periodic_projections):
            phase = 2.0 * math.pi * time_idx / period
            sin_cos = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)
            outputs.append(projection(sin_cos))
        return torch.cat(outputs, dim=-1)

    def forward(self, x: torch.Tensor):
        batch, steps, _, _ = x.shape
        hidden = self.input_projection(x)
        time_idx = torch.arange(steps, device=x.device, dtype=x.dtype).unsqueeze(0).expand(batch, -1)

        if self.use_periodic:
            periodic = self._periodic_features(time_idx)
        else:
            periodic = torch.zeros(
                batch,
                steps,
                self.d_model // 2,
                device=x.device,
                dtype=x.dtype,
            )

        if self.use_context:
            context = self.context_projection(x.mean(dim=2))
        else:
            context = torch.zeros(
                batch,
                steps,
                self.d_model // 2,
                device=x.device,
                dtype=x.dtype,
            )

        hidden = hidden + torch.cat([periodic, context], dim=-1).unsqueeze(2)
        spectrum = torch.fft.rfft(hidden.permute(0, 2, 3, 1), dim=-1)
        available_bins = spectrum.shape[-1]
        bin_features = []

        for index in range(min(self.num_frequencies, available_bins)):
            real_imag = torch.cat(
                [spectrum[..., index].real, spectrum[..., index].imag], dim=-1
            )
            bin_features.append(self.bin_processors[index](real_imag))

        if not bin_features:
            bin_features.append(
                torch.zeros(
                    batch,
                    x.shape[2],
                    self.d_model,
                    device=x.device,
                    dtype=x.dtype,
                )
            )

        while len(bin_features) < self.num_frequencies:
            bin_features.append(torch.zeros_like(bin_features[0]))

        bins = torch.stack(bin_features, dim=2)
        spectral_signature = F.normalize(bins.norm(dim=-1), dim=-1)
        frequency_correction = self.frequency_combiner(torch.cat(bin_features, dim=-1))
        hidden = hidden + 0.3 * frequency_correction.unsqueeze(1)
        pooling_weights = torch.softmax(self.temporal_pool.squeeze(-1), dim=0)
        spatial_representation = (
            hidden * pooling_weights[None, :, None, None]
        ).sum(dim=1)
        return spatial_representation, bins, spectral_signature


class BinPrototypeAffinity(nn.Module):
    def forward(
        self, bin_features: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        normalized_bins = F.normalize(bin_features, dim=-1)
        normalized_prototypes = F.normalize(prototypes, dim=-1)
        similarities = torch.einsum(
            "bnfd,kd->bnfk", normalized_bins, normalized_prototypes
        )
        return similarities.max(dim=2).values


class FrequencyAwarePrototypeRouting(nn.Module):
    def __init__(
        self,
        d_model: int,
        h: int,
        w: int,
        num_prototypes: int,
        num_frequencies: int,
        epsilon: float = 0.08,
        temperature: float = 1.4,
        use_latent_distance: bool = True,
        dropout: float = 0.1,
        coords: Optional[torch.Tensor] = None,
        node_mode: bool = True,
        bpa_init_alpha: float = 0.5,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.epsilon = epsilon
        self.temperature = temperature
        self.use_latent_distance = use_latent_distance
        self.node_mode = node_mode
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model) * 0.02)
        self.prototype_coordinates = nn.Parameter(torch.randn(num_prototypes, 2) * 0.1)
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.prototype_projection = nn.Linear(d_model, d_model, bias=False)
        self.bpa = BinPrototypeAffinity()
        self.norm = nn.LayerNorm(d_model)
        self.refinement = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.spectral_gate = nn.Sequential(
            nn.Linear(num_frequencies, d_model), nn.Sigmoid()
        )
        self.log_alpha = nn.Parameter(torch.tensor(math.log(bpa_init_alpha)))

        if coords is None:
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, h),
                torch.linspace(-1, 1, w),
                indexing="ij",
            )
            coords_buffer = torch.stack([yy, xx], dim=-1).reshape(-1, 2)
        else:
            coords_buffer = coords.float()
            if coords_buffer.size(1) != 2:
                coords_buffer = coords_buffer[:, :2].contiguous()
        self.register_buffer("coords", coords_buffer, persistent=False)

    def forward(
        self,
        spatial_representation: torch.Tensor,
        bin_features: torch.Tensor,
        spectral_signature: torch.Tensor,
    ) -> torch.Tensor:
        batch = spatial_representation.shape[0]
        query = self.query_projection(spatial_representation)
        projected_prototypes = self.prototype_projection(self.prototypes)
        routed_prototypes = projected_prototypes.unsqueeze(0).expand(batch, -1, -1)

        if self.use_latent_distance:
            normalized_query = F.normalize(query, dim=-1)
            normalized_prototypes = F.normalize(routed_prototypes, dim=-1)
            cost = 1.0 - torch.einsum(
                "bnd,bkd->bnk", normalized_query, normalized_prototypes
            )
        else:
            cost = torch.zeros(
                batch,
                spatial_representation.size(1),
                self.num_prototypes,
                device=spatial_representation.device,
                dtype=spatial_representation.dtype,
            )

        if not self.node_mode:
            coordinates = self.coords.to(spatial_representation.device)
            prototype_coordinates = torch.tanh(self.prototype_coordinates)
            euclidean_cost = torch.cdist(
                coordinates.unsqueeze(0),
                prototype_coordinates.unsqueeze(0),
                p=2.0,
            ).expand(batch, -1, -1)
            cost = cost + euclidean_cost

        affinity = self.bpa(bin_features, self.prototypes)
        alpha = F.softplus(self.log_alpha)
        frequency_aware_cost = cost - alpha * affinity
        routing_weights = torch.softmax(
            -self.temperature * frequency_aware_cost / (self.epsilon + 1e-6),
            dim=-1,
        )
        prototype_message = torch.einsum(
            "bnk,bkd->bnd", routing_weights, routed_prototypes
        )
        output = self.norm(spatial_representation + prototype_message)
        output = self.norm(output + self.refinement(output))
        output = output * self.spectral_gate(spectral_signature)
        return output

    def get_alpha(self) -> float:
        return float(F.softplus(self.log_alpha).item())


class BPRNet(nn.Module):
    def __init__(
        self,
        h: int,
        w: int,
        c: int,
        p: int,
        d_model: int = 128,
        periods: Sequence[int] = (24, 168),
        num_frequencies: int = 13,
        dropout: float = 0.1,
        num_prototypes: int = 128,
        routing_epsilon: float = 0.08,
        routing_temperature: float = 1.4,
        use_latent_distance: bool = True,
        bpa_init_alpha: float = 0.5,
        node_mode: bool = True,
        coords: Optional[torch.Tensor] = None,
        use_periodic: bool = True,
        use_context: bool = True,
        residual_to_last: bool = True,
    ):
        super().__init__()
        self.h = h
        self.w = w
        self.c = c
        self.p = p
        self.residual_to_last = residual_to_last
        self.brse = BinResolvedSpectralEncoder(
            c_in=c,
            d_model=d_model,
            input_length=p,
            periods=periods,
            num_frequencies=num_frequencies,
            use_periodic=use_periodic,
            use_context=use_context,
        )
        self.fapr = FrequencyAwarePrototypeRouting(
            d_model=d_model,
            h=h,
            w=w,
            num_prototypes=num_prototypes,
            num_frequencies=num_frequencies,
            epsilon=routing_epsilon,
            temperature=routing_temperature,
            use_latent_distance=use_latent_distance,
            dropout=dropout,
            coords=coords,
            node_mode=node_mode,
            bpa_init_alpha=bpa_init_alpha,
        )
        self.output_projection = nn.Linear(d_model, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, h, w, channels = x.shape
        nodes = h * w
        last_input = x[:, -1]
        node_input = x.reshape(batch, steps, nodes, channels)
        spatial_representation, bin_features, spectral_signature = self.brse(node_input)
        routed_representation = self.fapr(
            spatial_representation, bin_features, spectral_signature
        )
        prediction_delta = self.output_projection(routed_representation).reshape(
            batch, h, w, channels
        )
        if self.residual_to_last:
            return last_input + prediction_delta
        return prediction_delta


BRSE = BinResolvedSpectralEncoder
BPA = BinPrototypeAffinity
FAPR = FrequencyAwarePrototypeRouting
