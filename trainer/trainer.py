import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import Parallel, delayed

from trainer.base_trainer import BaseTrainer
from util.others import set_requires_grad
from util.speech_processing import compute_STOI, compute_PESQ

plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(self,
                 config: dict,
                 resume: bool,
                 generator,
                 discriminator,
                 generator_optimizer,
                 discriminator_optimizer,
                 loss_function,
                 train_dl,
                 validation_dl):
        super(Trainer, self).__init__(
            config,
            resume,
            generator,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            loss_function
        )
        self.train_dataloader = train_dl
        self.validation_dataloader = validation_dl

    @torch.no_grad()
    def compute_batch_metric(self, batch_m, noisy_phase_list, clean_list, metric="pesq"):
        """
        计算 batch 内的评价指标，输入与输出均为 numpy 格式

        Args:
            batch_m: one batch stft matrix, [batch_size, 1, F, T]
            clean_list: one batch clean signal
            metric: "pesq" or "stoi"

        Returns:
            metrics list
        """
        assert len(batch_m) == len(noisy_phase_list) == len(clean_list)

        if metric == "pesq":
            metric_function = compute_PESQ
        elif metric == "stoi":
            metric_function = compute_STOI
        else:
            raise NotImplemented(f"Metric {metric} is not implemented.")

        batch_length = Parallel(n_jobs=30)(delayed(len)(i) for i in clean_list)

        batch_y = Parallel(n_jobs=30)(
            delayed(librosa.istft)(
                m, win_length=512, hop_length=256, length=length
            ) for m, length in zip(batch_m, batch_length)
        )

        batch_metric = Parallel(n_jobs=30)(
            delayed(metric_function)(clean, noisy) for clean, noisy in zip(clean_list, batch_y))

        return batch_metric

    def gpu_to_cpu(self, m):
        return m.detach().to("cpu").numpy()

    def cpu_to_gpu(self, m):
        return torch.tensor(m).to(self.device)

    def _train_epoch(self, epoch):
        batch_size = self.train_dataloader.batch_size
        n_batch = len(self.train_dataloader)
        for i, (_, clean_list, _, noisy_mag, noisy_phase_list, clean_mag) in enumerate(
                self.train_dataloader, start=1):
            # For visualization
            n_iter = n_batch * batch_size * (epoch - 1) + i * batch_size

            noisy_mag = noisy_mag.to(self.device)
            clean_mag = clean_mag.to(self.device)

            pred_mask = self.generator(noisy_mag)
            pred_mask = torch.max(pred_mask, torch.full(pred_mask.shape, 0.05).to(self.device))
            enhanced_mag = noisy_mag * pred_mask

            """================ Optimize D ================"""
            set_requires_grad(self.discriminator, True)
            self.optimizer_D.zero_grad()

            # D(clean, clean) => 1
            clean_clean_pair = torch.cat((clean_mag, clean_mag), dim=1)
            clean_clean_score_in_D = self.discriminator(clean_clean_pair)
            clean_clean_loss_in_D = self.loss_function(
                clean_clean_score_in_D,
                torch.ones(clean_clean_score_in_D.shape).to(self.device)
            )

            # D(enhanced, clean) => PESQ(enhanced, clean) or STOI(enhanced, clean)
            enhanced_clean_pair = torch.cat((enhanced_mag.detach(), clean_mag), dim=1)
            enhanced_clean_score_in_D = self.discriminator(enhanced_clean_pair)
            enhanced_clean_score_in_metric = self.compute_batch_metric(
                enhanced_mag.detach().cpu().squeeze(1).numpy(),
                noisy_phase_list,
                clean_list,
                "pesq"
            )
            enhanced_clean_score_in_metric = torch.tensor(enhanced_clean_score_in_metric).unsqueeze(1).to(self.device)
            enhanced_clean_loss_in_D = self.loss_function(enhanced_clean_score_in_D, enhanced_clean_score_in_metric)

            loss_D = (clean_clean_loss_in_D + enhanced_clean_loss_in_D) * 0.5
            loss_D.backward()
            self.optimizer_D.step()

            with torch.no_grad():
                self.writer.add_scalar(f"判别器/总损失", loss_D, n_iter)
                self.writer.add_scalar(f"判别器/D(y, y)", clean_clean_loss_in_D.item(), n_iter)
                self.writer.add_scalar(f"判别器/D(G(x), y)", enhanced_clean_loss_in_D.item(), n_iter)

            """================ Optimize G ================"""
            set_requires_grad(self.discriminator, False)
            self.optimizer_G.zero_grad()

            # D(enhanced, clean) => 1
            enhanced_clean_pair = torch.cat((enhanced_mag, clean_mag), dim=1)
            enhanced_clean_score_in_G = self.discriminator(enhanced_clean_pair)
            loss_G = self.loss_function(enhanced_clean_score_in_G, torch.ones(enhanced_clean_score_in_G.shape))

            loss_G.backward()
            self.optimizer_G.step()

            with torch.no_grad():
                self.writer.add_scalar(f"生成器/损失", loss_G.item(), n_iter)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]

        stoi_c_n = []  # Clean and Noisy
        stoi_c_e = []  # Clean and Enhanced
        pesq_c_n = []
        pesq_c_e = []

        for i, (noisy, clean, name, noisy_mag, noisy_phase, clean_mag) in enumerate(self.validation_dataloader):
            assert len(name) == 1, "The batch size of validation dataloader must be 1."
            name = name[0]
            noisy = noisy[0]
            clean = clean[0]

            # noisy_mag, noisy_phase, noisy_length = mag_and_phase(noisy)

            noisy_mag = noisy_mag.to(self.device)  # [1, 1, F, T]
            pred_mask = self.generator(noisy_mag)
            pred_mask = torch.max(pred_mask, torch.full(pred_mask.shape, 0.05).to(self.device))
            enhanced_mag = noisy_mag * pred_mask
            enhanced_mag = enhanced_mag.detach().cpu().numpy()

            enhanced = librosa.istft(enhanced_mag * noisy_phase.numpy(), hop_length=256, win_length=512, length=len(noisy))
            assert len(noisy) == len(clean) == len(enhanced)

            # Visualize audio
            if i <= visualize_audio_limit:
                self.writer.add_audio(f"Speech/{name}_Noisy", noisy, epoch, sample_rate=16000)
                self.writer.add_audio(f"Speech/{name}_Clean", clean, epoch, sample_rate=16000)
                self.writer.add_audio(f"Speech/{name}_Enhanced", enhanced, epoch, sample_rate=16000)

            # Visualize waveform
            if i <= visualize_waveform_limit:
                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([noisy, clean, enhanced]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        np.mean(y),
                        np.std(y),
                        np.max(y),
                        np.min(y)
                    ))
                    librosa.display.waveplot(y, sr=16000, ax=ax[j])
                plt.tight_layout()
                self.writer.add_figure(f"Waveform/{name}", fig, epoch)

            # Visualize spectrogram
            noisy_mag, _ = librosa.magphase(librosa.stft(noisy, n_fft=320, hop_length=160, win_length=320))
            enhanced_mag, _ = librosa.magphase(librosa.stft(enhanced, n_fft=320, hop_length=160, win_length=320))
            clean_mag, _ = librosa.magphase(librosa.stft(clean, n_fft=320, hop_length=160, win_length=320))

            if i <= visualize_spectrogram_limit:
                fig, axes = plt.subplots(3, 1, figsize=(6, 6))
                for k, mag in enumerate([
                    noisy_mag,
                    enhanced_mag,
                    clean_mag,
                ]):
                    axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
                                      f"std: {np.std(mag):.3f}, "
                                      f"max: {np.max(mag):.3f}, "
                                      f"min: {np.min(mag):.3f}")
                    librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k],
                                             sr=16000)
                plt.tight_layout()
                self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)

            # Metrics
            stoi_c_n.append(compute_STOI(clean, noisy, sr=16000))
            stoi_c_e.append(compute_STOI(clean, enhanced, sr=16000))
            pesq_c_n.append(compute_PESQ(clean, noisy, sr=16000))
            pesq_c_e.append(compute_PESQ(clean, enhanced, sr=16000))

        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        self.writer.add_scalars(f"Metrics/STOI", {
            "clean and noisy": get_metrics_ave(stoi_c_n),
            "clean and denoisy": get_metrics_ave(stoi_c_e)
        }, epoch)
        self.writer.add_scalars(f"Metrics/PESQ", {
            "clean and noisy": get_metrics_ave(pesq_c_n),
            "clean and denoisy": get_metrics_ave(pesq_c_e)
        }, epoch)

        score = (get_metrics_ave(stoi_c_e) + self._transform_pesq_range(get_metrics_ave(pesq_c_e))) / 2
        return score
