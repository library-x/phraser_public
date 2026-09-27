from typing import List

import librosa
import mir_eval
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt, gridspec
from sklearn.metrics import classification_report, accuracy_score
from matplotlib.path import Path
import matplotlib.patches as patches

from phraser.modules.phraser import PhraserModel
from phraser.utils.metric_logger import MetricLogger
from phraser.utils.models import Segment
from phraser.utils.utils import SECTION_COLORS, SECTION_LABELS, STEPS_PER_SEC, MUQ_SR, most_common_spacing, robust_periodic_reconstruction, BEATS_HZ

ENCODER_BATCH = 4


def calculate_flat_tolerant_metrics(y_true, y_pred, tolerance=1):
    # Ensure float type for pooling operations
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Reshape to [1, 1, N] for F.max_pool1d (expects 3D input: Batch, Channel, Length)
    y_true_3d = y_true.view(1, 1, -1)
    y_pred_3d = y_pred.view(1, 1, -1)

    # Create dilated (expanded) versions
    # kernel_size=3 with padding=1 covers [i-1, i, i+1]
    kernel_size = 2 * tolerance + 1
    y_true_dilated = F.max_pool1d(y_true_3d, kernel_size=kernel_size, stride=1, padding=tolerance)
    y_pred_dilated = F.max_pool1d(y_pred_3d, kernel_size=kernel_size, stride=1, padding=tolerance)

    # Flatten back to 1D
    y_true_dilated = y_true_dilated.view(-1)
    y_pred_dilated = y_pred_dilated.view(-1)

    # --- METRICS CALCULATION ---

    # 1. RECALL: True positives found within the tolerance window
    # Calculation: How many actual 1s have a predicted 1 nearby?
    tp_for_recall = (y_true * y_pred_dilated).sum()
    actual_positives = y_true.sum()
    recall = tp_for_recall / (actual_positives + 1e-7)

    # 2. PRECISION: Predicted positives that are "valid" within tolerance
    # Calculation: How many predicted 1s have an actual 1 nearby?
    tp_for_precision = (y_pred * y_true_dilated).sum()
    predicted_positives = y_pred.sum()
    precision = tp_for_precision / (predicted_positives + 1e-7)

    # 3. F1 SCORE: Harmonic mean of tolerant precision and recall
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    # 4. ACCURACY: (Total - Misclassifications) / Total
    # Note: Accuracy is still dominated by the huge amount of zeros (background)
    fn = actual_positives - tp_for_recall
    fp = predicted_positives - tp_for_precision
    total_elements = y_true.numel()
    accuracy = (total_elements - (fp + fn)) / total_elements

    return accuracy.item(), precision.item(), recall.item(), f1.item()


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster




def __train_test(model: PhraserModel, loader, optimizer, epoch, device, loggers: List[MetricLogger],  cfg):

    for batch_idx, (embedding, segments_info, stems_info, beats_info, urls ) in enumerate(loader):

        if cfg.distributed:
            torch.distributed.barrier()


        embedding = embedding.to(device)

        segments_split_encoded = segments_info['segments_splits'].to(device)
        segments_type_encoded = segments_info['segments_type'].to(device)
        elements_silent_encoded = stems_info['stem_silences'].to(device)
        elements_splits_encoded = stems_info['stem_splits'].to(device)
        on_set_encoding = beats_info['on_set'].to(device)
        beats_encoding = beats_info['beats'].to(device)

        (logits_all_split, logits_element_silent, logits_segment_predict), ss_matrix, (beats_predict, on_set_predict)  = model.forward_encoded(embedding)


        # gathered_logits_section = output.logits_section
        # gathered_logits_function = output.logits_function

        # gathered_logits_element_silent = logits_element_silent.unsqueeze(1).float()
        # gathered_logits_element_split = logits_element_split.permute(0, 2, 1)

        # if cfg.distributed:
        #
        #     output_tensor_list = [torch.zeros(gathered_logits_section.shape, device=device) for _ in range(cfg.world_size)]
        #     torch.distributed.all_gather(output_tensor_list, gathered_logits_section)
        #
        #     gathered_logits_section = torch.cat(output_tensor_list, dim=0)
        #
        #     output_tensor_list = [torch.zeros(gathered_logits_function.shape, device=device) for _ in range(cfg.world_size)]
        #     torch.distributed.all_gather(output_tensor_list, gathered_logits_function)
        #
        #     gathered_logits_function= torch.cat(output_tensor_list, dim=0)
        #
        #     output_tensor_list = [torch.zeros(gathered_splits_section.shape, device=device) for _ in range(cfg.world_size)]
        #     torch.distributed.all_gather(output_tensor_list, gathered_splits_section)
        #
        #     gathered_splits_section = torch.cat(output_tensor_list, dim=0)
        #
        #     output_tensor_list = [torch.zeros(gathered_function_section.shape, device=device) for _ in range(cfg.world_size)]
        #     torch.distributed.all_gather(output_tensor_list, gathered_function_section)
        #
        #     gathered_function_section = torch.cat(output_tensor_list, dim=0)

        if cfg.distributed:
            torch.distributed.barrier()

        loss_elements_splits = F.binary_cross_entropy_with_logits(logits_all_split[:, 1:4, :].squeeze(3), elements_splits_encoded[:, 1:, :].float())
        loss_elements_silent = F.binary_cross_entropy_with_logits(logits_element_silent[:, 1:4, :, :].squeeze(3), elements_silent_encoded[:, 1:, :].float())

        loss_segments_splits = F.binary_cross_entropy_with_logits(logits_all_split[:, -1, :].squeeze(2), segments_split_encoded.float())
        loss_segments_type = _label_ce(logits_segment_predict, segments_type_encoded)  # class-axis CE, weight via PHRASER_LABEL_W (axfix retrain)


        loss_beats = F.binary_cross_entropy_with_logits(beats_predict.squeeze(2), beats_encoding.float())
        loss_on_set = F.binary_cross_entropy_with_logits(on_set_predict.squeeze(2), on_set_encoding.float())

        # loss_elements_splits = loss_elements_splits # * cfg.loss_weight_section
        # loss_elements_silent = loss_elements_silent  #* cfg.loss_weight_function

        if __import__("os").environ.get("PHRASER_ONLY_TYPE", "0") == "1":
            loss = loss_segments_type  # diagnostic: single-task on the label head
        else:
            _stem_w = float(__import__("os").environ.get("PHRASER_STEM_W", "1.0"))  # 0 = no per-stem split/silence loss
            loss = _stem_w * (loss_elements_splits + loss_elements_silent) + loss_segments_splits + loss_segments_type + loss_beats + loss_on_set
        print(f"{batch_idx}: {loss.item()}")

        if model.training:
            # compute the gradients
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)

            # step
            optimizer.step()
            optimizer.zero_grad()


        if not model.training:
            # print("XD")
            # logits_section = output.logits_section.detach().cpu()
            # logits_function = output.logits_function.detach().cpu()

            logits_on_set_predict = F.sigmoid(on_set_predict).squeeze(2).detach().cpu()
            logits_elements_split = F.sigmoid(logits_all_split[:, :4, :].squeeze(3)).detach().cpu()
            logits_segments_split = F.sigmoid(logits_all_split[:, -1, :].squeeze(2)).detach().cpu()

            for idx in range(on_set_predict.shape[0]):
                on_set_predict_one = logits_on_set_predict[idx]
                elements_slit_one = logits_elements_split[idx]
                segments_split_one = logits_segments_split[idx]

                parsed_on_sets = robust_periodic_reconstruction(on_set_predict_one)

                onset_accuracy = accuracy_score(on_set_encoding[idx].cpu(), parsed_on_sets)
                onset_true_accuracy = accuracy_score(on_set_encoding[idx].cpu()[on_set_encoding[idx].cpu() == 1], parsed_on_sets[on_set_encoding[idx].cpu() == 1])

                parsed_splits = (elements_slit_one > 0.3).float().repeat_interleave(BEATS_HZ, dim=1) * parsed_on_sets

                stem_true = stems_info['stem_splits_long'][idx].flatten()
                stem_pred = parsed_splits.flatten()
                stems_splits_accuracy, stems_splits_precision, stems_splits_recall, stems_splits_f1 = calculate_flat_tolerant_metrics(stem_true, stem_pred, tolerance=1)



                for logger in loggers:
                    logger.update(onset_accuracy=onset_accuracy)
                    logger.update(onset_true_accuracy=onset_true_accuracy)
                    logger.update(stems_splits_accuracy=stems_splits_accuracy)
                    logger.update(stems_splits_precision=stems_splits_precision)
                    logger.update(stems_splits_recall=stems_splits_recall)
                    logger.update(stems_splits_f1=stems_splits_f1)

                

            try:
                pass
            #     for idx, segment in enumerate(segments):
            #         predicted_segments, boundary = postprocess_functional_structure_boundary(logits_section[idx].squeeze(0), logits_function[idx], cfg)
            #         correct_segments = [Segment(start=start, end=end, label=label) for (label, start, end) in segment]
            #
            #         scores = mir_eval.segment.evaluate(
            #                 np.array([(x.start, x.end) for x in correct_segments]),
            #                 [(SECTION_LABELS.index(x.label)) for x in correct_segments],
            #                 np.array([(x.start, x.end) for x in predicted_segments]),
            #                 [(SECTION_LABELS.index(x.label)) for x in predicted_segments],
            #                 trim=False
            #             )
            #
            #         boundary_scores = classification_report(segments_split_encoded[idx].detach().cpu(), boundary, output_dict=True)
            #         scores.update({f"Boundary {k}":v for k,v in boundary_scores['1'].items()})
            #
            #         for logger in loggers:
            #             logger.update(**scores)

            except Exception as e:
                print(f"ERROR {str(e)}")

        if cfg.is_master:
            for logger in loggers:
                logger.update(loss=loss.item())
                logger.update(loss_elements_splits=loss_elements_splits)
                logger.update(loss_elements_silent=loss_elements_silent)
                logger.update(loss_segments_splits=loss_segments_splits)
                logger.update(loss_segments_type=loss_segments_type)
                logger.update(loss_beats=loss_beats)
                logger.update(loss_on_set=loss_on_set)

                logger.auto_log()

        if cfg.distributed:
            torch.distributed.barrier()

        # if batch_idx >= 50:
        #     break


def train(model, loader, optimizer, epoch, device, loggers, cfg):
    model.train()
    return __train_test(model, loader, optimizer, epoch, device, loggers=loggers, cfg=cfg)

def evaluate(model, loader, optimizer, epoch, device, loggers, cfg):
    model.eval()
    with torch.no_grad():
        return __train_test(model, loader, optimizer, epoch, device, loggers=loggers, cfg=cfg)




def draw_vertical_brace(fig, x, y0, y1, width=0.01, lw=2):

    verts = [
        (x, y0),
        (x - width, y0),
        (x - width, (y0 + y1) / 2),
        (x, (y0 + y1) / 2),
        (x - width, (y0 + y1) / 2),
        (x - width, y1),
        (x, y1),
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
    ]

    path = Path(verts, codes)
    patch = patches.PathPatch(
        path,
        transform=fig.transFigure,
        fill=False,
        lw=lw,
        color="black"
    )
    fig.add_artist(patch)

@torch.no_grad()
def project(model: PhraserModel, project_batch, config, device, logger):
    model.eval()
    embedding, segments_info, stems_info, beats_info, urls = project_batch

    embedding = embedding[:8, :]
    embedding = embedding.to(device)


    (logits_all_split, logits_element_silent, logits_segment_predict), ss_matrix, (beats_predict, on_set_predict) = model.forward_encoded(embedding)

    if not config.is_master:
        return

    pred_splits = F.sigmoid(logits_all_split).squeeze(1).cpu().detach().numpy()
    pred_element_silent = F.sigmoid(logits_element_silent).squeeze(1).cpu().detach().numpy()
    pred_segments = torch.softmax(logits_segment_predict, dim=2).cpu().detach().numpy()

    pread_beats = F.sigmoid(beats_predict).squeeze(2).cpu().detach().numpy()
    pread_onset = F.sigmoid(on_set_predict).squeeze(2).cpu().detach().numpy()

    max_length = segments_info['segments_splits'].shape[1]

    for idx in range(embedding.shape[0]):
        correct_segments = [Segment(start=start, end=end, label=label) for (label, start, end) in segments_info['segments'][idx]]
        
        fig = plt.figure(figsize=(24, 12))
        fig.suptitle(urls[idx].split('parsed/')[-1], fontsize=16)

        rows = 15
        axes = {}
        gs = gridspec.GridSpec(rows, 1, height_ratios=[1 for x in range(rows)])
        
        for row_idx in range(rows):
            axes[row_idx] = plt.subplot(gs[row_idx])

        # Set up segment visualization
        axes[0].set_xlim(0, max_length)
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel('Segments')
        axes[0].set_xticks([])


        axes[2].set_xlim(0, max_length)
        axes[2].set_ylim(0, 1)
        axes[2].set_ylabel('TO DO!')
        axes[2].set_xticks([])

        # Plot ground truth segments
        for segment in correct_segments:
            start, end, label = segment.start, segment.end, segment.label
            color = SECTION_COLORS[SECTION_LABELS.index(label)]
            start = int(start * STEPS_PER_SEC)
            end = int(end * STEPS_PER_SEC)
            if start > len(embedding[idx][0]):
                break
            end = min(len(embedding[idx][0]), end)
            axes[0].axvspan(start, end, color=color)
            axes[0].axvline(start, color='black', linewidth=1)
            if label not in ['start', 'end']:
                axes[0].text(
                    (end - start) / 2 + start, 0.5, label,
                    fontsize=12, weight='bold',
                    horizontalalignment='center',
                    verticalalignment='center', rotation=90
                )

        # Plot predicted functions
        axes[1].set_xlim(0, max_length)
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel('Pred Segments')
        axes[1].set_xticks([])
        for i, (label, color) in enumerate(zip(SECTION_LABELS, SECTION_COLORS)):
            if i >= pred_segments.shape[2]:
                break
            axes[1].plot(pred_segments[idx, :, i], linewidth=1, color=color)

        # Plot stem-wise visualizations
        stem_start = 3
        rows_per_stem = 3

        for stem_idx, stem_name in enumerate(['vocal', 'other', 'bass', 'drums']):
            # RMS plot
            # rms = librosa.feature.rms(
            #     y=audio[idx][stem_idx].detach().cpu().numpy(),
            #     frame_length=4096,
            #     hop_length=int(MUQ_SR / 1)
            # )[0]
            #
            # axes[stem_start + stem_idx * rows_per_stem].plot(rms, color='black', linewidth=1)
            # axes[stem_start + stem_idx * rows_per_stem].set_xlim(0, len(rms) - 1)
            # axes[stem_start + stem_idx * rows_per_stem].set_ylim(0, rms.max())
            axes[stem_start + stem_idx * rows_per_stem].set_ylabel('RMS')
            axes[stem_start + stem_idx * rows_per_stem].set_xticks([])

            # Ground truth splits and silence
            stem_splits = stems_info['stem_splits'][idx][stem_idx]
            stem_silent = stems_info['stem_silences'][idx][stem_idx]

            for i, val in enumerate(stem_silent):
                if val == 1:
                    axes[stem_start + stem_idx * rows_per_stem + 1].axvline(i, color='red', linewidth=1)
            for i, val in enumerate(stem_splits):
                if val == 1:
                    axes[stem_start + stem_idx * rows_per_stem + 1].axvline(i, color='black', linewidth=1)

            axes[stem_start + stem_idx * rows_per_stem + 1].set_xlim(0, max_length)
            axes[stem_start + stem_idx * rows_per_stem + 1].set_ylim(0, 1)
            axes[stem_start + stem_idx * rows_per_stem + 1].set_ylabel("org")
            axes[stem_start + stem_idx * rows_per_stem + 1].set_xticks([])

            # Predictions
            axes[stem_start + stem_idx * rows_per_stem + 2].plot(pred_element_silent[idx, stem_idx], color='red', linewidth=1)
            axes[stem_start + stem_idx * rows_per_stem + 2].plot(pred_splits[idx, stem_idx], color='black', linewidth=1)
            axes[stem_start + stem_idx * rows_per_stem + 2].set_xlim(0, max_length)
            axes[stem_start + stem_idx * rows_per_stem + 2].set_ylim(0, 1)
            axes[stem_start + stem_idx * rows_per_stem + 2].set_ylabel(f"pred")
            axes[stem_start + stem_idx * rows_per_stem + 2].set_xticks([])

            first_ax = axes[stem_start + stem_idx * rows_per_stem]
            last_ax  = axes[stem_start + stem_idx * rows_per_stem + 2]

            bbox_top = first_ax.get_position()
            bbox_bot = last_ax.get_position()

            y_top = bbox_top.y1
            y_bot = bbox_bot.y0

            x_brace = bbox_top.x0 - 0.02

            draw_vertical_brace(fig, x_brace, y_bot, y_top, width=0.01, lw=2)

            # Opis stema
            fig.text(
                x_brace - 0.03,
                (y_top + y_bot) / 2,
                stem_name,
                rotation=90,
                ha='center',
                va='center',
                fontsize=14,
                weight='bold'
            )

        # Save figure
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        plt.close()
        # Convert RGBA to RGB
        image = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,:3]
        logger.tensorboard.add_image(f'{logger.name}/project/{idx}', image, logger.curr_index, dataformats='HWC')

    beats_max_length = beats_info['beats'].shape[-1]
    beats_max_length = 3000
    for idx in range(embedding.shape[0]):

        fig = plt.figure(figsize=(24, 4))
        fig.suptitle(urls[idx].split('parsed/')[-1], fontsize=16)

        rows = 4
        axes = {}
        gs = gridspec.GridSpec(rows, 1, height_ratios=[1 for x in range(rows)])

        for row_idx in range(rows):
            axes[row_idx] = plt.subplot(gs[row_idx])



        axes[0].set_xlim(0, beats_max_length)
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("org")
        axes[0].set_xticks([])

        for i, val in enumerate(beats_info['beats'][idx][:beats_max_length]):
            if val == 1:
                axes[0].axvline(i, color='black', linewidth=1)

        for i, val in enumerate(beats_info['on_set'][idx][:beats_max_length]):
            if val == 1:
                axes[0].axvline(i, color='red', linewidth=1)

        axes[1].plot(pread_beats[idx][:beats_max_length], color='black', linewidth=1)
        axes[1].plot(pread_onset[idx][:beats_max_length], color='red', linewidth=1)
        axes[1].set_xlim(0, beats_max_length)
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel(f"pred")
        axes[1].set_xticks([])


        for i, val in enumerate(pread_beats[idx][:beats_max_length] > 0.5):
            if beats_info['beats'][idx][i] == val:
                axes[2].axvline(i, color='green', linewidth=1)
            else:
                axes[2].axvline(i, color='black', linewidth=1)


        axes[2].set_xlim(0, beats_max_length)
        axes[2].set_ylim(0, 1)
        axes[2].set_ylabel(f"pred beats")
        axes[2].set_xticks([])


        for i, val in enumerate(pread_onset[idx][:beats_max_length] > 0.5):
            if beats_info['on_set'][idx][i] == val:
                axes[3].axvline(i, color='green', linewidth=1)
            else:
                axes[3].axvline(i, color='black', linewidth=1)


        axes[3].set_xlim(0, beats_max_length)
        axes[3].set_ylim(0, 1)
        axes[3].set_ylabel(f"pred on sets")
        axes[3].set_xticks([])

        # Save figure
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        plt.close()
        # Convert RGBA to RGB
        image = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,:3]
        logger.tensorboard.add_image(f'{logger.name}/beats/{idx}', image, logger.curr_index, dataformats='HWC')
    logger.log(reset=False)

def _label_ce(logits, onehot):
    import os as _os
    import torch.nn.functional as _F
    _w = float(_os.environ.get("PHRASER_LABEL_W", "0.1"))
    _C = logits.shape[-1]
    _lg = logits.reshape(-1, _C)
    _oh = onehot.reshape(-1, _C)
    _m = _oh.sum(1) > 0
    if _m.sum() == 0:
        return _lg.sum() * 0.0
    _lg = _lg[_m]
    _tgt = _oh[_m].argmax(1)
    out = _F.cross_entropy(_lg, _tgt)
    # PHRASER_LABEL_FOCAL=1: add SongFormer-style softmax focal (models/SongFormer.py
    # SoftmaxFocalLoss): alpha_t on target class, (1-p_t)^gamma, added with weight 0.2.
    if _os.environ.get("PHRASER_LABEL_FOCAL", "0") == "1":
        _a = float(_os.environ.get("PHRASER_FOCAL_ALPHA", "0.25"))
        _g = float(_os.environ.get("PHRASER_FOCAL_GAMMA", "2.0"))
        _fw = float(_os.environ.get("PHRASER_FOCAL_W", "0.2"))
        _lp = _F.log_softmax(_lg, dim=-1)
        _p_t = _lp.exp().gather(1, _tgt[:, None]).squeeze(1).clamp(min=1e-8, max=1.0 - 1e-8)
        _focal = -(_a * (1.0 - _p_t) ** _g * _lp.gather(1, _tgt[:, None]).squeeze(1))
        out = out + _fw * _focal.mean()
    return out * _w
