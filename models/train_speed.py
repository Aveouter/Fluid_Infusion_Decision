import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader


def run_epoch(model, data_loader, optimizer, criterion_classification, criterion_regression, device, args, is_train=True, epoch=0, writer=None):
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    total_regression_loss = 0.0

    all_preds = []
    all_labels = []

    preview_preds = []
    preview_labels = []

    with torch.set_grad_enabled(is_train):
        for i, (time_series_data, baseline_data, label_data) in enumerate(data_loader):
            time_series_data, label_data, baseline_data = time_series_data.to(device), label_data.to(device), baseline_data.to(device)

            if args.model == 'lstm':
                input_T = time_series_data
                input_B = baseline_data
            else:
                input_T = torch.cat((time_series_data[:, :-args.pred_length, :], label_data[:, :-args.pred_length, :]), dim=-1)
                input_B = baseline_data

            if is_train:
                optimizer.zero_grad()

            flow_output, type_out = model(input_T, input_B)

            if args.model == 'lstm':
                labels_type = label_data[:, :, :3].to(device)
                labels_flow = label_data[:, :, 3:].to(device)
                type_out = type_out.view(-1, type_out.size(2))
                flow_output = flow_output.view(-1, flow_output.size(2))
                labels_type = labels_type.view(-1, labels_type.size(2))
                labels_flow = labels_flow.view(-1, labels_flow.size(2))
            else:
                labels_type = label_data[:, -args.pred_length:, :3].squeeze(1).to(device)
                labels_flow = label_data[:, -args.pred_length:, 3:].squeeze(1).to(device)

            type_out = type_out.flatten()
            labels_type = labels_type.float().flatten()

            loss_type = criterion_classification(type_out, labels_type)
            loss_flow = criterion_regression(flow_output, labels_flow.float() / 1000.0)
            loss = loss_type + 100* loss_flow

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            predicted = (torch.sigmoid(type_out) > 0.3).long()
            correct_preds += (predicted == labels_type).sum().item()
            total_preds += labels_type.numel()
            total_regression_loss += loss_flow.item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_type.cpu().numpy())

            if not is_train and len(preview_preds) < 5:
                preview_preds.append(predicted[:10].cpu().numpy())
                preview_labels.append(labels_type[:10].cpu().numpy())

    epoch_loss = running_loss / len(data_loader)
    epoch_acc = correct_preds / total_preds
    avg_regression_loss = total_regression_loss / len(data_loader)

    if not is_train:
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        try:
            auc_score = roc_auc_score(all_labels, all_preds, average='macro')
        except ValueError:
            auc_score = 0.0
    else:
        f1 = precision = recall = auc_score = 0.0

    if writer:
        prefix = 'train' if is_train else 'val'
        writer.add_scalar(f'{prefix}/Loss_total', epoch_loss, epoch)
        writer.add_scalar(f'{prefix}/Accuracy', epoch_acc, epoch)
        writer.add_scalar(f'{prefix}/Loss_regression', avg_regression_loss, epoch)
        if not is_train:
            writer.add_scalar(f'{prefix}/F1_score', f1, epoch)
            writer.add_scalar(f'{prefix}/Precision', precision, epoch)
            writer.add_scalar(f'{prefix}/Recall', recall, epoch)
            writer.add_scalar(f'{prefix}/AUC', auc_score, epoch)

    return epoch_loss, epoch_acc, avg_regression_loss, f1, precision, recall, auc_score, preview_preds, preview_labels

timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'training{timestamp}.log')
    ]
)

def train(model, train_loader, val_loader, args):
    device = args.device
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion_classification = nn.BCEWithLogitsLoss()
    criterion_regression = nn.HuberLoss(delta=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    best_val_loss = float('inf')
    patience = 100
    epochs_without_improvement = 0

    os.makedirs(args.output_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_path, 'logs'))

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        logging.info(f"Epoch {epoch + 1}/{args.epochs} started at {time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}")

        train_loss, train_acc, _, _, _, _, _, _, _ = run_epoch(
            model, train_loader, optimizer, criterion_classification,
            criterion_regression, device, args, is_train=True, epoch=epoch, writer=writer)

        val_loss, val_acc, avg_reg_loss, f1, precision, recall, auc, preds, labels = run_epoch(
            model, val_loader, optimizer, criterion_classification,
            criterion_regression, device, args, is_train=False, epoch=epoch, writer=writer)

        scheduler.step()

        logging.info(f"Epoch [{epoch + 1}/{args.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        logging.info(f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")
        logging.info(f"Avg Regression Loss: {avg_reg_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        for i in range(min(len(preds), 5)):
            logging.info(f"Preview Sample {i + 1}: Pred: {preds[i]}, Label: {labels[i]}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_path, 'best_model.pt'))
            logging.info(f"Model improved at epoch {epoch+1}, saved.")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            logging.info(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience:
            logging.info("Early stopping triggered.")
            break

    writer.close()
