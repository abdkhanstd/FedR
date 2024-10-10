import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.model2 import RoadSegmenter
from dataloaders.loader import get_company_dataloaders
from scriptss.config import Config
import concurrent.futures
import logging
import csv
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from threading import Barrier
from sklearn.metrics import confusion_matrix

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.backends.cudnn.enabled = False

import abdutils as abd
abd.ClearScreen()
abd.LookForKeys()

# Early stopping patience
PATIENCE = 40

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Model Training Logs")

os.makedirs(Config.WEIGHTS_PATH, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Synchronization barrier for synchronizing epoch completion across models
EPOCH_BARRIER = Barrier(parties=3)  # Assuming we have 3 companies

# Variable to control whether to continue training or start fresh
resume_training = True

# Function to load model weights if they exist
def load_best_model_if_exists(model, company_id, config):
    model_path = f"{config.WEIGHTS_PATH}/best_road_model_company_{company_id}.pth"
    if os.path.exists(model_path):
        print(f"Loading the best model for Company {company_id} from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"No pre-trained model found for Company {company_id}, starting from scratch.")

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, inputs, targets):
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        dice_loss = self.dice_loss(inputs, targets)
        bce_loss = self.bce_loss(inputs, targets)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

def calculate_kitti_metrics(preds, labels):
    preds = torch.sigmoid(preds) > 0.5
    preds = preds.view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()
    
    cm = confusion_matrix(labels, preds)
    
    tp = cm[1, 1]  # True Positive
    fp = cm[0, 1]  # False Positive
    fn = cm[1, 0]  # False Negative
    tn = cm[0, 0]  # True Negative
    
    pixel_accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'accuracy': pixel_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'fpr': fpr,
        'fnr': fnr
    }

def log_to_csv(log_path, data):
    try:
        with open(log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
    except Exception as e:
        print(f"Error logging data to CSV: {e}")

def aggregate_model_weights(company_models, weights):
    avg_model = {}
    total_weight = sum(weights)
    
    for key in company_models[0].keys():
        avg_model[key] = sum(weights[i] * company_models[i][key].cpu() for i in range(len(company_models))) / total_weight
    
    return avg_model

def calculate_weights_based_on_loss(val_losses):
    inverse_losses = [1.0 / loss for loss in val_losses]
    total = sum(inverse_losses)
    weights = [inv_loss / total for inv_loss in inverse_losses]
    return weights

def train_road_model(model, train_loader, val_loader, config, company_id, device, logger, round_num, logs_dir):
    torch.cuda.empty_cache()

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.ROAD_LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    criterion = CombinedLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    training_log = {"train_loss": [], "val_loss": [], "val_iou": [], "val_f1": []}

    for epoch in range(config.ROAD_EPOCHS):
        model.train()
        epoch_loss = 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.ROAD_EPOCHS} - Train (Company {company_id})"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            road_masks_pred = model(images)
            road_masks_pred_resized = F.interpolate(road_masks_pred, size=(224, 224), mode='bilinear', align_corners=False)

            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            loss = criterion(road_masks_pred_resized, masks)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Company {company_id} Round {round_num}, Epoch {epoch+1}, Train Loss: {avg_epoch_loss}")

        torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_metrics = {
                'accuracy': 0, 'precision': 0, 'recall': 0,
                'f1': 0, 'iou': 0, 'fpr': 0, 'fnr': 0
            }
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                road_masks_pred = model(images)
                road_masks_pred_resized = F.interpolate(road_masks_pred, size=(224, 224), mode='bilinear', align_corners=False)

                loss = criterion(road_masks_pred_resized, masks)
                val_loss += loss.item()

                batch_metrics = calculate_kitti_metrics(road_masks_pred_resized, masks)
                for key in val_metrics:
                    val_metrics[key] += batch_metrics[key]

            avg_val_loss = val_loss / len(val_loader)
            for key in val_metrics:
                val_metrics[key] /= len(val_loader)
            
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
                    
            log_data = [epoch+1, avg_epoch_loss, avg_val_loss] + list(val_metrics.values()) + [current_lr]
            log_to_csv(f'logs/Company_{company_id}.csv', log_data)
            logger.info(f"Company {company_id} Round {round_num}, Epoch {epoch+1}, Val Loss: {avg_val_loss}, Val Metrics: {val_metrics}")

            training_log["train_loss"].append(avg_epoch_loss)
            training_log["val_loss"].append(avg_val_loss)
            training_log["val_iou"].append(val_metrics['iou'])
            training_log["val_f1"].append(val_metrics['f1'])

            save_validation_samples(images.cpu(), road_masks_pred_resized.cpu(), masks.cpu(), epoch + 1, company_id, save_dir="ValSamples")

        EPOCH_BARRIER.wait()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Saving the best model at epoch {epoch+1} for Company {company_id}")
            torch.save(model.state_dict(), f"{config.WEIGHTS_PATH}/best_road_model_company_{company_id}.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} for Company {company_id}")
                break

    return training_log

def iterative_aggregation_round(models_per_company, company_ids, config, train_loader, val_loader, rounds=3, AggAfterRounds=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logs_dir = "logs"
    
    best_global_val_f1 = 0

    for r in range(rounds):
        print(f"===== Iteration {r+1}/{rounds} =====")
        all_company_models = []
        all_company_losses = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for company_id in company_ids:
                model = models_per_company[company_id]
                futures.append(executor.submit(parallel_train_road_model, company_id, model, train_loader[company_id], val_loader, config, device, r+1, logs_dir))

            for future in concurrent.futures.as_completed(futures):
                training_log = future.result()
                all_company_models.append(models_per_company[company_id].state_dict())

                avg_val_loss = np.mean(training_log["val_loss"])
                all_company_losses.append(avg_val_loss)

        if (r + 1) % AggAfterRounds == 0:
            weights = calculate_weights_based_on_loss(all_company_losses)

            global_weights = aggregate_model_weights(all_company_models, weights)
            
            torch.cuda.empty_cache()

            for company_id in company_ids:
                models_per_company[company_id].load_state_dict(global_weights)

            global_model = models_per_company[company_ids[0]]
            avg_val_loss, val_metrics = validate_global_model(global_model, val_loader, device, r + 1, logs_dir)

            if val_metrics['f1'] > best_global_val_f1:
                best_global_val_f1 = val_metrics['f1']
                print(f"Saving the best global model at round {r+1}")
                torch.save(global_model.state_dict(), os.path.join(Config.WEIGHTS_PATH, "best_global_model.pth"))

            # Log global model metrics
            log_data = [r+1, avg_val_loss] + list(val_metrics.values())
            log_to_csv('logs/global_model_metrics.csv', log_data)

    print("Federated Learning completed.")
    return models_per_company

def validate_global_model(global_model, val_loader, device, round_num, logs_dir):
    global_model.to(device)
    global_model.eval()

    val_loss = 0
    val_metrics = {
        'accuracy': 0, 'precision': 0, 'recall': 0,
        'f1': 0, 'iou': 0, 'fpr': 0, 'fnr': 0
    }
    criterion = CombinedLoss()

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Validating Global Model - Round {round_num}"):
            images, masks = images.to(device), masks.to(device)
            road_masks_pred = global_model(images)
            road_masks_pred_resized = F.interpolate(road_masks_pred, size=(224, 224), mode='bilinear', align_corners=False)

            loss = criterion(road_masks_pred_resized, masks)
            val_loss += loss.item()

            batch_metrics = calculate_kitti_metrics(road_masks_pred_resized, masks)
            for key in val_metrics:
                val_metrics[key] += batch_metrics[key]

            save_validation_samples(images.cpu(), road_masks_pred_resized.cpu(), masks.cpu(), round_num, "Global_Model", save_dir="GlobalValSamples")

    avg_val_loss = val_loss / len(val_loader)
    for key in val_metrics:
        val_metrics[key] /= len(val_loader)

    logger.info(f"Global Model - Round {round_num}, Val Loss: {avg_val_loss}, Val Metrics: {val_metrics}")

    log_data = [round_num, avg_val_loss] + list(val_metrics.values())

    global_model_save_path = os.path.join(Config.WEIGHTS_PATH, f"best_global_model_round_{round_num}.pth")
    torch.save(global_model.state_dict(), global_model_save_path)

    return avg_val_loss, val_metrics

def save_validation_samples(images, preds, masks, epoch, company_id, save_dir="ValSamples", num_samples=10):
    os.makedirs(save_dir, exist_ok=True)
    company_folder = os.path.join(save_dir, f"Company_{company_id}")
    os.makedirs(company_folder, exist_ok=True)

    indices = random.sample(range(len(images)), min(num_samples, len(images)))
    for i, idx in enumerate(indices):
        image = images[idx].cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)

        pred = preds[idx].cpu().numpy().squeeze() * 255
        if pred.ndim == 2:
            pred = np.stack([pred] * 3, axis=-1)
        pred = pred.astype(np.uint8)

        mask = masks[idx].cpu().numpy().squeeze() * 255
        if mask.ndim == 2:
            mask = np.stack([mask] * 3, axis=-1)
        mask = mask.astype(np.uint8)

        combined_image = np.concatenate((image, mask, pred), axis=1)

        cv2.imwrite(os.path.join(company_folder, f"epoch_{epoch}_sample_{i}_combined.png"), combined_image)

def parallel_train_road_model(company_id, model, train_loader, val_loader, config, device, round_num, logs_dir):
    print(f"Training model for Company {company_id}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for Company {company_id}")
        model = nn.DataParallel(model)

    return train_road_model(model, train_loader, val_loader, config, company_id, device, logger, round_num, logs_dir)

def validate_global_model(global_model, val_loader, device, round_num, logs_dir):
    global_model.to(device)
    global_model.eval()

    val_loss = 0
    val_metrics = {
        'accuracy': 0, 'precision': 0, 'recall': 0,
        'f1': 0, 'iou': 0, 'fpr': 0, 'fnr': 0
    }
    criterion = CombinedLoss()

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Validating Global Model - Round {round_num}"):
            images, masks = images.to(device), masks.to(device)
            road_masks_pred = global_model(images)
            road_masks_pred_resized = F.interpolate(road_masks_pred, size=(224, 224), mode='bilinear', align_corners=False)

            loss = criterion(road_masks_pred_resized, masks)
            val_loss += loss.item()

            batch_metrics = calculate_kitti_metrics(road_masks_pred_resized, masks)
            for key in val_metrics:
                val_metrics[key] += batch_metrics[key]

            save_validation_samples(images.cpu(), road_masks_pred_resized.cpu(), masks.cpu(), round_num, "Global_Model", save_dir="GlobalValSamples")

    avg_val_loss = val_loss / len(val_loader)
    for key in val_metrics:
        val_metrics[key] /= len(val_loader)

    logger.info(f"Global Model - Round {round_num}, Val Loss: {avg_val_loss}, Val Metrics: {val_metrics}")

    log_data = [round_num, avg_val_loss] + list(val_metrics.values())

    global_model_save_path = os.path.join(Config.WEIGHTS_PATH, f"best_global_model_round_{round_num}.pth")
    torch.save(global_model.state_dict(), global_model_save_path)

    return avg_val_loss, val_metrics

# Main function
if __name__ == "__main__":
    COMPANY_IDS = [1, 2, 3]
    config = Config()

    models_per_company = {company_id: RoadSegmenter() for company_id in COMPANY_IDS}
    train_loader, val_loader = {}, {}

    for company_id in COMPANY_IDS:
        company_csv = f"dataloaders/csv_files/company_{company_id}_dataset.csv"
        train_loader[company_id], val_loader = get_company_dataloaders(company_csv, Config.ROAD_BATCH_SIZE)

        # Load the best model weights if they exist and resume training if needed
        if resume_training:
            load_best_model_if_exists(models_per_company[company_id], company_id, config)

    # Initialize the CSV files with headers
    company_headers = ['Epoch', 'Train Loss', 'Val Loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'IoU', 'FPR', 'FNR', 'Learning Rate']
    global_headers = ['Round', 'Val Loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'IoU', 'FPR', 'FNR']

    for company_id in COMPANY_IDS:
        log_to_csv(f'logs/Company_{company_id}.csv', company_headers)
    log_to_csv('logs/global_model_metrics.csv', global_headers)

    final_models = iterative_aggregation_round(models_per_company, COMPANY_IDS, config, train_loader, val_loader, rounds=100)

    print("Training completed. Final models saved.")
