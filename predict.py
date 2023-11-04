import numpy as np
import torch
from model import UNet
from torch.utils.data import DataLoader
from dataset import CustomDataset

def preprocess(N, test_images):
    batch_size = 16
    test_dataset = CustomDataset(N, test_images, train = False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def flatten_predicted_masks(N, predicted_masks):
    flattened_masks = []
    for i in range(N):
        segmentation_mask = predicted_masks[i].cpu()
        flattened = torch.argmax(segmentation_mask, dim=0)
        flattened = flattened.view(-1)
        flattened = flattened.numpy()
        flattened_masks.append(flattened)
    flattened_masks = np.array(flattened_masks, dtype=np.int32)
    return flattened_masks

def predict_classes(N, predicted_masks):
    predicted_classes = []
    for i in range(N):
        non_background = (predicted_masks[i] != 10)
        unique_values, counts = np.unique(predicted_masks[i][non_background], return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        classes = unique_values[sorted_indices][:2]
        classes = sorted(classes.tolist())
        if len(classes) < 2:
            classes = [classes[0], classes[0]]
        predicted_classes.append(classes)
    predicted_classes = np.array(predicted_classes, dtype=np.int32)
    return predicted_classes

def predict_boxes(N, predicted_masks, predicted_classes):
    predicted_bboxes = []
    for i in range(N):
        mask = predicted_masks[i].reshape(64, 64)
        classes = predicted_classes[i]
        a = classes[0]
        b = classes[1]
        row_a, col_a = np.where(mask == a)
        row_b, col_b = np.where(mask == b)
        median_col_a = np.mean(col_a)
        median_row_a = np.mean(row_a)
        median_col_b = np.mean(col_b)
        median_row_b = np.mean(row_b)
        a_x_min = median_col_a - 14
        a_x_max = median_col_a + 14
        a_y_min = median_row_a - 14
        a_y_max = median_row_a + 14
        b_x_min = median_col_b - 14
        b_x_max = median_col_b + 14
        b_y_min = median_row_b - 14
        b_y_max = median_row_b + 14
        a_bbox = [a_y_min, a_x_min, a_y_max, a_x_max]
        b_bbox = [b_y_min, b_x_min, b_y_max, b_x_max]
        predicted_bboxes.append([a_bbox, b_bbox])
    predicted_bboxes = np.array(predicted_bboxes, dtype=np.float64)
    return predicted_bboxes

def detect_and_segment(images):
    """
    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """
    # empty cuda's cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # extract test dataset's size
    N = images.shape[0]

    # batch size
    batch_size = 16

    # load trained model
    model = UNet()
    checkpoint = torch.load(
        'checkpoints.pth', 
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()

    # preprocess images
    test_loader = preprocess(N, images)

    # empty arrays
    pred_seg = np.empty((0, 4096), dtype=np.int32)
    pred_class = np.empty((0, 2), dtype=np.int32)
    pred_bboxes = np.empty((0, 2, 4), dtype=np.float64)

    # generate predictions
    for batch in test_loader:
        test_images = batch['image'].cuda() if torch.cuda.is_available() else batch['image']
        predicted_masks = model(test_images)
        if predicted_masks.shape[0] != batch_size:
            batch_size = predicted_masks.shape[0]
        predicted_masks = flatten_predicted_masks(batch_size, predicted_masks)
        predicted_classes = predict_classes(batch_size, predicted_masks)
        predicted_bboxes = predict_boxes(batch_size, predicted_masks, predicted_classes)
        pred_seg = np.vstack((pred_seg, predicted_masks))
        pred_class = np.vstack((pred_class, predicted_classes))
        pred_bboxes = np.vstack((pred_bboxes, predicted_bboxes))
    return pred_class, pred_bboxes, pred_seg