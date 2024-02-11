import os
import sys
from pprint import pprint

import cv2
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_Weights, FastRCNNPredictor)
from tqdm import tqdm

from class_cardataset import CarDataset
from transforms_part import get_transforms
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers

# import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_object_detection_model(num_classes):
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=0)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT, trainable_backbone_layers=0)
    in_features =  model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_trained_model(num_classes, weights_file_path):
    model = get_object_detection_model(num_classes)
    if weights_file_path.endswith(".pth") and os.path.exists(weights_file_path):
        model.load_state_dict(torch.load(weights_file_path))
    else:
        print("wrong weights file")
    return model


def val_forward(model, images, targets):
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            It returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).
    """
    model.eval()

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError(
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}."
                )

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    model.rpn.training=True
    #model.roi_heads.training=True


    #####proposals, proposal_losses = model.rpn(images, features, targets)
    features_rpn = list(features.values())
    objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
    anchors = model.rpn.anchor_generator(images, features_rpn)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    proposals, scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    proposal_losses = {}
    assert targets is not None
    labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
    regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
    loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
        objectness, pred_bbox_deltas, labels, regression_targets
    )
    proposal_losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }

    #####detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
    image_shapes = images.image_sizes
    proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    result: List[Dict[str, torch.Tensor]] = []
    detector_losses = {}
    loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
    num_images = len(boxes)
    for i in range(num_images):
        result.append(
            {
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            }
        )
    detections = result
    detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
    model.rpn.training=False
    model.roi_heads.training=False
    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses, detections

def validate_for_one_epoch(model, val_data_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, targets in val_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # pprint(images)
            # pprint(targets)
            loss_dict, _ = val_forward(model, images, targets)
            # pprint(loss_dict)

            losses = sum(loss for loss in loss_dict.values())

            total_loss += losses.item()

    return total_loss / len(val_data_loader)


def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0

    train_bar = tqdm(data_loader, file=sys.stdout, ncols=100)
    for images, targets in train_bar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        # print("train step")
        # pprint(loss_dict)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)


def train_process(model, train_data_loader, val_data_loader, optimizer, training_opts):
    # training model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    best_loss = float('inf')
    num_epochs = training_opts["epochs"]
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, train_data_loader, optimizer, device)
        val_loss = validate_for_one_epoch(model, val_data_loader, device)
        print(f"Epoch: {epoch}, Training Loss: {loss}, Validation Loss: {val_loss}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), training_opts["best_model_path"])

def get_optimizer(model, training_hyps = None):
    params = [p for p in model.parameters() if p.requires_grad]
    if training_hyps== None:
        return torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    

def training_pipeline(training_opts):
    ''' 
    return a model trained to passs to the following evaluation stage
    '''
    model = get_object_detection_model(training_opts["num_classes"])
    transforms_to_image = get_transforms()
    train_dataset = CarDataset(training_opts["splitted_dataset_path"], transforms_to_image)   # from transforms import get_transforms
    val_dataset = CarDataset(training_opts["splitted_dataset_path"], transforms_to_image, mode="val")
    train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))
    val_data_loader = DataLoader(val_dataset, batch_size=4, collate_fn=lambda batch: tuple(zip(*batch)))

    train_process(model, train_data_loader, val_data_loader, get_optimizer(model), training_opts)

    return

    # return get_trained_model(2, training_opts["best_model_path"])
