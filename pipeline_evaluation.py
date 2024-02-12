import numpy as np
import cv2
from PIL import Image, ImageDraw
from pprint import pprint
import os
import matplotlib.pyplot as plt

import torch 
# import torchvision
from torchvision.ops import box_iou
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
from sklearn.metrics import auc

from pipeline_training import get_trained_model
from transforms_part import get_transforms
from class_cardataset import CarDataset

def inference_detection(weighs_file_path, image_path, num_classes=2, threshold = 0.5):
    model = get_trained_model(num_classes=num_classes, weights_file_path=weighs_file_path)

    img = cv2.imread(image_path)

    img_tensor = get_transforms()(img)

    model.eval()
    with torch.no_grad():
        predictions = model(img_tensor.unsqueeze(0))[0]
    # pprint(predictions)
    img_pillow = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img_pillow)
    for i in range(len(predictions["boxes"])):
        score = predictions["scores"][i].item()
        if score > threshold:
            box = predictions["boxes"][i].cpu().numpy()
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
    return img_pillow


def treatment_boxes_for_one_image(pred_boxes, true_boxes, scores, iou_threshold=0.5):
    result_table_for_one_image = []  # to improve at the data structure level

    for i, pred_box in enumerate(pred_boxes): 
        for true_box in true_boxes:
            detected_box = False
            if box_iou(pred_box.unsqueeze(0), true_box.unsqueeze(0)) > iou_threshold:
                result_table_for_one_image.append([scores[i], 1])
                detected_box = True
        if detected_box == False:
            result_table_for_one_image.append([scores[i], 0])
    return result_table_for_one_image


# from class_cardataset import CarDataset
# from torch.utils.data import DataLoader
# from sklearn.metrics import auc

def evaluation_pipeline(model_weights_file_path, dataset_path, eval_opts):
    ''' 
    given a model weights path and a dataset path, output a test(evaluation) result using mAP metric
    attention: there must be a subfolder named "test" in the directory of dataset_path/images and so is the dataset_path/labels
    '''
    # get model 
    model = get_trained_model(num_classes=eval_opts["num_classes"], weights_file_path=model_weights_file_path)
    # define device 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    transforms_to_image = get_transforms()
    test_dataset = CarDataset(dataset_path, transforms_to_image, "test")
    test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=lambda batch: tuple(zip(*batch)))

    model.eval()
    model.to(device)
    with torch.no_grad():
        result_stat_table = []
        for images, targets in test_data_loader:
            images = list(img.to(device) for img in images)
            predictions = model(images)[0] # knowing that the returned object is a list

            result_table_for_one_image = treatment_boxes_for_one_image(predictions["boxes"], targets[0]["boxes"].to(device), predictions["scores"], eval_opts["iou_threshold"])
            result_stat_table += result_table_for_one_image

        result_stat_table = np.array(torch.tensor(result_stat_table, device="cpu"))
        order = np.argsort(-result_stat_table[:,0], axis=0)  # sorting by descending the score of each prediction
        result_stat_table = result_stat_table[order]

        # calculate precision list, recall list
        precision_list = []
        recall_list = []
        for threshold in np.linspace(0, 1, eval_opts["PR_precision"], endpoint=False):
            search_index = np.argwhere(result_stat_table[:,0] > threshold)
            precision = np.squeeze(result_stat_table[search_index])[:,1].sum() / len(search_index)
            recall = np.squeeze(result_stat_table[search_index])[:,1].sum() / len(result_stat_table)
            precision_list.append(precision)
            recall_list.append(recall)

    # draw a PR-lines and save as a image
    plt.figure()
    plt.plot(precision_list, recall_list, marker = "o", linestyle='-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(eval_opts["PR_curve_path"])
    
    # draw a test image with bouding box
    a_test_img_name = os.listdir(os.path.join(dataset_path,"images","test"))[0]
    a_test_img_path = os.path.join(dataset_path,"images","test",a_test_img_name)
    img_pillow = inference_detection(model_weights_file_path, a_test_img_path, eval_opts["num_classes"], eval_opts["iou_threshold"])
    img_pillow.save(eval_opts["demo_img_path"])

    return auc(precision_list, recall_list)