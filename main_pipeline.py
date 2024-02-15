from pipeline_training import training_pipeline
from pipeline_evaluation import evaluation_pipeline
from dataset_spliter import split_dataset
import os


def main():
    training_opts = {
        "dataset_path":".\\dataset",  # could change
        "splitted_dataset_path":".\\splitted_dataset", # could change, but make sure there is no folder with same name before execution
        "num_classes":2, # can't change
        "epochs":20,  # could change
        "best_model_path":"best_model.pth"  # could change
    }
    # splitting data
    print("phase 1: splitting data...")
    labels_dir = os.path.join(training_opts["dataset_path"], "labels")
    images_dir = os.path.join(training_opts["dataset_path"], "images")
    splited_dataset_dir = training_opts["splitted_dataset_path"]
    # split_dataset(labels_dir, images_dir, splited_dataset_dir)

    # training
    print("phase 2: training...")
    training_pipeline(training_opts)

    eval_opts = {
        "num_classes":2,  # can't change
        "PR_precision":10,  # could change. This param means the number of intervals to use when drawing the PR-curve
        "iou_threshold":0.5, # could change
        "eval_result_root_path":".\\run",  # make sure that folder run exists
        "demo_img_path":".\\runs\\demo.png", # could change but make sure that the parent folder exists
        "PR_curve_path":".\\runs\\PR_curve.png", # could change but make sure that the parent folder exists
    }
    print("phase 3: evaluation...")
    mAP = evaluation_pipeline(training_opts["best_model_path"], training_opts["splitted_dataset_path"], eval_opts)
    print(mAP)
    return

if __name__ == "__main__":
    main()