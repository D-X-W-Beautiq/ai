import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Add a abs path for importing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gc
from torch.utils import data
import shutil
import torch.nn as nn
import numpy as np
from torchvision import models
from tensorboardX import SummaryWriter
from utils import mkdir, resume_checkpoint, fix_seed, CB_loss
from logger import setup_logger
from tool.data_loader import CustomDataset_class, CustomDataset_regress
from model import Model
import argparse

fix_seed(523)
git_name = os.popen("git branch --show-current").readlines()[0].rstrip()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="none",
        type=str,
    )

    parser.add_argument("--equ", type=int, default=[1], choices=[1, 2, 3], nargs="+")

    parser.add_argument("--stop_early", type=int, default=50)

    parser.add_argument(
        "--mode",
        default="both",  # 기본값을 both로 변경
        choices=["regression", "class", "both"],
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        default=f"checkpoint/{git_name}",
        type=str,
    )

    parser.add_argument(
        "--epoch",
        default=100,  # 200 -> 100으로 변경
        type=int,
    )

    parser.add_argument(
        "--res",
        default=256,
        type=int,
    )

    parser.add_argument(
        "--gamma",
        default=2,
        type=int,
    )

    parser.add_argument(
        "--load_epoch",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--lr",
        default=0.005,
        type=float,
    )

    parser.add_argument(
        "--batch_size",
        default=16,  # 32 -> 16으로 변경 (메모리 절약)
        type=int,
    )
    
    parser.add_argument(
        "--num_workers",
        default=4,  # 8 -> 4로 변경 (안정성)
        type=int,
    )

    parser.add_argument("--reset", action="store_true")

    args = parser.parse_args()

    return args


def save_task_result_csv(task_name, train_data_size, val_data_size, final_train_loss, final_metrics, mode_type, save_path="/home/work/conf/NIA/results"):
    """각 태스크 완료 즉시 CSV에 저장"""
    
    os.makedirs(save_path, exist_ok=True)
    csv_file = os.path.join(save_path, f"{mode_type}_results.csv")
    
    # 헤더 생성 (파일이 없을 경우)
    if not os.path.exists(csv_file):
        if mode_type == "class":
            header = "Task,Train_Data_Size,Val_Data_Size,Train_Loss,Accuracy,Precision,Recall,F1_Score\n"
        else:
            header = "Task,Train_Data_Size,Val_Data_Size,Train_Loss,MAE,MSE,RMSE,R2_Score\n"
        
        with open(csv_file, 'w') as f:
            f.write(header)
    
    # 데이터 추가
    with open(csv_file, 'a') as f:
        if mode_type == "class":
            f.write(f"{task_name},{train_data_size},{val_data_size},{final_train_loss:.4f},"
                   f"{final_metrics.get('accuracy', 0.0):.4f},{final_metrics.get('precision', 0.0):.4f},"
                   f"{final_metrics.get('recall', 0.0):.4f},{final_metrics.get('f1_score', 0.0):.4f}\n")
        else:
            f.write(f"{task_name},{train_data_size},{val_data_size},{final_train_loss:.4f},"
                   f"{final_metrics.get('mae', 0.0):.4f},{final_metrics.get('mse', 0.0):.4f},"
                   f"{final_metrics.get('rmse', 0.0):.4f},{final_metrics.get('r2_score', 0.0):.4f}\n")
    
    print(f"Task {task_name} result saved to CSV")


def csv_to_excel_final(mode_type, save_path="/home/work/conf/NIA/results"):
    """모든 태스크 완료 후 CSV를 엑셀로 변환"""
    
    csv_file = os.path.join(save_path, f"{mode_type}_results.csv")
    excel_file = os.path.join(save_path, f"{mode_type}_results.xlsx")
    
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            df.to_excel(excel_file, index=False)
            print(f"Final results converted to Excel: {excel_file}")
        except Exception as e:
            print(f"Excel conversion failed: {e}")


def calculate_classification_metrics(y_true, y_pred):
    """분류 성능 지표 계산"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def calculate_regression_metrics(y_true, y_pred):
    """회귀 성능 지표 계산"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2_score': r2
    }


def cleanup_memory():
    """메모리 정리"""
    torch.cuda.empty_cache()
    gc.collect()


def run_single_mode(args, mode_type):
    """단일 모드 실행 (분류 또는 회귀)"""
    
    # 임시로 mode 변경
    original_mode = args.mode
    args.mode = mode_type
    
    print(f"\n========== Starting {mode_type.upper()} Mode ==========")
    print(f"Epochs: {args.epoch}, Batch Size: {args.batch_size}, Workers: {args.num_workers}")
    
    check_path = os.path.join(args.output_dir, args.mode, args.name)
    log_path = os.path.join("tensorboard", git_name, args.mode, args.name)

    model_num_class = (
        {"dryness": 5, "pigmentation": 6, "pore": 6, "sagging": 6, "wrinkle": 7}
        if args.mode == "class"
        else {
            "pigmentation": 1,
            "moisture": 1,
            "elasticity_R2": 1,
            "wrinkle_Ra": 1,
            "pore": 1,
        }
    )
    
    pass_list = list()
    args.best_loss = {item: np.inf for item in model_num_class}
    args.load_epoch = {item: 0 for item in model_num_class}

    model_list = {
        key: models.resnet50(weights=models.ResNet50_Weights.DEFAULT, args=args)
        for key, _ in model_num_class.items()
    }

    model_path = os.path.join(check_path, "save_model")
        
    for key, model in model_list.items(): 
        model.fc = nn.Linear(model.fc.in_features, model_num_class[key], bias = True)
        model_list.update({key: model})
        
    args.save_img = os.path.join(check_path, "save_img")
    args.pred_path = os.path.join(check_path, "prediction")

    if args.reset:
        print(f"Reseting......{check_path}")
        if os.path.isdir(check_path):
            shutil.rmtree(check_path)
        if os.path.isdir(log_path):
            shutil.rmtree(log_path)

    if os.path.isdir(model_path):
        for path in os.listdir(model_path):
            dig_path = os.path.join(model_path, path)
            if os.path.isfile(os.path.join(dig_path, "state_dict.bin")):
                print(f"Resuming......{dig_path}")
                model_list[path] = resume_checkpoint(
                    args,
                    model_list[path],
                    os.path.join(model_path, f"{path}", "state_dict.bin"),
                    path, 
                )
                if os.path.isdir(os.path.join(dig_path, "done")):
                    print(f"Passing......{dig_path}")
                    pass_list.append(path)

    mkdir(model_path)
    mkdir(log_path)
    writer = SummaryWriter(log_path)

    logger = setup_logger(
        args.name + args.mode, os.path.join(check_path, "log", "train")
    )
    logger.info(args)
    logger.info("Command Line: " + " ".join(sys.argv))

    dataset = (
        CustomDataset_class(args, logger, "train")
        if args.mode == "class"
        else CustomDataset_regress(args, logger)
    )

    print(f"Tasks to process: {list(model_num_class.keys())}")
    
    for task_idx, key in enumerate(model_list.keys()):
        if key in pass_list:
            print(f"Skipping {key} (already completed)")
            continue

        print(f"\n---------- Training {key} ({task_idx+1}/{len(model_list)}) ----------")
        
        try:
            model = model_list[key].cuda()

            trainset, grade_num = dataset.load_dataset("train", key)
            trainset_loader = data.DataLoader(
                dataset=trainset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                pin_memory=True  # GPU 최적화
            )

            valset, _ = dataset.load_dataset("valid", key)
            valset_loader = data.DataLoader(
                dataset=valset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                pin_memory=True
            )

            print(f"Train Data Size: {len(trainset)}, Validation Data Size: {len(valset)}")

            resnet_model = Model(
                args,
                model,
                trainset_loader,
                valset_loader,
                logger,
                check_path,
                model_num_class,
                writer,
                key,
                grade_num
            )

            # 학습 실행
            final_train_loss = 0.0
            final_val_metrics = {}
            
            for epoch in range(args.load_epoch[key], args.epoch):
                if epoch % 10 == 0:  # 10 에포크마다 진행상황 출력
                    print(f"Epoch {epoch+1}/{args.epoch} - {key}")
                
                resnet_model.update_e(epoch + 1) if args.load_epoch else None

                # 학습
                train_loss = resnet_model.train()
                
                # 검증
                val_metrics = resnet_model.valid()

                # 마지막 값들 저장
                if train_loss is not None:
                    final_train_loss = train_loss
                if val_metrics is not None:
                    final_val_metrics = val_metrics

                resnet_model.update_e(epoch + 1)
                resnet_model.reset_log()

                # 조기 종료 확인
                if resnet_model.stop_early():
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                # 메모리 정리 (매 5 에포크마다)
                if epoch % 5 == 0:
                    cleanup_memory()

            # 태스크 완료 즉시 CSV 저장
            save_task_result_csv(
                task_name=key,
                train_data_size=len(trainset),
                val_data_size=len(valset),
                final_train_loss=final_train_loss,
                final_metrics=final_val_metrics,
                mode_type=args.mode
            )

            # 메모리 정리
            del trainset_loader, valset_loader, trainset, valset, resnet_model
            cleanup_memory()
            
            print(f"Task {key} completed successfully")

        except Exception as e:
            print(f"Error in task {key}: {e}")
            cleanup_memory()
            continue

    # 모든 태스크 완료 후 엑셀 변환
    csv_to_excel_final(args.mode)
    
    # mode 원복
    args.mode = original_mode
    
    print(f"{mode_type.upper()} mode completed!")


def main(args):
    """메인 함수 - both 모드 지원"""
    
    print("Configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Epochs: {args.epoch}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Num Workers: {args.num_workers}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    try:
        if args.mode == "both":
            # 분류 먼저 실행
            print("\n" + "="*80)
            print("STARTING CLASSIFICATION")
            print("="*80)
            run_single_mode(args, "class")
            
            print("\n" + "="*80)
            print("CLASSIFICATION COMPLETED! STARTING REGRESSION")
            print("="*80)
            
            # 메모리 완전 정리
            cleanup_memory()
            
            # 회귀 실행
            run_single_mode(args, "regression")
            
            print("\n" + "="*80)
            print("ALL TRAINING COMPLETED!")
            print("Results saved in /home/work/conf/NIA/results/")
            print("="*80)
            
        else:
            # 단일 모드 실행
            run_single_mode(args, args.mode)
    
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print("Partial results saved in CSV files")
    
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Partial results may be saved in CSV files")
    
    finally:
        cleanup_memory()
        print("Process finished")


if __name__ == "__main__":
    args = parse_args()
    main(args)