#!/usr/bin/env python3
import os
import sys
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score, cohen_kappa_score, roc_curve, auc
from skimage.feature import hog
from sklearn.tree import DecisionTreeClassifier

sys.path.append(str(Path(__file__).parent / "src"))
from utility.hair_util import hair_mask, hair_coverage, remove_hair


def calculate_fleiss_kappa(votes, n_annotators=5):
    N = len(votes)
    P_i = []
    for k in votes:
        no_votes = n_annotators - k
        agree_pairs = (k * (k - 1) + no_votes * (no_votes - 1)) / 2.0
        total_pairs = (n_annotators * (n_annotators - 1)) / 2.0
        P_i.append(agree_pairs / total_pairs)
    P_o = np.mean(P_i)
    total_yes_votes = np.sum(votes)
    total_votes = N * n_annotators
    p_yes = total_yes_votes / total_votes
    p_no = 1.0 - p_yes
    P_e = p_yes**2 + p_no**2
    kappa = (P_o - P_e) / (1.0 - P_e)
    return P_o, P_e, kappa, p_yes


def run_analysis(features_path, annotations_path, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("result/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running agreement analysis on manual annotations...")
    
    df_annot = pd.read_csv(annotations_path)
    hair_votes = df_annot["hair"].values
    P_o, P_e, kappa, p_yes = calculate_fleiss_kappa(hair_votes, n_annotators=5)
    pen_votes = df_annot["pen"].values
    pen_prev = np.mean(pen_votes)
    
    agreement_summary = f"""Manual Hair Fleiss' Kappa: {kappa:.4f} (Observed: {P_o*100:.2f}%, Expected: {P_e*100:.2f}%, Prevalence: {p_yes*100:.2f}%)
Manual Penmarks Consensus Prevalence: {pen_prev*100:.2f}%
"""
    with open(reports_dir / "agreement_summary.txt", "w") as f:
        f.write(agreement_summary)
        
    print("Agreement analysis complete. Saved to result/reports/agreement_summary.txt")
    
    # Feature correlation matrix
    if features_path.exists():
        print("Generating clinical visual features correlation matrix...")
        df_feat = pd.read_csv(features_path)
        feature_cols = [
            "mask_area", "asymmetry", "border_compactness", "border_radial",
            "lab_colour_std", "bleeding_likelihood", "mean_h", "mean_s", "mean_v",
            "std_h", "std_s", "std_v", "hue_entropy", "melanoma_colour_count",
            "hair_coverage", "clean_lab_colour_std"
        ]
        cols_present = [c for c in feature_cols if c in df_feat.columns]
        if cols_present:
            corr = df_feat[cols_present].corr()
            plt.figure(figsize=(12, 10))
            sns.set_theme(style="white")
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(
                corr, 
                mask=mask, 
                annot=True, 
                fmt=".2f", 
                cmap="coolwarm", 
                vmin=-1.0, 
                vmax=1.0, 
                square=True, 
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8}
            )
            plt.title('Clinical Visual Features Pairwise Correlation Matrix', fontsize=13, fontweight='bold', pad=15)
            plt.xticks(rotation=45, ha='right', fontsize=8.5)
            plt.yticks(fontsize=8.5)
            plt.tight_layout()
            plt.savefig(figures_dir / "correlation_heatmap.png", dpi=300)
            plt.close()
            print("Correlation matrix saved to result/figures/correlation_heatmap.png")


def detect_penmarks(img_org, mask=None):
    hsv = cv2.cvtColor(img_org, cv2.COLOR_RGB2HSV)
    lower_blue = np.array([90, 45, 30])
    upper_blue = np.array([135, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    if mask is not None:
        if mask.shape[:2] != img_org.shape[:2]:
            mask = cv2.resize(mask, (img_org.shape[1], img_org.shape[0]), interpolation=cv2.INTER_NEAREST)
        outside_mask = cv2.bitwise_not(mask)
        blue_mask = cv2.bitwise_and(blue_mask, outside_mask)
        
    pen_pixels = np.sum(blue_mask > 0)
    total_pixels = img_org.shape[0] * img_org.shape[1]
    return float(pen_pixels / total_pixels)


def optimize_dt_depth(X_train, y_train, groups_train, base_splits):
    depths = [2, 3, 4, 5, 6, 8, 10, None]
    best_depth = 3
    best_auc = -np.inf
    
    for d in depths:
        aucs = []
        if groups_train is not None:
            inner_cv = GroupKFold(n_splits=min(4, len(set(groups_train))))
            inner_splits = list(inner_cv.split(X_train, y_train, groups_train))
        else:
            inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
            inner_splits = list(inner_cv.split(X_train, y_train))
            
        for itr, ival in inner_splits:
            X_itr = X_train.iloc[itr].fillna(X_train.median())
            X_ival = X_train.iloc[ival].fillna(X_train.median())
            
            clf = DecisionTreeClassifier(max_depth=d, random_state=42)
            clf.fit(X_itr, y_train.iloc[itr])
            probs = clf.predict_proba(X_ival)[:, 1]
            try:
                aucs.append(roc_auc_score(y_train.iloc[ival], probs))
            except ValueError:
                pass
        mean_auc = np.mean(aucs) if aucs else -np.inf
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_depth = d
    return best_depth


def train_eval_config(X, y, groups, splits, model_name, models_dir, predictions_path):
    scaler = StandardScaler()
    
    lr_accs, lr_senss, lr_specs, lr_aucs = [], [], [], []
    y_prob_lr_all = []
    
    dt_accs, dt_senss, dt_specs, dt_aucs = [], [], [], []
    y_prob_dt_all = []
    y_true_all = []
    
    dt_depths = []
    
    for tr, val in splits:
        X_tr = X.iloc[tr].fillna(X.median())
        X_val = X.iloc[val].fillna(X.median())
        
        # Scale for Logistic Regression
        X_tr_lr = scaler.fit_transform(X_tr)
        X_val_lr = scaler.transform(X_val)
        
        clf_lr = LogisticRegression(max_iter=1000, random_state=42)
        clf_lr.fit(X_tr_lr, y.iloc[tr])
        probs_lr = clf_lr.predict_proba(X_val_lr)[:, 1]
        preds_lr = clf_lr.predict(X_val_lr)
        
        lr_accs.append(accuracy_score(y.iloc[val], preds_lr))
        lr_senss.append(recall_score(y.iloc[val], preds_lr, zero_division=0))
        tn_lr, fp_lr, fn_lr, tp_lr = confusion_matrix(y.iloc[val], preds_lr, labels=[0, 1]).ravel()
        lr_specs.append(tn_lr / (tn_lr + fp_lr) if (tn_lr + fp_lr) > 0 else 0)
        lr_aucs.append(roc_auc_score(y.iloc[val], probs_lr))
        y_prob_lr_all.extend(probs_lr)
        
        # Decision Tree (optimized depth)
        tr_groups = groups[tr] if groups is not None else None
        best_d = optimize_dt_depth(X_tr, y.iloc[tr], tr_groups, splits)
        dt_depths.append(best_d)
        
        clf_dt = DecisionTreeClassifier(max_depth=best_d, random_state=42)
        clf_dt.fit(X_tr, y.iloc[tr])
        probs_dt = clf_dt.predict_proba(X_val)[:, 1]
        preds_dt = clf_dt.predict(X_val)
        
        dt_accs.append(accuracy_score(y.iloc[val], preds_dt))
        dt_senss.append(recall_score(y.iloc[val], preds_dt, zero_division=0))
        tn_dt, fp_dt, fn_dt, tp_dt = confusion_matrix(y.iloc[val], preds_dt, labels=[0, 1]).ravel()
        dt_specs.append(tn_dt / (tn_dt + fp_dt) if (tn_dt + fp_dt) > 0 else 0)
        dt_aucs.append(roc_auc_score(y.iloc[val], probs_dt))
        y_prob_dt_all.extend(probs_dt)
        
        y_true_all.extend(y.iloc[val])
        
    np.savez(predictions_path, y_true=y_true_all, y_prob_lr=y_prob_lr_all, y_prob_dt=y_prob_dt_all)
    
    # Save final models
    final_lr = LogisticRegression(max_iter=1000, random_state=42)
    final_lr.fit(scaler.fit_transform(X.fillna(X.median())), y)
    with open(models_dir / f"{model_name}_model_lr.pkl", "wb") as f:
        pickle.dump(final_lr, f)
        
    chosen_depth = int(np.round(np.mean([d for d in dt_depths if d is not None]))) if any(d is not None for d in dt_depths) else None
    final_dt = DecisionTreeClassifier(max_depth=chosen_depth, random_state=42)
    final_dt.fit(X.fillna(X.median()), y)
    with open(models_dir / f"{model_name}_model_dt.pkl", "wb") as f:
        pickle.dump(final_dt, f)
        
    with open(models_dir / f"{model_name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    return {
        "lr_acc": np.mean(lr_accs), "lr_sens": np.mean(lr_senss), "lr_spec": np.mean(lr_specs), "lr_auc": np.mean(lr_aucs),
        "dt_acc": np.mean(dt_accs), "dt_sens": np.mean(dt_senss), "dt_spec": np.mean(dt_specs), "dt_auc": np.mean(dt_aucs),
        "dt_depth": chosen_depth
    }


def train_and_evaluate(features_path, annotations_path, imgs_dir, masks_dir, models_dir):
    models_dir.mkdir(parents=True, exist_ok=True)
    df_feat = pd.read_csv(features_path)
    df_annot = pd.read_csv(annotations_path)
    df_merged = pd.merge(df_feat, df_annot, on="img_id")
    
    feature_cols = [
        "mask_area", "asymmetry", "border_compactness", "border_radial",
        "lab_colour_std", "bleeding_likelihood", "mean_h", "mean_s", "mean_v",
        "std_h", "std_s", "std_v", "hue_entropy", "melanoma_colour_count"
    ]
    
    # Automated Shortcut Detection
    print("Running automated shortcut detection...")
    auto_hair, auto_pen = [], []
    for idx, row in df_merged.iterrows():
        img_id = row["img_id"]
        img = cv2.imread(str(imgs_dir / img_id))
        mask_path = masks_dir / (img_id.replace(".png", "_mask.png"))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        
        if img is None:
            auto_hair.append(0.0)
            auto_pen.append(0.0)
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h_mask = hair_mask(img_rgb, mask)
        auto_hair.append(np.sum(h_mask > 0) / (img.shape[0] * img.shape[1]))
        auto_pen.append(detect_penmarks(img_rgb, mask))
        
    df_merged["auto_hair_density"] = auto_hair
    df_merged["auto_pen_density"] = auto_pen
    
    manual_hair_bin = (df_merged["hair"] > 1).astype(int)
    auto_hair_bin = (df_merged["auto_hair_density"] > 0.005).astype(int)
    hair_kappa = cohen_kappa_score(manual_hair_bin, auto_hair_bin)
    
    manual_pen_bin = df_merged["pen"].astype(int)
    auto_pen_bin = (df_merged["auto_pen_density"] > 0.0001).astype(int)
    pen_kappa = cohen_kappa_score(manual_pen_bin, auto_pen_bin)
    
    # Prep feature sets
    X_base = df_merged[feature_cols]
    
    ext_feature_cols = feature_cols + ["auto_hair_density", "auto_pen_density"]
    X_ext = df_merged[ext_feature_cols]
    
    clean_feature_cols = feature_cols + [
        "hair_coverage", "clean_lab_colour_std", "clean_mean_h", "clean_mean_s",
        "clean_mean_v", "clean_std_h", "clean_std_s", "clean_std_v",
        "clean_hue_entropy", "clean_melanoma_colour_count"
    ]
    # Filter features that exist in the CSV
    clean_feature_cols = [c for c in clean_feature_cols if c in df_merged.columns]
    X_clean = df_merged[clean_feature_cols]
    
    y = df_merged["cancer_label"]
    
    # Establish Validation Splits (GroupKFold patient-aware if possible)
    if "patient_id" in df_merged and df_merged["patient_id"].notna().any():
        groups = df_merged["patient_id"].astype(str).to_numpy()
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(X_base, y, groups))
        print("Using patient-aware GroupKFold splits for cross-validation.")
    else:
        groups = None
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(skf.split(X_base, y))
        print("Using standard StratifiedKFold splits for cross-validation.")
        
    predictions_dir = Path("result/predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    print("Training Baseline clinical visual models...")
    res_base = train_eval_config(X_base, y, groups, splits, "baseline", models_dir, predictions_dir / "baseline_predictions.npz")
    
    print("Training Extended confounded shortcut models...")
    res_ext = train_eval_config(X_ext, y, groups, splits, "extended", models_dir, predictions_dir / "extended_predictions.npz")
    
    print("Training Cleaned hair-removed clinical models...")
    res_clean = train_eval_config(X_clean, y, groups, splits, "cleaned", models_dir, predictions_dir / "cleaned_predictions.npz")
    
    summary_text = f"""Cross-Validation Classification Results Summary:

Baseline Logistic Regression:
  Accuracy:    {res_base['lr_acc']:.4f}
  Sensitivity: {res_base['lr_sens']:.4f}
  Specificity: {res_base['lr_spec']:.4f}
  ROC-AUC:     {res_base['lr_auc']:.4f}

Baseline Decision Tree (depth={res_base['dt_depth']}):
  Accuracy:    {res_base['dt_acc']:.4f}
  Sensitivity: {res_base['dt_sens']:.4f}
  Specificity: {res_base['dt_spec']:.4f}
  ROC-AUC:     {res_base['dt_auc']:.4f}

Shortcut Detector Agreement (Cohen's Kappa):
  Hair Detector Kappa:    {hair_kappa:.4f}
  Penmark Detector Kappa: {pen_kappa:.4f}

Extended Logistic Regression (confounded):
  Accuracy:    {res_ext['lr_acc']:.4f}
  Sensitivity: {res_ext['lr_sens']:.4f}
  Specificity: {res_ext['lr_spec']:.4f}
  ROC-AUC:     {res_ext['lr_auc']:.4f}

Extended Decision Tree (confounded, depth={res_ext['dt_depth']}):
  Accuracy:    {res_ext['dt_acc']:.4f}
  Sensitivity: {res_ext['dt_sens']:.4f}
  Specificity: {res_ext['dt_spec']:.4f}
  ROC-AUC:     {res_ext['dt_auc']:.4f}

Cleaned Logistic Regression (hair-removed):
  Accuracy:    {res_clean['lr_acc']:.4f}
  Sensitivity: {res_clean['lr_sens']:.4f}
  Specificity: {res_clean['lr_spec']:.4f}
  ROC-AUC:     {res_clean['lr_auc']:.4f}

Cleaned Decision Tree (hair-removed, depth={res_clean['dt_depth']}):
  Accuracy:    {res_clean['dt_acc']:.4f}
  Sensitivity: {res_clean['dt_sens']:.4f}
  Specificity: {res_clean['dt_spec']:.4f}
  ROC-AUC:     {res_clean['dt_auc']:.4f}
"""
    reports_dir = Path("result/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "model_training_summary.txt", "w") as f:
        f.write(summary_text)
        
    df_merged.to_csv("result/features_with_shortcuts.csv", index=False)
    print("Baseline, extended, and cleaned models trained successfully.")
    print("Saved outputs to result/models/, result/features_with_shortcuts.csv, and result/reports/model_training_summary.txt")


def run_evaluation(figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = Path("result/predictions")
    
    base_pred = np.load(predictions_dir / "baseline_predictions.npz")
    ext_pred = np.load(predictions_dir / "extended_predictions.npz")
    clean_pred = np.load(predictions_dir / "cleaned_predictions.npz")
    
    y_true_b = base_pred["y_true"]
    fpr_b_lr, tpr_b_lr, _ = roc_curve(y_true_b, base_pred["y_prob_lr"])
    auc_b_lr = auc(fpr_b_lr, tpr_b_lr)
    fpr_b_dt, tpr_b_dt, _ = roc_curve(y_true_b, base_pred["y_prob_dt"])
    auc_b_dt = auc(fpr_b_dt, tpr_b_dt)
    
    y_true_e = ext_pred["y_true"]
    fpr_e_lr, tpr_e_lr, _ = roc_curve(y_true_e, ext_pred["y_prob_lr"])
    auc_e_lr = auc(fpr_e_lr, tpr_e_lr)
    fpr_e_dt, tpr_e_dt, _ = roc_curve(y_true_e, ext_pred["y_prob_dt"])
    auc_e_dt = auc(fpr_e_dt, tpr_e_dt)
    
    y_true_c = clean_pred["y_true"]
    fpr_c_lr, tpr_c_lr, _ = roc_curve(y_true_c, clean_pred["y_prob_lr"])
    auc_c_lr = auc(fpr_c_lr, tpr_c_lr)
    fpr_c_dt, tpr_c_dt, _ = roc_curve(y_true_c, clean_pred["y_prob_dt"])
    auc_c_dt = auc(fpr_c_dt, tpr_c_dt)
    
    plt.figure(figsize=(9.5, 8.5))
    sns.set_theme(style="whitegrid")
    
    plt.plot(fpr_b_lr, tpr_b_lr, color='darkorange', lw=2.5, linestyle='-', label=f'Baseline LR (AUC = {auc_b_lr:.4f})')
    plt.plot(fpr_b_dt, tpr_b_dt, color='forestgreen', lw=2.5, linestyle='--', label=f'Baseline DT (AUC = {auc_b_dt:.4f})')
    plt.plot(fpr_e_lr, tpr_e_lr, color='navy', lw=2.5, linestyle='-', label=f'Extended LR (AUC = {auc_e_lr:.4f})')
    plt.plot(fpr_e_dt, tpr_e_dt, color='crimson', lw=2.5, linestyle='--', label=f'Extended DT (AUC = {auc_e_dt:.4f})')
    plt.plot(fpr_c_lr, tpr_c_lr, color='purple', lw=2.5, linestyle='-', label=f'Cleaned LR (AUC = {auc_c_lr:.4f})')
    plt.plot(fpr_c_dt, tpr_c_dt, color='teal', lw=2.5, linestyle='--', label=f'Cleaned DT (AUC = {auc_c_dt:.4f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', labelpad=10, fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', labelpad=10, fontsize=11)
    plt.title('Integrated Model Performance Comparison ROC Curves', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_comparison.png", dpi=300)
    plt.close()
    
    def get_metrics(y_true, probs):
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_true, preds)
        sens = recall_score(y_true, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        auc_score = roc_auc_score(y_true, probs)
        return acc, sens, spec, auc_score
        
    acc_b_lr, sens_b_lr, spec_b_lr, auc_b_lr = get_metrics(y_true_b, base_pred["y_prob_lr"])
    acc_b_dt, sens_b_dt, spec_b_dt, auc_b_dt = get_metrics(y_true_b, base_pred["y_prob_dt"])
    
    acc_e_lr, sens_e_lr, spec_e_lr, auc_e_lr = get_metrics(y_true_e, ext_pred["y_prob_lr"])
    acc_e_dt, sens_e_dt, spec_e_dt, auc_e_dt = get_metrics(y_true_e, ext_pred["y_prob_dt"])
    
    acc_c_lr, sens_c_lr, spec_c_lr, auc_c_lr = get_metrics(y_true_c, clean_pred["y_prob_lr"])
    acc_c_dt, sens_c_dt, spec_c_dt, auc_c_dt = get_metrics(y_true_c, clean_pred["y_prob_dt"])
    
    print("Evaluating models and generating ROC curves...")
    
    summary_text = f"""Comparative Performance Table:

Baseline Logistic Regression:
  Accuracy:    {acc_b_lr*100:.2f}%
  Sensitivity: {sens_b_lr*100:.2f}%
  Specificity: {spec_b_lr*100:.2f}%
  ROC-AUC:     {auc_b_lr:.4f}

Baseline Decision Tree:
  Accuracy:    {acc_b_dt*100:.2f}%
  Sensitivity: {sens_b_dt*100:.2f}%
  Specificity: {spec_b_dt*100:.2f}%
  ROC-AUC:     {auc_b_dt:.4f}

Extended Logistic Regression (confounded):
  Accuracy:    {acc_e_lr*100:.2f}%
  Sensitivity: {sens_e_lr*100:.2f}%
  Specificity: {spec_e_lr*100:.2f}%
  ROC-AUC:     {auc_e_lr:.4f}

Extended Decision Tree (confounded):
  Accuracy:    {acc_e_dt*100:.2f}%
  Sensitivity: {sens_e_dt*100:.2f}%
  Specificity: {spec_e_dt*100:.2f}%
  ROC-AUC:     {auc_e_dt:.4f}

Cleaned Logistic Regression (hair-removed):
  Accuracy:    {acc_c_lr*100:.2f}%
  Sensitivity: {sens_c_lr*100:.2f}%
  Specificity: {spec_c_lr*100:.2f}%
  ROC-AUC:     {auc_c_lr:.4f}

Cleaned Decision Tree (hair-removed):
  Accuracy:    {acc_c_dt*100:.2f}%
  Sensitivity: {sens_c_dt*100:.2f}%
  Specificity: {spec_c_dt*100:.2f}%
  ROC-AUC:     {auc_c_dt:.4f}
"""

    reports_dir = Path("result/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "model_comparison_summary.txt", "w") as f:
        f.write(summary_text)
        
    features_path = Path("data/features.csv")
    if features_path.exists():
        df_feat = pd.read_csv(features_path)
        feature_cols = [
            "mask_area", "asymmetry", "border_compactness", "border_radial",
            "lab_colour_std", "bleeding_likelihood", "mean_h", "mean_s", "mean_v",
            "std_h", "std_s", "std_v", "hue_entropy", "melanoma_colour_count",
            "hair_coverage", "clean_lab_colour_std"
        ]
        cols_present = [c for c in feature_cols if c in df_feat.columns]
        if cols_present:
            corr = df_feat[cols_present].corr()
            plt.figure(figsize=(12, 10))
            sns.set_theme(style="white")
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(
                corr, 
                mask=mask, 
                annot=True, 
                fmt=".2f", 
                cmap="coolwarm", 
                vmin=-1.0, 
                vmax=1.0, 
                square=True, 
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8}
            )
            plt.title('Clinical Visual Features Pairwise Correlation Matrix', fontsize=13, fontweight='bold', pad=15)
            plt.xticks(rotation=45, ha='right', fontsize=8.5)
            plt.yticks(fontsize=8.5)
            plt.tight_layout()
            plt.savefig(figures_dir / "correlation_heatmap.png", dpi=300)
            plt.close()
            
    cm_base = confusion_matrix(y_true_b, (base_pred["y_prob_lr"] >= 0.5).astype(int), labels=[0, 1])
    plt.figure(figsize=(5.5, 4.5))
    sns.set_theme(style="white")
    sns.heatmap(
        cm_base, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=["Benign", "Cancer"], 
        yticklabels=["Benign", "Cancer"],
        cbar=False,
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.ylabel("True Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.title("Baseline Classifier Confusion Matrix (LR)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix_baseline.png", dpi=300)
    plt.close()
    
    cm_ext = confusion_matrix(y_true_e, (ext_pred["y_prob_lr"] >= 0.5).astype(int), labels=[0, 1])
    plt.figure(figsize=(5.5, 4.5))
    sns.set_theme(style="white")
    sns.heatmap(
        cm_ext, 
        annot=True, 
        fmt="d", 
        cmap="Oranges", 
        xticklabels=["Benign", "Cancer"], 
        yticklabels=["Benign", "Cancer"],
        cbar=False,
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.ylabel("True Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.title("Extended Classifier Confusion Matrix (LR)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix_extended.png", dpi=300)
    plt.close()
    
    cm_clean = confusion_matrix(y_true_c, (clean_pred["y_prob_lr"] >= 0.5).astype(int), labels=[0, 1])
    plt.figure(figsize=(5.5, 4.5))
    sns.set_theme(style="white")
    sns.heatmap(
        cm_clean, 
        annot=True, 
        fmt="d", 
        cmap="Purples", 
        xticklabels=["Benign", "Cancer"], 
        yticklabels=["Benign", "Cancer"],
        cbar=False,
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.ylabel("True Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.title("Cleaned Classifier Confusion Matrix (LR)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix_cleaned.png", dpi=300)
    plt.close()
    
    print("Evaluation complete. Generated ROC, correlation heatmap, and confusion matrices in result/figures/")


def run_open_question(features_path, imgs_dir, models_dir):
    models_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_path)
    
    print("Extracting texture HOG features and training classifiers...")
    
    hog_feats = []
    labels = []
    
    for idx, row in df.iterrows():
        img_id = row["img_id"]
        img = cv2.imread(str(imgs_dir / img_id), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        feats = hog(img_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        hog_feats.append(feats)
        labels.append(row["cancer_label"])
        
    X_hog = np.array(hog_feats)
    y_hog = pd.Series(labels)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    
    lr_accs, lr_senss, lr_specs, lr_aucs = [], [], [], []
    dt_accs, dt_senss, dt_specs, dt_aucs = [], [], [], []
    
    for tr, val in skf.split(X_hog, y_hog):
        X_tr_lr, X_val_lr = scaler.fit_transform(X_hog[tr]), scaler.transform(X_hog[val])
        X_tr_dt, X_val_dt = X_hog[tr], X_hog[val]
        
        clf_lr = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
        clf_lr.fit(X_tr_lr, y_hog.iloc[tr])
        probs_lr = clf_lr.predict_proba(X_val_lr)[:, 1]
        preds_lr = clf_lr.predict(X_val_lr)
        
        lr_accs.append(accuracy_score(y_hog.iloc[val], preds_lr))
        lr_senss.append(recall_score(y_hog.iloc[val], preds_lr, zero_division=0))
        tn_lr, fp_lr, fn_lr, tp_lr = confusion_matrix(y_hog.iloc[val], preds_lr, labels=[0, 1]).ravel()
        lr_specs.append(tn_lr / (tn_lr + fp_lr) if (tn_lr + fp_lr) > 0 else 0)
        lr_aucs.append(roc_auc_score(y_hog.iloc[val], probs_lr))
        
        clf_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        clf_dt.fit(X_tr_dt, y_hog.iloc[tr])
        probs_dt = clf_dt.predict_proba(X_val_dt)[:, 1]
        preds_dt = clf_dt.predict(X_val_dt)
        
        dt_accs.append(accuracy_score(y_hog.iloc[val], preds_dt))
        dt_senss.append(recall_score(y_hog.iloc[val], preds_dt, zero_division=0))
        tn_dt, fp_dt, fn_dt, tp_dt = confusion_matrix(y_hog.iloc[val], preds_dt, labels=[0, 1]).ravel()
        dt_specs.append(tn_dt / (tn_dt + fp_dt) if (tn_dt + fp_dt) > 0 else 0)
        dt_aucs.append(roc_auc_score(y_hog.iloc[val], probs_dt))
        
    final_hog_lr = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
    final_hog_lr.fit(scaler.fit_transform(X_hog), y_hog)
    with open(models_dir / "hog_model_lr.pkl", "wb") as f:
        pickle.dump(final_hog_lr, f)
        
    final_hog_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    final_hog_dt.fit(X_hog, y_hog)
    with open(models_dir / "hog_model_dt.pkl", "wb") as f:
        pickle.dump(final_hog_dt, f)
        
    with open(models_dir / "hog_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    hog_summary = f"""HOG feature matrix shape: {X_hog.shape}

HOG Logistic Regression 5-Fold CV:
  Accuracy:    {np.mean(lr_accs):.4f} \u00b1 {np.std(lr_accs):.4f}
  ROC-AUC:     {np.mean(lr_aucs):.4f} \u00b1 {np.std(lr_aucs):.4f}
  Sensitivity: {np.mean(lr_senss):.4f} \u00b1 {np.std(lr_senss):.4f}
  Specificity: {np.mean(lr_specs):.4f} \u00b1 {np.std(lr_specs):.4f}

HOG Decision Tree 5-Fold CV:
  Accuracy:    {np.mean(dt_accs):.4f} \u00b1 {np.std(dt_accs):.4f}
  ROC-AUC:     {np.mean(dt_aucs):.4f} \u00b1 {np.std(dt_aucs):.4f}
  Sensitivity: {np.mean(dt_senss):.4f} \u00b1 {np.std(dt_senss):.4f}
  Specificity: {np.mean(dt_specs):.4f} \u00b1 {np.std(dt_specs):.4f}
"""
    reports_dir = Path("result/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "hog_summary.txt", "w") as f:
        f.write(hog_summary)
        
    print("HOG evaluation complete. Saved summary to result/reports/hog_summary.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PAD-UFES-20 Group F (Fox) skin lesion classification pipeline.")
    parser.add_argument("--mode", type=str, default="analysis", choices=["analysis", "train", "evaluate", "open-question"])
    args = parser.parse_args()
    
    features_path = Path("data/features.csv")
    annotations_path = Path("data/annotations_combined.csv")
    imgs_dir = Path("data/imgs")
    masks_dir = Path("data/masks")
    figures_dir = Path("result/figures")
    models_dir = Path("result/models")
    
    if args.mode == "analysis":
        run_analysis(features_path, annotations_path, figures_dir)
    elif args.mode == "train":
        train_and_evaluate(features_path, annotations_path, imgs_dir, masks_dir, models_dir)
    elif args.mode == "evaluate":
        run_evaluation(figures_dir)
    elif args.mode == "open-question":
        run_open_question(features_path, imgs_dir, models_dir)
