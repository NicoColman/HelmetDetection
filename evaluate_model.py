#!/usr/bin/env python3
"""
Helmet Detection Model Evaluation Script
Calculates True Positives, False Positives, False Negatives, and metrics
"""

from ultralytics import YOLO
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

def load_yolo_labels(label_path):
    """Load ground truth labels from YOLO format txt file"""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    boxes.append([class_id, x_center, y_center, width, height])
    return boxes

def convert_yolo_to_xyxy(yolo_box, img_width, img_height):
    """Convert YOLO format (normalized) to absolute coordinates"""
    class_id, x_center, y_center, width, height = yolo_box
    
    # Convert to absolute coordinates
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    # Convert to x1, y1, x2, y2
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    return [x1, y1, x2, y2, class_id]

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def evaluate_model(model_path, test_images_dir, test_labels_dir, iou_threshold=0.5, conf_threshold=0.25):
    """
    Evaluate the model and calculate TP, FP, FN metrics
    """
    # Load model
    model = YOLO(model_path)
    
    # Corrected class mapping (swap the labels as in your original script)
    corrected_class_mapping = {0: 'With_Helmet', 1: 'Without_Helmet'}
    
    # Initialize counters
    tp_with_helmet = 0
    fp_with_helmet = 0
    fn_with_helmet = 0
    tp_without_helmet = 0
    fp_without_helmet = 0
    fn_without_helmet = 0
    
    total_predictions = 0
    total_ground_truth = 0
    
    # Get all test images
    test_images_path = Path(test_images_dir)
    test_labels_path = Path(test_labels_dir)
    
    image_files = list(test_images_path.glob('*.jpg')) + list(test_images_path.glob('*.png'))
    
    print(f"Evaluating {len(image_files)} test images...")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Confidence threshold: {conf_threshold}")
    print("-" * 50)
    
    for img_path in image_files:
        # Get corresponding label file
        label_path = test_labels_path / f"{img_path.stem}.txt"
        
        # Load ground truth
        ground_truth = load_yolo_labels(str(label_path))
        
        # Run model prediction
        results = model(str(img_path))
        result = results[0]
        
        # Get image dimensions
        img_height, img_width = result.orig_shape
        
        # Convert ground truth to absolute coordinates
        gt_boxes = []
        for gt_box in ground_truth:
            abs_box = convert_yolo_to_xyxy(gt_box, img_width, img_height)
            gt_boxes.append(abs_box)
            total_ground_truth += 1
        
        # Get predictions above confidence threshold
        predictions = []
        if len(result.boxes) > 0:
            for box in result.boxes:
                if float(box.conf) >= conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    predictions.append([x1, y1, x2, y2, class_id, confidence])
                    total_predictions += 1
        
        # Match predictions with ground truth
        matched_gt = set()
        matched_pred = set()
        
        # For each prediction, find best matching ground truth
        for pred_idx, pred in enumerate(predictions):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                    
                iou = calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match is good enough
            if best_iou >= iou_threshold and best_gt_idx != -1:
                gt_class = gt_boxes[best_gt_idx][4]
                pred_class = pred[4]
                
                # Check if classes match (remember we swapped labels)
                if gt_class == pred_class:
                    # True Positive
                    if pred_class == 0:  # With_Helmet
                        tp_with_helmet += 1
                    else:  # Without_Helmet
                        tp_without_helmet += 1
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(pred_idx)
                else:
                    # False Positive (wrong class)
                    if pred_class == 0:
                        fp_with_helmet += 1
                    else:
                        fp_without_helmet += 1
            else:
                # False Positive (no matching ground truth)
                if pred[4] == 0:
                    fp_with_helmet += 1
                else:
                    fp_without_helmet += 1
        
        # Count False Negatives (unmatched ground truth)
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx not in matched_gt:
                if gt[4] == 0:  # With_Helmet
                    fn_with_helmet += 1
                else:  # Without_Helmet
                    fn_without_helmet += 1
    
    # Calculate metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # With Helmet metrics
    precision_with = tp_with_helmet / (tp_with_helmet + fp_with_helmet) if (tp_with_helmet + fp_with_helmet) > 0 else 0
    recall_with = tp_with_helmet / (tp_with_helmet + fn_with_helmet) if (tp_with_helmet + fn_with_helmet) > 0 else 0
    f1_with = 2 * (precision_with * recall_with) / (precision_with + recall_with) if (precision_with + recall_with) > 0 else 0
    
    # Without Helmet metrics
    precision_without = tp_without_helmet / (tp_without_helmet + fp_without_helmet) if (tp_without_helmet + fp_without_helmet) > 0 else 0
    recall_without = tp_without_helmet / (tp_without_helmet + fn_without_helmet) if (tp_without_helmet + fn_without_helmet) > 0 else 0
    f1_without = 2 * (precision_without * recall_without) / (precision_without + recall_without) if (precision_without + recall_without) > 0 else 0
    
    # Overall metrics
    total_tp = tp_with_helmet + tp_without_helmet
    total_fp = fp_with_helmet + fp_without_helmet
    total_fn = fn_with_helmet + fn_without_helmet
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    # Print detailed results
    print(f"WITH HELMET:")
    print(f"  True Positives (TP):  {tp_with_helmet}")
    print(f"  False Positives (FP): {fp_with_helmet}")
    print(f"  False Negatives (FN): {fn_with_helmet}")
    print(f"  Precision: {precision_with:.3f}")
    print(f"  Recall:    {recall_with:.3f}")
    print(f"  F1-Score:  {f1_with:.3f}")
    
    print(f"\nWITHOUT HELMET:")
    print(f"  True Positives (TP):  {tp_without_helmet}")
    print(f"  False Positives (FP): {fp_without_helmet}")
    print(f"  False Negatives (FN): {fn_without_helmet}")
    print(f"  Precision: {precision_without:.3f}")
    print(f"  Recall:    {recall_without:.3f}")
    print(f"  F1-Score:  {f1_without:.3f}")
    
    print(f"\nOVERALL METRICS:")
    print(f"  Total TP: {total_tp}")
    print(f"  Total FP: {total_fp}")
    print(f"  Total FN: {total_fn}")
    print(f"  Precision: {overall_precision:.3f}")
    print(f"  Recall:    {overall_recall:.3f}")
    print(f"  F1-Score:  {overall_f1:.3f}")
    
    print(f"\nSUMMARY:")
    print(f"  Total test images: {len(image_files)}")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Total ground truth: {total_ground_truth}")
    
def create_evaluation_plots(results, output_dir="evaluation_plots"):
    """Create visualization plots for the evaluation results"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Confusion Matrix Style Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # TP/FP/FN Bar Chart
    categories = ['Con Casco', 'Sin Casco']
    tp_values = [results['tp_with_helmet'], results['tp_without_helmet']]
    fp_values = [results['fp_with_helmet'], results['fp_without_helmet']]
    fn_values = [results['fn_with_helmet'], results['fn_without_helmet']]
    
    x = np.arange(len(categories))
    width = 0.25
    
    ax1.bar(x - width, tp_values, width, label='Verdaderos Positivos (TP)', color='#2ecc71', alpha=0.8)
    ax1.bar(x, fp_values, width, label='Falsos Positivos (FP)', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width, fn_values, width, label='Falsos Negativos (FN)', color='#f39c12', alpha=0.8)
    
    ax1.set_xlabel('Clases')
    ax1.set_ylabel('Cantidad de Detecciones')
    ax1.set_title('Distribución de TP, FP, FN por Clase')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (tp, fp, fn) in enumerate(zip(tp_values, fp_values, fn_values)):
        ax1.text(i-width, tp + 0.5, str(tp), ha='center', va='bottom', fontweight='bold')
        ax1.text(i, fp + 0.5, str(fp), ha='center', va='bottom', fontweight='bold')
        ax1.text(i+width, fn + 0.5, str(fn), ha='center', va='bottom', fontweight='bold')
    
    # 2. Precision, Recall, F1-Score Comparison
    metrics = ['Precisión', 'Recall', 'F1-Score']
    with_helmet_scores = [results['precision_with'], results['recall_with'], results['f1_with']]
    without_helmet_scores = [results['precision_without'], results['recall_without'], results['f1_without']]
    
    x = np.arange(len(metrics))
    ax2.bar(x - width/2, with_helmet_scores, width, label='Con Casco', color='#3498db', alpha=0.8)
    ax2.bar(x + width/2, without_helmet_scores, width, label='Sin Casco', color='#9b59b6', alpha=0.8)
    
    ax2.set_xlabel('Métricas')
    ax2.set_ylabel('Puntuación (0-1)')
    ax2.set_title('Métricas de Rendimiento por Clase')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # Add value labels
    for i, (with_score, without_score) in enumerate(zip(with_helmet_scores, without_helmet_scores)):
        ax2.text(i-width/2, with_score + 0.02, f'{with_score:.3f}', ha='center', va='bottom', fontweight='bold')
        ax2.text(i+width/2, without_score + 0.02, f'{without_score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Overall Performance Pie Chart
    total_tp = results['tp_with_helmet'] + results['tp_without_helmet']
    total_fp = results['fp_with_helmet'] + results['fp_without_helmet']
    total_fn = results['fn_with_helmet'] + results['fn_without_helmet']
    
    sizes = [total_tp, total_fp, total_fn]
    labels = [f'Verdaderos Positivos\n({total_tp})', f'Falsos Positivos\n({total_fp})', f'Falsos Negativos\n({total_fn})']
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    explode = (0.1, 0, 0)  # explode TP slice
    
    ax3.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax3.set_title('Distribución General de Resultados')
    
    # 4. Overall Metrics Gauge/Bar
    overall_metrics = ['Precisión General', 'Recall General', 'F1-Score General']
    overall_values = [results['overall_precision'], results['overall_recall'], results['overall_f1']]
    
    bars = ax4.barh(overall_metrics, overall_values, color=['#1abc9c', '#34495e', '#e67e22'], alpha=0.8)
    ax4.set_xlabel('Puntuación (0-1)')
    ax4.set_title('Métricas Generales del Modelo')
    ax4.set_xlim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, overall_values)):
        ax4.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/evaluation_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Create a summary report figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Title
    fig.suptitle('REPORTE DE EVALUACIÓN - DETECTOR DE CASCOS', fontsize=20, fontweight='bold', y=0.95)
    
    # Create text summary
    summary_text = f"""
MÉTRICAS POR CLASE:

CON CASCO:
    • Verdaderos Positivos (TP): {results['tp_with_helmet']}
    • Falsos Positivos (FP): {results['fp_with_helmet']}
    • Falsos Negativos (FN): {results['fn_with_helmet']}
    • Precisión: {results['precision_with']:.3f}
    • Recall: {results['recall_with']:.3f}
    • F1-Score: {results['f1_with']:.3f}

SIN CASCO:
    • Verdaderos Positivos (TP): {results['tp_without_helmet']}
    • Falsos Positivos (FP): {results['fp_without_helmet']}
    • Falsos Negativos (FN): {results['fn_without_helmet']}
    • Precisión: {results['precision_without']:.3f}
    • Recall: {results['recall_without']:.3f}
    • F1-Score: {results['f1_without']:.3f}

MÉTRICAS GENERALES:
    • Precisión General: {results['overall_precision']:.3f}
    • Recall General: {results['overall_recall']:.3f}
    • F1-Score General: {results['overall_f1']:.3f}
    
RESUMEN:
    • Total Detecciones Correctas: {total_tp}
    • Total Detecciones Incorrectas: {total_fp}
    • Total Detecciones Perdidas: {total_fn}
    """
    
    ax.text(0.05, 0.85, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=1", facecolor='lightgray', alpha=0.8))
    
    plt.savefig(f'{output_dir}/evaluation_report.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Gráficos guardados en: {output_dir}/")
    print("   - evaluation_metrics.png (gráficos detallados)")
    print("   - evaluation_report.png (reporte resumen)")
    
    return output_dir

def main():
    """Main function to run evaluation"""
    
    # Configuration - UPDATE THESE PATHS
    MODEL_PATH = "runs/detect/helmet_detection_v1_single_gpu/weights/best.pt"
    TEST_IMAGES_DIR = "./Set/test/images"
    TEST_LABELS_DIR = "./Set/test/labels"
    
    print("🎯 Helmet Detection Model Evaluation")
    print("="*60)
    
    # Check if paths exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"❌ Test images directory not found: {TEST_IMAGES_DIR}")
        return
        
    if not os.path.exists(TEST_LABELS_DIR):
        print(f"❌ Test labels directory not found: {TEST_LABELS_DIR}")
        return
    
    # Run evaluation
    results = evaluate_model(
        model_path=MODEL_PATH,
        test_images_dir=TEST_IMAGES_DIR,
        test_labels_dir=TEST_LABELS_DIR,
        iou_threshold=0.5,
        conf_threshold=0.25
    )
    
    print("\n🎉 Evaluation complete!")
    print("This gives you the TP/FP metrics your professor asked for!")

if __name__ == "__main__":
    main()