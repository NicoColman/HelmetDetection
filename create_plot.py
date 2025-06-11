#!/usr/bin/env python3
"""
Standalone script to create plots from your helmet detection evaluation results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def create_helmet_evaluation_plots():
    """Create plots using your actual results"""
    
    # Your actual results from the terminal output
    results = {
        'tp_with_helmet': 156,
        'fp_with_helmet': 49,
        'fn_with_helmet': 15,
        'tp_without_helmet': 84,
        'fp_without_helmet': 20,
        'fn_without_helmet': 19,
        'precision_with': 0.761,
        'recall_with': 0.912,
        'f1_with': 0.830,
        'precision_without': 0.808,
        'recall_without': 0.816,
        'f1_without': 0.812,
        'overall_precision': 0.777,
        'overall_recall': 0.876,
        'overall_f1': 0.823
    }
    
    # Create output directory
    output_dir = "evaluation_plots"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create the main figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. TP/FP/FN Bar Chart
    categories = ['Con Casco', 'Sin Casco']
    tp_values = [results['tp_with_helmet'], results['tp_without_helmet']]
    fp_values = [results['fp_with_helmet'], results['fp_without_helmet']]
    fn_values = [results['fn_with_helmet'], results['fn_without_helmet']]
    
    x = np.arange(len(categories))
    width = 0.25
    
    bars1 = ax1.bar(x - width, tp_values, width, label='Verdaderos Positivos (TP)', 
                    color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x, fp_values, width, label='Falsos Positivos (FP)', 
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax1.bar(x + width, fn_values, width, label='Falsos Negativos (FN)', 
                    color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Clases', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cantidad de Detecciones', fontsize=12, fontweight='bold')
    ax1.set_title('Distribución de TP, FP, FN por Clase', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (tp, fp, fn) in enumerate(zip(tp_values, fp_values, fn_values)):
        ax1.text(i-width, tp + 1, str(tp), ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax1.text(i, fp + 1, str(fp), ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax1.text(i+width, fn + 1, str(fn), ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. Precision, Recall, F1-Score Comparison
    metrics = ['Precisión', 'Recall', 'F1-Score']
    with_helmet_scores = [results['precision_with'], results['recall_with'], results['f1_with']]
    without_helmet_scores = [results['precision_without'], results['recall_without'], results['f1_without']]
    
    x = np.arange(len(metrics))
    bars1 = ax2.bar(x - width/2, with_helmet_scores, width, label='Con Casco', 
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax2.bar(x + width/2, without_helmet_scores, width, label='Sin Casco', 
                    color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax2.set_xlabel('Métricas', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Puntuación (0-1)', fontsize=12, fontweight='bold')
    ax2.set_title('Métricas de Rendimiento por Clase', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # Add value labels
    for i, (with_score, without_score) in enumerate(zip(with_helmet_scores, without_helmet_scores)):
        ax2.text(i-width/2, with_score + 0.02, f'{with_score:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
        ax2.text(i+width/2, without_score + 0.02, f'{without_score:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    # 3. Overall Performance Pie Chart
    total_tp = results['tp_with_helmet'] + results['tp_without_helmet']
    total_fp = results['fp_with_helmet'] + results['fp_without_helmet']
    total_fn = results['fn_with_helmet'] + results['fn_without_helmet']
    
    sizes = [total_tp, total_fp, total_fn]
    labels = [f'Verdaderos Positivos\n({total_tp})', f'Falsos Positivos\n({total_fp})', f'Falsos Negativos\n({total_fn})']
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    explode = (0.1, 0, 0)  # explode TP slice
    
    wedges, texts, autotexts = ax3.pie(sizes, explode=explode, labels=labels, colors=colors, 
                                       autopct='%1.1f%%', shadow=True, startangle=90, 
                                       textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax3.set_title('Distribución General de Resultados', fontsize=14, fontweight='bold')
    
    # 4. Overall Metrics Horizontal Bar
    overall_metrics = ['Precisión\nGeneral', 'Recall\nGeneral', 'F1-Score\nGeneral']
    overall_values = [results['overall_precision'], results['overall_recall'], results['overall_f1']]
    
    bars = ax4.barh(overall_metrics, overall_values, 
                   color=['#1abc9c', '#34495e', '#e67e22'], alpha=0.8, 
                   edgecolor='black', linewidth=1)
    ax4.set_xlabel('Puntuación (0-1)', fontsize=12, fontweight='bold')
    ax4.set_title('Métricas Generales del Modelo', fontsize=14, fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, overall_values)):
        ax4.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', ha='left', va='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/evaluation_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/evaluation_metrics.png")
    
    # 5. Create a professional summary report
    fig2, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    
    # Title
    fig2.suptitle('REPORTE DE EVALUACIÓN\nDETECTOR DE CASCOS YOLOv8', 
                  fontsize=22, fontweight='bold', y=0.95)
    
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
    
RESUMEN FINAL:
    • Total Detecciones Correctas: {total_tp}
    • Total Detecciones Incorrectas: {total_fp}
    • Total Detecciones Perdidas: {total_fn}
    • Rendimiento: EXCELENTE (F1: 0.823)
    """
    
    ax.text(0.05, 0.85, summary_text, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=1.5", facecolor='lightblue', alpha=0.8))
    
    # Add conclusion box
    conclusion_text = """
CONCLUSIÓN:
El modelo YOLOv8 muestra un rendimiento excelente:
• Alta precisión en ambas clases (>76%)
• Excelente recall para detección de cascos (91.2%)
• F1-Score balanceado y robusto (0.823)
• Modelo listo para aplicación práctica
    """
    
    ax.text(0.05, 0.25, conclusion_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle="round,pad=1", facecolor='lightgreen', alpha=0.8))
    
    plt.savefig(f'{output_dir}/evaluation_report.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/evaluation_report.png")
    
    plt.close('all')
    
    print(f"\n📊 All plots saved in '{output_dir}/' folder")
    print("📈 Your model performance is EXCELLENT!")
    print("   - F1-Score: 0.823 (Very Good)")
    print("   - High recall for helmet detection: 91.2%")
    print("   - Balanced precision across both classes")

if __name__ == "__main__":
    print("🎯 Creating Helmet Detection Evaluation Plots...")
    print("=" * 50)
    
    try:
        create_helmet_evaluation_plots()
        print("\n🎉 Success! Check the 'evaluation_plots' folder for your graphs!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have matplotlib and seaborn installed:")
        print("pip install matplotlib seaborn")