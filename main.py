#!/usr/bin/env python3
"""
YOLOv8 Helmet Detection Model Training Script
University Project - Computer Vision

This script trains a YOLOv8 model to detect helmets (with/without helmet classification)
Author: [Your Name]
Date: [Current Date]
"""

import os
import yaml
import torch
import torch.distributed as dist
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2
import numpy as np

def setup_gpu_environment():
    """
    Setup GPU environment for dual GPU training
    """
    print("GPU Setup Information:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Set memory growth to avoid allocation issues
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    else:
        print("WARNING: CUDA not available, will use CPU (very slow)")
    
    return torch.cuda.device_count()

class HelmetDetectionTrainer:
    def __init__(self, dataset_path, project_name="helmet_detection"):
        """
        Initialize the helmet detection trainer
        
        Args:
            dataset_path (str): Path to your dataset directory
            project_name (str): Name for your project outputs
        """
        self.dataset_path = Path(dataset_path)
        self.project_name = project_name
        self.model = None
        self.results = None
        
        # Create output directories
        self.output_dir = Path(f"runs/{project_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_dataset_yaml(self):
        """
        Create the dataset configuration file required by YOLO
        Assumes your dataset structure is:
        dataset/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/  (or val/)
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
        """
        
        # Define class names for helmet detection
        class_names = ['Without_Helmet', 'With_Helmet']  # 0: Without Helmet, 1: With Helmet
        
        # Create dataset configuration
        dataset_config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'train/images',
            'val': 'valid/images',  # Changed from 'val' to 'valid' to match your folder structure
            'test': 'test/images',
            'nc': len(class_names),  # number of classes
            'names': class_names
        }
        
        # Save the configuration file
        yaml_path = self.dataset_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Dataset configuration saved to: {yaml_path}")
        return yaml_path
    
    def load_model(self, model_size='n'):
        """
        Load YOLOv8 model
        
        Args:
            model_size (str): Model size ('n', 's', 'm', 'l', 'x')
                            'n' = nano (fastest, least accurate)
                            's' = small
                            'm' = medium  
                            'l' = large
                            'x' = extra large (slowest, most accurate)
        """
        model_name = f'yolov8{model_size}.pt'
        self.model = YOLO(model_name)
        print(f"Loaded YOLOv8{model_size.upper()} model")
        
    def train_model(self, epochs=100, img_size=640, batch_size=16, use_dual_gpu=True):
        """
        Train the YOLO model
        
        Args:
            epochs (int): Number of training epochs
            img_size (int): Image size for training
            batch_size (int): Batch size (reduce if you get memory errors)
            use_dual_gpu (bool): Whether to attempt dual GPU training
        """
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Create dataset.yaml if it doesn't exist
        yaml_path = self.dataset_path / 'dataset.yaml'
        if not yaml_path.exists():
            yaml_path = self.create_dataset_yaml()
        
        print("Starting training...")
        print(f"Epochs: {epochs}")
        print(f"Image size: {img_size}")
        print(f"Batch size: {batch_size}")
        
        # Determine device configuration
        device_config = 'cpu'  # Default fallback
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            
            if use_dual_gpu and gpu_count >= 2:
                print("Attempting dual GPU training...")
                device_config = [0, 1]  # Try dual GPU first
            elif gpu_count >= 1:
                print("Using single GPU training...")
                device_config = 0  # Single GPU
            else:
                print("No GPU available, using CPU (this will be very slow)...")
                device_config = 'cpu'
        
        # Try training with different configurations
        training_configs = []
        
        if use_dual_gpu and torch.cuda.device_count() >= 2:
            # First try: Dual GPU with high settings
            training_configs.append({
                'device': [0, 1],
                'batch': batch_size,
                'workers': 8,
                'name': f"{self.project_name}_dual_gpu"
            })
            
            # Second try: Dual GPU with reduced settings
            training_configs.append({
                'device': [0, 1],
                'batch': max(batch_size // 2, 8),
                'workers': 4,
                'name': f"{self.project_name}_dual_gpu_reduced"
            })
        
        # Single GPU fallback
        if torch.cuda.is_available():
            training_configs.append({
                'device': 0,
                'batch': max(batch_size // 2, 8),
                'workers': 4,
                'name': f"{self.project_name}_single_gpu"
            })
        
        # CPU fallback
        training_configs.append({
            'device': 'cpu',
            'batch': max(batch_size // 4, 4),
            'workers': 2,
            'name': f"{self.project_name}_cpu"
        })
        
        # Try each configuration until one works
        for i, config in enumerate(training_configs):
            try:
                print(f"\nTrying training configuration {i+1}/{len(training_configs)}:")
                print(f"  Device: {config['device']}")
                print(f"  Batch size: {config['batch']}")
                print(f"  Workers: {config['workers']}")
                
                # Train the model
                self.results = self.model.train(
                    data=str(yaml_path),
                    epochs=epochs,
                    imgsz=img_size,
                    batch=config['batch'],
                    name=config['name'],
                    patience=50,  # Early stopping patience
                    save=True,
                    device=config['device'],
                    workers=config['workers'],
                    verbose=True
                )
                
                # Update project name to match successful configuration
                self.project_name = config['name']
                
                print("Training completed successfully!")
                return
                
            except Exception as e:
                print(f"Training configuration {i+1} failed: {str(e)}")
                if i < len(training_configs) - 1:
                    print("Trying next configuration...")
                else:
                    print("All training configurations failed!")
                    raise e
        
    def validate_model(self):
        """Validate the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        print("Validating model...")
        validation_results = self.model.val()
        
        return validation_results
    
    def test_model(self, test_image_path=None):
        """
        Test the model on a single image or test dataset
        
        Args:
            test_image_path (str): Path to test image (optional)
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        if test_image_path:
            # Test on single image
            results = self.model(test_image_path)
            
            # Display results
            for r in results:
                # Plot results
                img_with_detections = r.plot()
                
                # Save result
                output_path = self.output_dir / f"test_result_{Path(test_image_path).name}"
                cv2.imwrite(str(output_path), img_with_detections)
                print(f"Test result saved to: {output_path}")
                
                # Print detection details
                if len(r.boxes) > 0:
                    for i, box in enumerate(r.boxes):
                        class_id = int(box.cls)
                        confidence = float(box.conf)
                        class_name = r.names[class_id]
                        print(f"Detection {i+1}: {class_name} (confidence: {confidence:.2f})")
                else:
                    print("No detections found")
        else:
            # Test on test dataset
            test_results = self.model.val(split='test')
            return test_results
    
    def save_model(self, model_name="best_helmet_model.pt"):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        # The best model is automatically saved during training
        # We can also export it in different formats
        best_model_path = f"runs/detect/{self.project_name}/weights/best.pt"
        
        if os.path.exists(best_model_path):
            # Copy to our output directory
            import shutil
            output_model_path = self.output_dir / model_name
            shutil.copy2(best_model_path, output_model_path)
            print(f"Model saved to: {output_model_path}")
            
            # Export to ONNX format (useful for deployment)
            try:
                model = YOLO(best_model_path)
                model.export(format='onnx')
                print("Model also exported to ONNX format")
            except Exception as e:
                print(f"ONNX export failed: {e}")
        
    def generate_report(self):
        """Generate a training report for your university project"""
        
        print("\n" + "="*50)
        print("HELMET DETECTION MODEL - TRAINING REPORT")
        print("="*50)
        
        # Model information
        print(f"Project Name: {self.project_name}")
        print(f"Dataset Path: {self.dataset_path}")
        print(f"Model Architecture: YOLOv8")
        
        # Training results
        if self.results:
            try:
                # Get the results directory
                results_dir = f"runs/detect/{self.project_name}"
                results_csv = f"{results_dir}/results.csv"
                
                if os.path.exists(results_csv):
                    df = pd.read_csv(results_csv)
                    
                    print(f"\nTraining completed in {len(df)} epochs")
                    print(f"Best mAP50: {df['metrics/mAP50(B)'].max():.4f}")
                    print(f"Best mAP50-95: {df['metrics/mAP50-95(B)'].max():.4f}")
                    print(f"Final training loss: {df['train/box_loss'].iloc[-1]:.4f}")
                    print(f"Final validation loss: {df['val/box_loss'].iloc[-1]:.4f}")
                
                # List output files
                print(f"\nOutput files located in: {results_dir}")
                print("- best.pt: Best model weights")
                print("- last.pt: Final epoch weights")  
                print("- results.csv: Training metrics")
                print("- confusion_matrix.png: Confusion matrix")
                print("- results.png: Training curves")
                
            except Exception as e:
                print(f"Error generating detailed report: {e}")
        
        print("\nModel ready for deployment!")
        print("="*50)

def main():
    """
    Main function to run the complete training pipeline
    """
    
    # CONFIGURATION - MODIFY THESE PATHS FOR YOUR SETUP
    DATASET_PATH = "./Set"  # Relative path to your dataset folder
    PROJECT_NAME = "helmet_detection_v1"
    
    # Training parameters (optimized for dual GPU setup: RTX 3090 + RTX 4090)
    EPOCHS = 100          # Increased epochs for better results with powerful GPUs
    IMG_SIZE = 640        # Image size (can increase to 1280 with your GPU power)
    BATCH_SIZE = 32       # Increased batch size for dual GPU (adjust based on VRAM)
    MODEL_SIZE = 'm'      # Medium model - good balance for your GPU setup
    
    print("YOLOv8 Helmet Detection Training Pipeline")
    print("=========================================")
    
    # Setup GPU environment
    print("\n0. Setting up GPU environment...")
    num_gpus = setup_gpu_environment()
    
    # Initialize trainer
    trainer = HelmetDetectionTrainer(DATASET_PATH, PROJECT_NAME)
    
    # Step 1: Load model
    print("\n1. Loading YOLOv8 model...")
    trainer.load_model(MODEL_SIZE)
    
    # Step 2: Create dataset configuration
    print("\n2. Creating dataset configuration...")
    trainer.create_dataset_yaml()
    
    # Step 3: Train model
    print("\n3. Training model...")
    trainer.train_model(epochs=EPOCHS, img_size=IMG_SIZE, batch_size=BATCH_SIZE, use_dual_gpu=True)
    
    # Step 4: Validate model
    print("\n4. Validating model...")
    trainer.validate_model()
    
    # Step 5: Save model
    print("\n5. Saving model...")
    trainer.save_model()
    
    # Step 6: Generate report
    print("\n6. Generating report...")
    trainer.generate_report()
    
    # Optional: Test on a specific image
    # Uncomment and modify the path below to test on a specific image
    # test_image = "path/to/test/image.jpg"
    # trainer.test_model(test_image)
    
    print("\nTraining pipeline completed successfully!")
    print("Check the 'runs/detect/' directory for all outputs.")

if __name__ == "__main__":
    # Install required packages first (comment out if already installed)
    print("Installing required packages...")
    try:
        os.system("pip install ultralytics opencv-python matplotlib pandas pillow pyyaml")
    except:
        print("Package installation failed, assuming they're already installed...")
    
    # Run main training pipeline
    main()