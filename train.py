import torch
import torchvision
import ultralytics
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Check for CUDA availability and print versions
def check_environment():
    print("CUDA available:", torch.cuda.is_available())
    print("Torch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("Ultralytics version:", ultralytics.__version__)
    if torch.cuda.is_available():
        print("Available GPU(s):")
        for i in range(torch.cuda.device_count()):
            print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU available.")

# Main function to execute training
def main():
    check_environment()
    
  
    model = ultralytics.YOLO("yolov9t.pt")  


    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=416,
        batch=16,
        project="runs/train",
        name="yolov8_custom",
        optimizer="AdamW",
        lr0=0.01,
        weight_decay=0.0005,
        plots=True
    )

    
    
    print("Training completed.")
    print("Training results saved at:", results.save_dir)

if __name__ == '__main__':
    main()
