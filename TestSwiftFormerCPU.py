import os
import time
import argparse
import base64
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
from transformers import SwiftFormerForImageClassification, AutoImageProcessor
from collections import defaultdict

# Mapping from English class names to Unicode GD&T symbols
# Adjust these class names based on your actual `train` folder names
SYMBOL_MAP = {
    "angularity": "∠",
    "circular_runout": "↗",
    "circularity": "○",
    "concentricity": "◎",
    "cylindricity": "⌭",
    "flatness": "⏥",
    "parallelism": "⫽",
    "perpendicularity": "⏊",
    "position": "⌖",
    "profile_of_a_line": "⌒",
    "profile_of_a_surface": "⌓",
    "straightness": "⏤",
    "symmetry": "⌯",
    "total_runout": "⌰"
}

def image_to_base64(image, max_size=(224, 224)):
    """Convert a PIL Image to base64 for embedding in HTML"""
    img_copy = image.copy()
    img_copy.thumbnail(max_size)
    buffered = BytesIO()
    img_copy.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

def get_classes(train_dir):
    """Dynamically get class names from the training directory"""
    if os.path.exists(train_dir):
        classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes.sort()  # ImageFolder sorts classes alphabetically
        return classes
    return [f"Class_{i}" for i in range(14)]

def main(args):
    # 1. Setup device to simulate an 8-core x86 CPU
    torch.set_num_threads(8)
    device = torch.device("cpu")
    print(f"Using device: {device} with {torch.get_num_threads()} threads")

    # Load processor and model
    model_name = "MBZUAI/swiftformer-s"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SwiftFormerForImageClassification.from_pretrained(
        model_name,
        num_labels=14,
        ignore_mismatched_sizes=True
    )
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded trained weights from {args.model_path}")
    else:
        print(f"Warning: Model weights {args.model_path} not found. Using untrained weights.")
        
    model.to(device)
    model.eval()

    # Get class mapping
    train_dir = "/Users/shashwanthsivakumar/Desktop/Projects/Adeos/Edocr/edocr2/swiftformer_dataset/train"
    classes = get_classes(train_dir)
    print(f"Detected Classes: {classes}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    image_files = [f for f in os.listdir(args.input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in {args.input_folder}")
        return

    # Track metrics
    total_images = 0
    correct_predictions = 0
    total_latency = 0
    
    # Track per-class metrics: class_name -> {'correct': int, 'wrong': int}
    class_metrics = defaultdict(lambda: {'correct': 0, 'wrong': 0})
    
    table_rows = ""

    # Warmup MPS 
    if device.type == "mps":
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        _ = model(dummy_input)

    for filename in image_files:
        img_path = os.path.join(args.input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        txt_path = os.path.join(args.input_folder, f"{base_name}.txt")
        
        # Read Ground Truth from Text File
        true_symbol = "Unknown"
        has_ground_truth = False
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                true_symbol = f.read().strip()
                has_ground_truth = True
                
        try:
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            if device.type == "mps": torch.mps.synchronize()
            elif device.type == "cuda": torch.cuda.synchronize()
                
            start_time = time.perf_counter()
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs.logits, 1)
                
            if device.type == "mps": torch.mps.synchronize()
            elif device.type == "cuda": torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            total_latency += latency_ms
            
            pred_class_name = classes[predicted.item()]
            
            # Map predicted english class to unicode symbol if available
            pred_symbol = SYMBOL_MAP.get(pred_class_name, pred_class_name)
            
            # Check correctness if GT exists
            is_correct = False
            status_html = ""
            row_style = ""
            
            if has_ground_truth:
                is_correct = (pred_symbol == true_symbol)
                total_images += 1
                if is_correct:
                    correct_predictions += 1
                    class_metrics[true_symbol]['correct'] += 1
                    status_html = "<span class='badge correct'>Correct</span>"
                else:
                    class_metrics[true_symbol]['wrong'] += 1
                    status_html = "<span class='badge wrong'>Wrong</span>"
                    row_style = "background-color: #fef2f2;"
            
            img_b64 = image_to_base64(image)
            
            table_rows += f"""
                    <tr style="{row_style}">
                        <td>{filename}</td>
                        <td class="symbol">{true_symbol}</td>
                        <td class="symbol">{pred_symbol}</td>
                        <td>{status_html}</td>
                        <td><span class="latency">{latency_ms:.2f} ms</span></td>
                        <td class="image-col"><img src="{img_b64}" alt="{filename}"></td>
                    </tr>
            """
            print(f"Processed {filename} | True: {true_symbol} | Pred: {pred_symbol} | { '✔' if is_correct else '✘' } | {latency_ms:.2f}ms")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Build Summary HTML
    accuracy = (correct_predictions / total_images * 100) if total_images > 0 else 0
    avg_latency = (total_latency / len(image_files)) if len(image_files) > 0 else 0
    
    summary_rows = ""
    for sym, counts in sorted(class_metrics.items()):
        total_sym = counts['correct'] + counts['wrong']
        acc_sym = (counts['correct'] / total_sym * 100) if total_sym > 0 else 0
        summary_rows += f"""
            <tr>
                <td class="symbol" style="font-size: 1.5em; text-align: center;">{sym}</td>
                <td style="text-align: right;">{total_sym}</td>
                <td style="text-align: right; color: #16a34a; font-weight: bold;">{counts['correct']}</td>
                <td style="text-align: right; color: #dc2626; font-weight: bold;">{counts['wrong']}</td>
                <td style="text-align: right;">{acc_sym:.1f}%</td>
            </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SwiftFormer Inference Results</title>
        <style>
            body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 40px; background-color: #f8fafc; color: #0f172a; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); padding: 30px; margin-bottom: 30px; }}
            h1, h2 {{ color: #1e293b; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #e2e8f0; vertical-align: middle; }}
            th {{ background-color: #f1f5f9; font-weight: 600; color: #475569; text-transform: uppercase; font-size: 12px; letter-spacing: 0.05em; }}
            .symbol {{ font-weight: 600; color: #3b82f6; font-size: 1.2em; }}
            .latency {{ font-family: ui-monospace, monospace; color: #64748b; background: #f1f5f9; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; }}
            .image-col {{ text-align: right; width: 120px; }}
            .image-col img {{ max-width: 100px; max-height: 100px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            .badge {{ padding: 4px 8px; border-radius: 6px; font-size: 0.85em; font-weight: 600; }}
            .badge.correct {{ background-color: #dcfce7; color: #166534; }}
            .badge.wrong {{ background-color: #fee2e2; color: #991b1b; }}
            .summary-cards {{ display: flex; gap: 20px; margin-bottom: 30px; }}
            .card {{ flex: 1; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; text-align: center; }}
            .card .value {{ font-size: 2em; font-weight: bold; color: #0f172a; margin-top: 10px; }}
            .card .label {{ color: #64748b; font-size: 0.9em; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Evaluation Summary</h1>
            <div class="summary-cards">
                <div class="card">
                    <div class="label">Total Evaluated</div>
                    <div class="value">{total_images}</div>
                </div>
                <div class="card">
                    <div class="label">Overall Accuracy</div>
                    <div class="value" style="color: {'#16a34a' if accuracy > 80 else '#ea580c'};">{accuracy:.1f}%</div>
                </div>
                <div class="card">
                    <div class="label">Correct / Wrong</div>
                    <div class="value"><span style="color:#16a34a">{correct_predictions}</span> / <span style="color:#dc2626">{total_images - correct_predictions}</span></div>
                </div>
                <div class="card">
                    <div class="label">Avg Latency</div>
                    <div class="value">{avg_latency:.1f} ms</div>
                </div>
            </div>

            <h2>Class Breakdown</h2>
            <table style="width: 50%;">
                <thead>
                    <tr>
                        <th style="text-align: center;">Symbol</th>
                        <th style="text-align: right;">Total</th>
                        <th style="text-align: right;">Correct</th>
                        <th style="text-align: right;">Wrong</th>
                        <th style="text-align: right;">Accuracy</th>
                    </tr>
                </thead>
                <tbody>
                    {summary_rows}
                </tbody>
            </table>
        </div>

        <div class="container">
            <h2>Detailed Inference Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>File</th>
                        <th>Ground Truth</th>
                        <th>Predicted</th>
                        <th>Status</th>
                        <th>Latency</th>
                        <th class="image-col">Image</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(args.output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"\n--- Final Results ---")
    print(f"Overall Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_images})")
    print(f"Average Latency:  {avg_latency:.2f}ms")
    print(f"Detailed HTML report saved to {os.path.abspath(args.output_html)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SwiftFormer inference and evaluate against txt ground truth.")
    parser.add_argument("input_folder", type=str, help="Folder containing images AND .txt files")
    parser.add_argument("--output_html", type=str, default="evaluation_results.html", help="Path to save the output HTML file")
    parser.add_argument("--model_path", type=str, default="best_swiftformer_gdt.pth", help="Path to the trained model weights")
    args = parser.parse_args()
    
    main(args)
