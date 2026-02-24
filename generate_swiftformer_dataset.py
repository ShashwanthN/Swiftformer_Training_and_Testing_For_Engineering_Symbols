import os
import cv2
import random
import numpy as np
import argparse
from PIL import Image, ImageDraw, ImageFont

# Mapping symbols to English folder names to ensure cross-platform compatibility
# without relying on Unicode directory names.
SYMBOL_NAMES = {
    '⏤': 'straightness',
    '⏥': 'flatness',
    '○': 'circularity',
    '⌭': 'cylindricity',
    '⌒': 'profile_of_a_line',
    '⌓': 'profile_of_a_surface',
    '⏊': 'perpendicularity',
    '∠': 'angularity',
    '⫽': 'parallelism',
    '⌯': 'symmetry',
    '⌖': 'position',
    '◎': 'concentricity',
    '↗': 'circular_runout',
    '⌰': 'total_runout'
}

# The two fonts literally map GD&T symbols to ASCII standard english keyboard characters.
# We must translate the intended symbol into the correct ASCII letter before drawing it.
# Y14.5 maps flatness to `b`, cylindricity to `d`, etc.
# GDT Regular maps flatness to `c`, cylindricity to `g`, etc.
FONT_MAPS = {
    'Y145m.ttf': {
        '⏤': 'g',
        '⏥': 'i',
        '○': 'c',
        '⌭': 'b',
        '◎': 'e',
        '⌒': 'j',
        '⌓': 'k',
        '⫽': 'h',
        '∠': 'a',
        '⏊': 'o',
        '⌯': 'n',
        '⌖': 'v',
        '↗': 'q',
        '⌰': 'r'
    },
    'GDT Regular.ttf': {
        '∠': 'a',
        '⏊': 'b',
        '⏥': 'c',
        '⌓': 'd',
        '○': 'e',
        '⫽': 'f',
        '⌭': 'g',
        '↗': 'h',
        '⌯': 'i',
        '⌖': 'j',
        '⌒': 'k',
        '◎': 'r',
        '⌰': 't',
        '⏤': '-'
    }
}
# Fallbacks for any missing characters
for fmap in FONT_MAPS.values():
    if '⏤' not in fmap or fmap['⏤'] == '-':
        pass # Handle natively

GDT_SYMBOLS = list(SYMBOL_NAMES.keys())

def get_perspective_transform(img, strength=0.2):
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dw, dh = w * strength, h * strength
    pts2 = np.float32([
        [random.uniform(0, dw), random.uniform(0, dh)],
        [w - random.uniform(0, dw), random.uniform(0, dh)],
        [random.uniform(0, dw), h - random.uniform(0, dh)],
        [w - random.uniform(0, dw), h - random.uniform(0, dh)]
    ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

def add_lighting_gradient(img):
    h, w = img.shape[:2]
    Y, X = np.indices((h, w))
    cx, cy = random.randint(-w//2, int(w*1.5)), random.randint(-h//2, int(h*1.5))
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_dist = np.sqrt(h**2 + w**2)
    mask = 1.0 - (dist / max_dist) * random.uniform(0.3, 0.9)
    mask = np.clip(mask, 0.3, 1.1)
    return np.clip(img * mask, 0, 255).astype(np.uint8)

def apply_random_anomaly(img, severity):
    """ Applies a random mix of anomalies based on severity (mild vs extreme) """
    if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if severity == 'extreme' and random.random() > 0.4:
        img = get_perspective_transform(img, strength=random.uniform(0.05, 0.15)) # Toned down perspective
        
    if severity == 'extreme' and random.random() > 0.4:
        img = add_lighting_gradient(img)

    # Noise
    if random.random() > (0.5 if severity == 'mild' else 0.1):
        std_dev = 15 if severity == 'mild' else random.uniform(30, 80)
        noise = np.random.normal(0, std_dev, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Blur
    if random.random() > 0.3:
        k = random.choice([3, 5]) if severity == 'mild' else random.choice([5, 7, 9])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Contrast/Brightness
    if random.random() > (0.5 if severity == 'mild' else 0.4):
        if severity == 'mild':
            alpha = random.uniform(0.8, 1.1)
            beta = random.randint(-10, 10)
        else:
            alpha = random.uniform(0.5, 0.8) if random.random() > 0.5 else random.uniform(1.2, 1.5) # Toned down
            beta = random.randint(-40, 40)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
    # Simulated Scan Lines (Mild only)
    if severity == 'mild' and random.random() > 0.7:
        size = random.choice([3, 5])
        kernel = np.zeros((size, size))
        if random.random() > 0.5: kernel[int((size-1)/2), :] = np.ones(size) / size
        else: kernel[:, int((size-1)/2)] = np.ones(size) / size
        img = cv2.filter2D(img, -1, kernel)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def draw_fcf_box(draw, img_size):
    """ Draws an FCF cell that hugs the outer edges of the image, keeping the symbol inside padded """
    # Random edge distance between 5 and 15 pixels from the image bounds
    edge_px = random.randint(5, 15)
    
    left = edge_px
    top = edge_px
    right = img_size - edge_px
    bottom = img_size - edge_px
    
    # Random border thickness
    thickness = random.randint(3, 7)
    draw.rectangle([left, top, right, bottom], outline=0, width=thickness)


def generate_classification_dataset(samples_per_symbol, output_dir, train_split=0.8):
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup PyTorch ImageFolder structure: dataset/train/class_name and dataset/val/class_name
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    for sym_name in SYMBOL_NAMES.values():
        os.makedirs(os.path.join(train_dir, sym_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, sym_name), exist_ok=True)

    gdt_fonts = [os.path.join('Fonts', f) for f in ['Y145m.ttf', 'GDT Regular.ttf']]

    if not gdt_fonts:
        print("Error: No TTF fonts found in 'Fonts'")
        return

    num_train = int(samples_per_symbol * train_split)
    num_val = samples_per_symbol - num_train
    
    total_images = len(GDT_SYMBOLS) * samples_per_symbol
    print(f"Generating {total_images} images for SwiftFormer ({num_train} train, {num_val} val per class)...")
    
    img_size = 224 # Standard input size for typical Vision Transformers like SwiftFormer
    
    for sym in GDT_SYMBOLS:
        sym_name = SYMBOL_NAMES[sym]
        print(f"Generating {samples_per_symbol} samples for '{sym_name}'...")
        
        for i in range(samples_per_symbol):
            is_train = i < num_train
            split_dir = train_dir if is_train else val_dir
            
            # 50% chance of severe anomalies, 50% chance of mild anomalies
            severity = 'extreme' if random.random() > 0.5 else 'mild'
            
            font_path = random.choice(gdt_fonts)
            image = Image.new('L', (img_size, img_size), color=255)
            draw = ImageDraw.Draw(image)
            
            # Draw the box FIRST so the symbol is overlaid properly on top
            # 70% probability of drawing an FCF box
            if random.random() > 0.3:
                draw_fcf_box(draw, img_size)
                
            # Randomize font size to be very large but safely within the box padding
            fsize = random.randint(120, 160)
            
            try: 
                font = ImageFont.truetype(font_path, fsize)
            except Exception: 
                continue 
                
            font_basename = os.path.basename(font_path)
            char_to_draw = FONT_MAPS.get(font_basename, {}).get(sym, sym)
                
            bbox = draw.textbbox((0, 0), char_to_draw, font=font)
            
            # Exact dead center of the 224x224 image
            x = (img_size - (bbox[2] - bbox[0])) / 2 - bbox[0] + random.randint(-5, 5)
            y = (img_size - (bbox[3] - bbox[1])) / 2 - bbox[1] + random.randint(-5, 5)
            
            draw.text((x, y), char_to_draw, font=font, fill=0)
            
            img_np = np.array(image)
            
            # Simulate low resolution of cropped FCF cells (pixel breaking)
            low_res = random.randint(20, 120)
            img_np = cv2.resize(img_np, (low_res, low_res), interpolation=cv2.INTER_AREA)
            img_np = cv2.resize(img_np, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
            
            # Apply anomalies
            img_np = apply_random_anomaly(img_np, severity)
            
            # Save using an ImageFolder compatible path: val/circularity/0001.png
            image_filename = os.path.join(split_dir, sym_name, f'{i:04d}.png')
            cv2.imwrite(image_filename, img_np)

    print(f"\n✅ Dataset generation complete! Ready for SwiftFormer.")
    print(f"Directory Structure:")
    print(f"  {output_dir}/")
    print(f"    ├── train/ ({num_train} images per class)")
    print(f"    └── val/ ({num_val} images per class)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate PyTorch ImageFolder dataset for SwiftFormer classification")
    parser.add_argument("--count", "-c", type=int, default=625, help="Number of samples PER SYMBOL to generate")
    parser.add_argument("--output", "-o", type=str, default="swiftformer_dataset", help="Output dataset directory")
    args = parser.parse_args()
    
    generate_classification_dataset(args.count, args.output)
