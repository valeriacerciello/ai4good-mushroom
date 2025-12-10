"""
How to use?:
1. if you try to predict a single image then try
python Lora_inference.py --checkpoint best_model.pth --image mushroom.jpg
2. if you try to predict all image that in a folder
python Lora_inference.py --checkpoint best_model.pth --folder ./mushroom_photos
3. if you want to show more predictions
python Lora_inference.py --checkpoint best_model.pth --image mushroom.jpg --top_k z
where z is number of predicts that you want to show.
4.also, if you want to specify output file
python Lora_inference.py --checkpoint best_model.pth --image mushroom.jpg --output result.json

"""
import os
import json
import torch
import argparse
from PIL import Image
from torchvision import transforms
import open_clip
from peft import LoraConfig, get_peft_model
import glob

# Configuration (same as training)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.005

Lora_target_modules = [
    "visual.transformer.resblocks.10.attn.in_proj",
    "visual.transformer.resblocks.10.attn.out_proj",
    "visual.transformer.resblocks.11.attn.in_proj",
    "visual.transformer.resblocks.11.attn.out_proj",
    "visual.transformer.resblocks.10.mlp.c_fc",
    "visual.transformer.resblocks.10.mlp.c_proj",
    "visual.transformer.resblocks.11.mlp.c_fc",
    "visual.transformer.resblocks.11.mlp.c_proj",
]

# Image preprocessing
def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])
    ])

# Load trained model
def load_model(checkpoint_path):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE,weights_only=False)
    
    # Get class information
    class_names = checkpoint['classes']
    
    model_name = "ViT-B-32-quickgelu"
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained="laion400m_e32")
    
    # LoRA configuration 
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=Lora_target_modules,
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(DEVICE)
    model.eval()
    
    text_features = checkpoint['text_features'].to(DEVICE)
    
    return model, text_features, class_names

# Prediction function
def predict_image(model, text_features, class_names, image_path, top_k=5):
    """Predict single image"""
    transform = get_transform()
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_features.T
        probs = torch.softmax(logits / TEMPERATURE, dim=-1)
    
    # Get top-k predictions
    top_probs, top_indices = probs.topk(min(top_k, len(class_names)), dim=-1)
    
    # Format results
    results = []
    for i in range(top_probs.shape[1]):
        results.append({
            'class': class_names[top_indices[0, i].item()],
            'probability': top_probs[0, i].item()
        })
    
    return results

# Batch prediction
def predict_folder(model, text_features, class_names, folder_path, top_k=5):
    """Predict all images in folder"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    
    all_results = {}
    for img_path in image_paths:
        try:
            results = predict_image(model, text_features, class_names, img_path, top_k)
            all_results[img_path] = results
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return all_results

# Main function
def main():
    parser = argparse.ArgumentParser(description='Mushroom classification inference')
    parser.add_argument('--checkpoint', '-c', required=True, help='Model checkpoint path')
    parser.add_argument('--image', '-i', help='Single image path')
    parser.add_argument('--folder', '-f', help='Folder containing images')
    parser.add_argument('--top_k', '-k', type=int, default=3, help='Number of top predictions to show')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.checkpoint}")
    model, text_features, class_names = load_model(args.checkpoint)
    print(f"Number of classes: {len(class_names)}")
    
    # Perform prediction
    if args.image:
        # Single image prediction
        print(f"\nPredicting image: {args.image}")
        results = predict_image(model, text_features, class_names, args.image, args.top_k)
        
        # Display results
        print("\nPrediction results:")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['class']}: {r['probability']:.1%}")
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({args.image: results}, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    elif args.folder:
        # Batch prediction for folder
        print(f"\nPredicting folder: {args.folder}")
        all_results = predict_folder(model, text_features, class_names, args.folder, args.top_k)
        
        # Display statistics
        print(f"\nCompleted: {len(all_results)} images processed")
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Results saved to: {args.output}")
        else:
            # Save to default file
            output_file = os.path.join(args.folder, "predictions.json")
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Results saved to: {output_file}")
    
    else:
        print("Please provide either --image or --folder argument")

# For direct import usage
def quick_predict(checkpoint_path, image_path, top_k=3):
    """Quick prediction function for programmatic use"""
    model, text_features, class_names = load_model(checkpoint_path)
    results = predict_image(model, text_features, class_names, image_path, top_k)
    return results

if __name__ == "__main__":
    main()
