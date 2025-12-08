"""
CLIP-based Mushroom Classification Model

This script implements a CLIP-based approach for zero-shot mushroom classification
using pre-trained CLIP models and text prompts.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import os
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# For CLIP - we'll provide installation instructions
try:
    import clip
except ImportError:
    print("CLIP not installed. Please install with:")
    print("pip install torch torchvision")
    print("pip install git+https://github.com/openai/CLIP.git")
    exit(1)


class CLIPMushroomClassifier:
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """
        Initialize CLIP Mushroom Classifier
        
        Args:
            model_name: CLIP model variant to use
            device: Device to run model on (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load CLIP model
        print(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Initialize prompt templates
        self.species_prompts = {}
        self.text_features = {}
        
    def load_prompts(self, prompts_file: str):
        """Load species prompts from JSON file"""
        print(f"Loading prompts from {prompts_file}")
        with open(prompts_file, 'r') as f:
            self.species_prompts = json.load(f)
        print(f"Loaded prompts for {len(self.species_prompts)} species")
    
    def encode_text_prompts(self, species_list: Optional[List[str]] = None):
        """
        Encode all text prompts for the given species
        
        Args:
            species_list: List of species to encode. If None, encode all loaded species
        """
        if not self.species_prompts:
            raise ValueError("No prompts loaded. Call load_prompts() first.")
        
        species_to_encode = species_list or list(self.species_prompts.keys())
        print(f"Encoding text prompts for {len(species_to_encode)} species...")
        
        self.text_features = {}
        
        for species in tqdm(species_to_encode, desc="Encoding species prompts"):
            if species not in self.species_prompts:
                print(f"Warning: No prompts found for {species}")
                continue
                
            prompts = self.species_prompts[species]
            
            # Tokenize and encode prompts
            text_tokens = clip.tokenize(prompts).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                # Normalize features
                text_features = F.normalize(text_features, dim=-1)
                
                # Average all prompt embeddings for this species
                species_embedding = text_features.mean(dim=0)
                species_embedding = F.normalize(species_embedding, dim=-1)
                
                self.text_features[species] = species_embedding
        
        print(f"Encoded text features for {len(self.text_features)} species")
    
    def encode_image(self, image_path: str) -> torch.Tensor:
        """
        Encode a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized image embedding
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Encode image
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = F.normalize(image_features, dim=-1)
            
            return image_features.squeeze(0)
        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def predict_single(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict species for a single image
        
        Args:
            image_path: Path to the image
            top_k: Number of top predictions to return
            
        Returns:
            List of (species, confidence_score) tuples
        """
        if not self.text_features:
            raise ValueError("No text features encoded. Call encode_text_prompts() first.")
        
        # Encode image
        image_features = self.encode_image(image_path)
        if image_features is None:
            return []
        
        # Calculate similarities with all species
        similarities = {}
        for species, text_embedding in self.text_features.items():
            similarity = torch.cosine_similarity(
                image_features.unsqueeze(0), 
                text_embedding.unsqueeze(0)
            ).item()
            similarities[species] = similarity
        
        # Sort by similarity and return top-k
        sorted_predictions = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions[:top_k]
    
    def predict_batch(self, image_paths: List[str]) -> List[str]:
        """
        Predict species for a batch of images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of predicted species names
        """
        predictions = []
        
        for path in tqdm(image_paths, desc="Predicting images"):
            top_prediction = self.predict_single(path, top_k=1)
            if top_prediction:
                predictions.append(top_prediction[0][0])
            else:
                predictions.append("Unknown")
        
        return predictions
    
    def evaluate_dataset(self, csv_file: str, image_root: str = "") -> Dict:
        """
        Evaluate model on a dataset
        
        Args:
            csv_file: Path to CSV file with image_path and label columns
            image_root: Root directory to prepend to image paths if needed
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating on dataset: {csv_file}")
        
        # Load dataset
        df = pd.read_csv(csv_file)
        
        # Prepare image paths
        image_paths = df['image_path'].tolist()
        if image_root:
            image_paths = [os.path.join(image_root, path) for path in image_paths]
        else:
            # Convert kaggle paths to local paths
            image_paths = [path.replace('/kaggle/working/', '/zfs/ai4good/datasets/mushroom/') 
                          for path in image_paths]
        
        true_labels = df['label'].tolist()
        
        # Make predictions
        print("Making predictions...")
        predicted_labels = self.predict_batch(image_paths)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Get unique labels for classification report
        unique_labels = sorted(list(set(true_labels + predicted_labels)))
        
        results = {
            'accuracy': accuracy,
            'predictions': predicted_labels,
            'true_labels': true_labels
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        
        return results
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to file"""
        # Prepare results for saving
        save_data = {
            'accuracy': results['accuracy'],
            'predictions': results['predictions'],
            'true_labels': results['true_labels']
        }
        
        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Results saved to {output_file}")


def demo_single_prediction():
    """Demo function for single image prediction"""
    print("=== CLIP Mushroom Classification Demo ===\n")
    
    # Check if prompts file exists
    if not os.path.exists('mushroom_prompts.json'):
        print("Error: mushroom_prompts.json not found!")
        print("Please run generate_clip_prompts.py first")
        return
    
    # Initialize classifier
    classifier = CLIPMushroomClassifier(model_name="ViT-B/32")
    
    # Load prompts
    classifier.load_prompts('mushroom_prompts.json')
    
    # For demo, use subset of species
    test_csv = "/zfs/ai4good/datasets/mushroom/test.csv"
    df = pd.read_csv(test_csv)
    demo_species = df['label'].unique()[:20]  # First 20 species for demo
    
    print(f"Encoding prompts for {len(demo_species)} species...")
    classifier.encode_text_prompts(demo_species.tolist())
    
    # Test on a few images
    sample_images = df[df['label'].isin(demo_species)].head(5)
    
    print("\nTesting predictions:")
    print("-" * 50)
    
    for _, row in sample_images.iterrows():
        image_path = row['image_path'].replace('/kaggle/working/', '/zfs/ai4good/datasets/mushroom/')
        true_label = row['label']
        
        if os.path.exists(image_path):
            predictions = classifier.predict_single(image_path, top_k=3)
            
            print(f"Image: {os.path.basename(image_path)}")
            print(f"True: {true_label}")
            
            if predictions:
                top_pred = predictions[0][0]
                confidence = predictions[0][1]
                
                print(f"Predicted: {top_pred} (confidence: {confidence:.3f})")
                print("Top 3:")
                for i, (pred, conf) in enumerate(predictions, 1):
                    print(f"  {i}. {pred}: {conf:.3f}")
                
                if top_pred == true_label:
                    print("✓ CORRECT")
                else:
                    print("✗ INCORRECT")
            else:
                print("❌ Prediction failed")
            
            print("-" * 50)


def main():
    """Main function - can be used for full evaluation"""
    parser = argparse.ArgumentParser(description='CLIP-based Mushroom Classification')
    parser.add_argument('--prompts', default='mushroom_prompts.json', help='Path to prompts JSON file')
    parser.add_argument('--test_csv', default='/zfs/ai4good/datasets/mushroom/test.csv', help='Path to test CSV file')
    parser.add_argument('--model', default='ViT-B/32', help='CLIP model variant')
    parser.add_argument('--output', default='clip_results.json', help='Output file for results')
    parser.add_argument('--demo', action='store_true', help='Run demo mode')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_single_prediction()
        return
    
    # Full evaluation
    classifier = CLIPMushroomClassifier(model_name=args.model)
    classifier.load_prompts(args.prompts)
    classifier.encode_text_prompts()
    
    results = classifier.evaluate_dataset(args.test_csv)
    classifier.save_results(results, args.output)
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()