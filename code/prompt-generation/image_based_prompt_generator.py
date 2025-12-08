"""
Enhanced CLIP-based Mushroom Classification with Image Analysis Prompt Generator

This script generates comprehensive text prompts for mushroom species by analyzing
actual image content using computer vision techniques, in addition to the 
existing knowledge-based approach.
"""

import pandas as pd
import json
import cv2
import numpy as np
from PIL import Image, ImageStat
from typing import Dict, List, Set, Tuple
import re
import os
from collections import Counter
import colorsys
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors

class ImageBasedMushroomPromptGenerator:
    def __init__(self):
        # Color name mapping for better descriptions
        self.color_names = {
            'red': [(255, 0, 0), (220, 20, 60), (178, 34, 34), (139, 0, 0)],
            'orange': [(255, 165, 0), (255, 140, 0), (255, 69, 0)],
            'yellow': [(255, 255, 0), (255, 215, 0), (218, 165, 32)],
            'green': [(0, 128, 0), (34, 139, 34), (0, 100, 0), (107, 142, 35)],
            'blue': [(0, 0, 255), (0, 0, 139), (25, 25, 112)],
            'purple': [(128, 0, 128), (75, 0, 130), (138, 43, 226)],
            'brown': [(139, 69, 19), (160, 82, 45), (210, 180, 140), (222, 184, 135)],
            'white': [(255, 255, 255), (248, 248, 255), (245, 245, 245)],
            'black': [(0, 0, 0), (47, 79, 79), (105, 105, 105)],
            'gray': [(128, 128, 128), (169, 169, 169), (192, 192, 192)],
            'cream': [(255, 253, 208), (245, 245, 220), (255, 228, 196)],
            'tan': [(210, 180, 140), (244, 164, 96), (188, 143, 143)]
        }
        
        # Texture descriptors based on image analysis
        self.texture_descriptors = {
            'smooth': 'smooth surface texture',
            'rough': 'rough, irregular surface',
            'bumpy': 'bumpy, warty texture',
            'scaly': 'scaly surface pattern',
            'spotted': 'spotted or dotted pattern',
            'striped': 'striped or banded pattern',
            'ridged': 'ridged or lined texture',
            'fuzzy': 'fuzzy, hairy appearance'
        }
        
        # Shape descriptors
        self.cap_shapes = {
            'round': 'round, circular cap',
            'oval': 'oval, elliptical cap',
            'flat': 'flat, plate-like cap',
            'convex': 'convex, dome-shaped cap',
            'funnel': 'funnel-shaped, concave cap',
            'irregular': 'irregular, asymmetric cap'
        }
        
    def extract_dominant_colors(self, image_path: str, n_colors: int = 3) -> List[str]:
        """Extract dominant colors from mushroom image using K-means clustering"""
        try:
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                return ['brown', 'white']
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Reshape image to be a list of pixels
            pixels = image.reshape((-1, 3))
            
            # Use K-means to find dominant colors
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get the dominant colors
            colors = kmeans.cluster_centers_.astype(int)
            
            # Convert RGB colors to color names
            color_names = []
            for color in colors:
                color_name = self._rgb_to_color_name(color)
                if color_name not in color_names:
                    color_names.append(color_name)
            
            return color_names[:3]  # Return top 3 colors
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return ['brown', 'white']
    
    def _rgb_to_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB values to closest color name"""
        min_distance = float('inf')
        closest_color = 'brown'
        
        for color_name, color_values in self.color_names.items():
            for color_rgb in color_values:
                # Calculate Euclidean distance
                distance = np.sqrt(sum((rgb[i] - color_rgb[i]) ** 2 for i in range(3)))
                if distance < min_distance:
                    min_distance = distance
                    closest_color = color_name
        
        return closest_color
    
    def analyze_texture(self, image_path: str) -> List[str]:
        """Analyze texture characteristics of the mushroom"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return ['smooth']
            
            # Calculate texture features
            textures = []
            
            # Edge detection for roughness
            edges = cv2.Canny(image, 50, 150)
            edge_density = np.mean(edges) / 255.0
            
            if edge_density > 0.15:
                textures.append('rough')
            elif edge_density < 0.05:
                textures.append('smooth')
            else:
                textures.append('moderately textured')
            
            # Variance for texture analysis
            variance = np.var(image)
            if variance > 2000:
                textures.append('highly detailed')
            elif variance > 1000:
                textures.append('textured')
            
            # Detect spots/circles using HoughCircles
            circles = cv2.HoughCircles(
                image, cv2.HOUGH_GRADIENT, 1, 20,
                param1=50, param2=30, minRadius=3, maxRadius=20
            )
            
            if circles is not None and len(circles[0]) > 5:
                textures.append('spotted')
            
            return textures[:2] if textures else ['smooth']
            
        except Exception as e:
            print(f"Error analyzing texture for {image_path}: {e}")
            return ['smooth']
    
    def analyze_shape(self, image_path: str) -> List[str]:
        """Analyze shape characteristics of the mushroom"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return ['round cap']
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Find contours
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            shapes = []
            
            if contours:
                # Find the largest contour (presumably the mushroom)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate shape characteristics
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                if area > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.85:
                        shapes.append('round cap')
                    elif circularity > 0.7:
                        shapes.append('oval cap')
                    else:
                        shapes.append('irregular cap')
                
                # Analyze aspect ratio
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = float(w) / h
                
                if aspect_ratio > 1.5:
                    shapes.append('wide, spreading form')
                elif aspect_ratio < 0.7:
                    shapes.append('tall, narrow form')
                else:
                    shapes.append('balanced proportions')
            
            return shapes[:2] if shapes else ['round cap']
            
        except Exception as e:
            print(f"Error analyzing shape for {image_path}: {e}")
            return ['round cap']
    
    def analyze_brightness_contrast(self, image_path: str) -> List[str]:
        """Analyze brightness and contrast characteristics"""
        try:
            image = Image.open(image_path)
            stat = ImageStat.Stat(image)
            
            descriptors = []
            
            # Average brightness
            avg_brightness = sum(stat.mean) / len(stat.mean)
            
            if avg_brightness > 200:
                descriptors.append('bright, well-lit')
            elif avg_brightness < 80:
                descriptors.append('dark, shadowy')
            else:
                descriptors.append('naturally lit')
            
            # Contrast (standard deviation)
            avg_contrast = sum(stat.stddev) / len(stat.stddev)
            
            if avg_contrast > 60:
                descriptors.append('high contrast')
            elif avg_contrast < 30:
                descriptors.append('low contrast')
            else:
                descriptors.append('balanced contrast')
            
            return descriptors
            
        except Exception as e:
            print(f"Error analyzing brightness/contrast for {image_path}: {e}")
            return ['naturally lit']
    
    def generate_image_based_prompts(self, image_path: str, species_name: str) -> List[str]:
        """Generate prompts based on actual image analysis"""
        if not os.path.exists(image_path):
            return []
        
        # Analyze image features
        colors = self.extract_dominant_colors(image_path)
        textures = self.analyze_texture(image_path)
        shapes = self.analyze_shape(image_path)
        lighting = self.analyze_brightness_contrast(image_path)
        
        prompts = []
        
        # Color-based prompts
        for color in colors:
            prompts.extend([
                f"{species_name} mushroom with {color} coloration",
                f"mushroom displaying {color} tones",
                f"{color}-colored {species_name} fungus",
                f"photograph of {color} {species_name}"
            ])
        
        # Texture-based prompts
        for texture in textures:
            prompts.extend([
                f"{species_name} showing {texture} surface",
                f"mushroom with {texture} cap texture",
                f"{texture} {species_name} specimen"
            ])
        
        # Shape-based prompts
        for shape in shapes:
            prompts.extend([
                f"{species_name} with {shape}",
                f"mushroom displaying {shape}",
                f"{species_name} showing {shape}"
            ])
        
        # Lighting/quality-based prompts
        for light in lighting:
            prompts.extend([
                f"{light} photograph of {species_name}",
                f"{species_name} in {light} conditions"
            ])
        
        # Combined feature prompts
        if len(colors) >= 2 and len(textures) >= 1:
            prompts.append(f"{species_name} with {colors[0]} and {colors[1]} coloring and {textures[0]} texture")
        
        if len(shapes) >= 1 and len(colors) >= 1:
            prompts.append(f"{colors[0]} {species_name} mushroom with {shapes[0]}")
        
        return prompts
    
    def process_images_for_species(self, df: pd.DataFrame, species_name: str, max_images: int = 10) -> List[str]:
        """Process multiple images for a single species to generate comprehensive prompts"""
        species_images = df[df['label'] == species_name]['image_path'].head(max_images)
        
        all_colors = []
        all_textures = []
        all_shapes = []
        all_prompts = []
        
        for image_path in species_images:
            # Convert from dataset path to actual path
            actual_path = image_path.replace('/kaggle/working/merged_dataset', 
                                           '/zfs/ai4good/datasets/mushroom/merged_dataset')
            
            if os.path.exists(actual_path):
                # Generate prompts for this specific image
                image_prompts = self.generate_image_based_prompts(actual_path, species_name)
                all_prompts.extend(image_prompts)
                
                # Collect features for aggregation
                colors = self.extract_dominant_colors(actual_path)
                textures = self.analyze_texture(actual_path)
                shapes = self.analyze_shape(actual_path)
                
                all_colors.extend(colors)
                all_textures.extend(textures)
                all_shapes.extend(shapes)
        
        # Generate aggregate prompts based on common features
        if all_colors:
            common_colors = [color for color, count in Counter(all_colors).most_common(3)]
            for color in common_colors:
                all_prompts.append(f"{species_name} typically displaying {color} coloration")
        
        if all_textures:
            common_textures = [texture for texture, count in Counter(all_textures).most_common(2)]
            for texture in common_textures:
                all_prompts.append(f"{species_name} commonly showing {texture}")
        
        # Remove duplicates
        return list(dict.fromkeys(all_prompts))
    
    def enhance_existing_prompts(self, csv_file: str, existing_prompts_file: str, 
                                output_file: str = "enhanced_mushroom_prompts.json"):
        """Enhance existing prompts with image-based analysis"""
        print("=== Enhancing Mushroom Prompts with Image Analysis ===")
        
        # Load existing prompts
        with open(existing_prompts_file, 'r') as f:
            existing_prompts = json.load(f)
        
        # Load dataset
        df = pd.read_csv(csv_file)
        
        enhanced_prompts = {}
        
        for i, species in enumerate(existing_prompts.keys()):
            print(f"Processing {i+1}/{len(existing_prompts)}: {species}")
            
            # Start with existing prompts
            species_prompts = existing_prompts[species].copy()
            
            # Add image-based prompts
            image_prompts = self.process_images_for_species(df, species, max_images=5)
            species_prompts.extend(image_prompts)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_prompts = []
            for prompt in species_prompts:
                if prompt not in seen:
                    seen.add(prompt)
                    unique_prompts.append(prompt)
            
            enhanced_prompts[species] = unique_prompts
        
        # Save enhanced prompts
        with open(output_file, 'w') as f:
            json.dump(enhanced_prompts, f, indent=2, ensure_ascii=False)
        
        # Print statistics
        total_old = sum(len(prompts) for prompts in existing_prompts.values())
        total_new = sum(len(prompts) for prompts in enhanced_prompts.values())
        
        print(f"\n=== Enhancement Complete ===")
        print(f"Original prompts: {total_old}")
        print(f"Enhanced prompts: {total_new}")
        print(f"Added {total_new - total_old} new image-based prompts")
        print(f"Saved to: {output_file}")


def main():
    """Main function to enhance prompts with image analysis"""
    generator = ImageBasedMushroomPromptGenerator()
    
    # File paths
    train_csv = "/zfs/ai4good/datasets/mushroom/train.csv"
    existing_prompts = "mushroom_prompts.json"
    enhanced_output = "enhanced_mushroom_prompts.json"
    
    # Enhance existing prompts with image analysis
    generator.enhance_existing_prompts(
        csv_file=train_csv,
        existing_prompts_file=existing_prompts,
        output_file=enhanced_output
    )
    
    print("\n=== Sample Enhanced Prompts ===")
    
    # Show sample of enhanced prompts
    with open(enhanced_output, 'r') as f:
        enhanced_prompts = json.load(f)
    
    sample_species = list(enhanced_prompts.keys())[:2]
    for species in sample_species:
        print(f"\n{species} ({len(enhanced_prompts[species])} total prompts):")
        # Show last few prompts (likely the new image-based ones)
        for prompt in enhanced_prompts[species][-5:]:
            print(f"  - {prompt}")


if __name__ == "__main__":
    main()