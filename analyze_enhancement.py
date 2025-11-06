"""
Analysis script to compare original vs enhanced prompts
and show examples of image-based prompt generation
"""

import json

def analyze_prompt_enhancement():
    # Load both files
    with open('mushroom_prompts.json', 'r') as f:
        original_prompts = json.load(f)
    
    with open('enhanced_mushroom_prompts.json', 'r') as f:
        enhanced_prompts = json.load(f)
    
    print("=== PROMPT ENHANCEMENT ANALYSIS ===\n")
    
    # Overall statistics
    orig_total = sum(len(prompts) for prompts in original_prompts.values())
    enh_total = sum(len(prompts) for prompts in enhanced_prompts.values())
    
    print(f"Original prompts: {orig_total}")
    print(f"Enhanced prompts: {enh_total}")
    print(f"New image-based prompts added: {enh_total - orig_total}")
    print(f"Improvement: {((enh_total - orig_total) / orig_total * 100):.1f}% increase\n")
    
    # Sample species analysis
    sample_species = ['Amanita muscaria', 'Boletus edulis', 'Cantharellus cibarius']
    
    for species in sample_species:
        if species in original_prompts and species in enhanced_prompts:
            orig_count = len(original_prompts[species])
            enh_count = len(enhanced_prompts[species])
            new_prompts = enh_count - orig_count
            
            print(f"=== {species} ===")
            print(f"Original: {orig_count} prompts")
            print(f"Enhanced: {enh_count} prompts")
            print(f"New image-based: {new_prompts} prompts")
            
            # Show the new image-based prompts
            print("New image-based prompts:")
            image_based_prompts = enhanced_prompts[species][orig_count:]
            for i, prompt in enumerate(image_based_prompts[:8]):  # Show first 8 new ones
                print(f"  {i+1}. {prompt}")
            if len(image_based_prompts) > 8:
                print(f"  ... and {len(image_based_prompts) - 8} more")
            print()

if __name__ == "__main__":
    analyze_prompt_enhancement()