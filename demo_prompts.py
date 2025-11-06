"""
Demo script to show generated prompts and test basic functionality
"""

import json
import pandas as pd
from collections import Counter


def show_prompt_analysis():
    """Analyze and display the generated prompts"""
    print("=== CLIP Mushroom Classification - Prompt Analysis ===\n")
    
    # Load prompts
    with open('mushroom_prompts.json', 'r') as f:
        species_prompts = json.load(f)
    
    print(f"📊 DATASET STATISTICS")
    print("=" * 40)
    print(f"Total species: {len(species_prompts)}")
    
    # Analyze prompt counts
    prompt_counts = [len(prompts) for prompts in species_prompts.values()]
    total_prompts = sum(prompt_counts)
    avg_prompts = total_prompts / len(species_prompts)
    max_prompts = max(prompt_counts)
    min_prompts = min(prompt_counts)
    
    print(f"Total prompts generated: {total_prompts:,}")
    print(f"Average prompts per species: {avg_prompts:.1f}")
    print(f"Range: {min_prompts} - {max_prompts} prompts per species")
    
    # Show species with most prompts (likely have detailed knowledge)
    species_by_prompts = sorted(species_prompts.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"\n🍄 SPECIES WITH MOST DETAILED PROMPTS")
    print("=" * 40)
    for species, prompts in species_by_prompts[:10]:
        print(f"{len(prompts):2d} prompts: {species}")
    
    print(f"\n📝 SAMPLE PROMPTS FOR WELL-KNOWN SPECIES")
    print("=" * 40)
    
    # Show detailed prompts for well-known species
    showcase_species = [
        "Amanita muscaria",
        "Amanita phalloides", 
        "Boletus edulis",
        "Cantharellus cibarius",
        "Laetiporus sulphureus"
    ]
    
    for species in showcase_species:
        if species in species_prompts:
            prompts = species_prompts[species]
            print(f"\n{species} ({len(prompts)} prompts):")
            
            # Group prompts by type
            basic_prompts = [p for p in prompts if p.startswith(('a photo', 'a photograph', 'an image', 'a picture'))]
            common_name_prompts = [p for p in prompts if any(word in p.lower() for word in ['agaric', 'cap', 'bolete', 'chanterelle', 'chicken'])]
            descriptive_prompts = [p for p in prompts if 'with' in p or 'mushroom' in p.split()[-1:]]
            
            if basic_prompts:
                print(f"  Basic: {basic_prompts[0]}")
            if common_name_prompts:
                print(f"  Common: {common_name_prompts[0]}")
            if len(prompts) > 10:
                print(f"  Example: {prompts[8]}")
                print(f"  Context: {prompts[-3]}")
    
    # Analyze genus distribution
    print(f"\n🧬 GENUS DISTRIBUTION")
    print("=" * 40)
    
    genus_counts = Counter()
    for species in species_prompts.keys():
        genus = species.split()[0] if ' ' in species else species
        genus_counts[genus] += 1
    
    print("Top 10 most represented genera:")
    for genus, count in genus_counts.most_common(10):
        print(f"  {count:2d} species: {genus}")
    
    return species_prompts


def show_dataset_overview():
    """Show overview of the original dataset"""
    print(f"\n📊 ORIGINAL DATASET OVERVIEW")
    print("=" * 40)
    
    # Load training data
    try:
        train_df = pd.read_csv("/zfs/ai4good/datasets/mushroom/train.csv")
        
        species_counts = train_df['label'].value_counts()
        
        print(f"Training images: {len(train_df):,}")
        print(f"Unique species: {len(species_counts)}")
        print(f"Average images per species: {len(train_df) / len(species_counts):.1f}")
        
        print(f"\nMost photographed species:")
        for i, (species, count) in enumerate(species_counts.head(8).items(), 1):
            print(f"  {i:2d}. {species}: {count:,} images")
        
        print(f"\nLeast photographed species:")
        for i, (species, count) in enumerate(species_counts.tail(5).items(), 1):
            print(f"  {i:2d}. {species}: {count} images")
            
    except Exception as e:
        print(f"Could not load dataset: {e}")


def create_species_info_card(species_name, prompts):
    """Create an info card for a species"""
    print(f"\n🍄 {species_name.upper()}")
    print("=" * 50)
    
    # Parse genus and species
    parts = species_name.split()
    if len(parts) >= 2:
        print(f"Genus: {parts[0]}")
        print(f"Species: {parts[1]}")
    
    print(f"Generated prompts: {len(prompts)}")
    
    # Show different types of prompts
    print(f"\nPrompt Examples:")
    
    # Basic scientific name prompts
    basic = [p for p in prompts if p.startswith('a photo of') and species_name in p]
    if basic:
        print(f"  Scientific: {basic[0]}")
    
    # Common name prompts (if any)
    common_names = [p for p in prompts if not species_name in p and ('mushroom' in p or 'fungus' in p)]
    if common_names:
        print(f"  Common: {common_names[0]}")
    
    # Descriptive prompts
    descriptive = [p for p in prompts if 'with' in p]
    if descriptive:
        print(f"  Feature: {descriptive[0]}")
    
    # Contextual prompts
    contextual = [p for p in prompts if 'in' in p and ('habitat' in p or 'wild' in p or 'forest' in p)]
    if contextual:
        print(f"  Context: {contextual[0]}")
    
    print()


def main():
    """Main demo function"""
    
    # Check if prompts file exists
    try:
        species_prompts = show_prompt_analysis()
    except FileNotFoundError:
        print("❌ Error: mushroom_prompts.json not found!")
        print("Please run: python generate_clip_prompts.py")
        return
    
    # Show dataset overview
    show_dataset_overview()
    
    # Show detailed cards for interesting species
    print(f"\n🔍 DETAILED SPECIES EXAMPLES")
    print("=" * 50)
    
    interesting_species = [
        "Amanita muscaria",      # Well-known poisonous
        "Boletus edulis",        # Edible favorite  
        "Psilocybe cubensis",    # Psychoactive
        "Ganoderma tsugae",      # Medicinal
        "Trametes versicolor"    # Common bracket fungus
    ]
    
    for species in interesting_species:
        if species in species_prompts:
            create_species_info_card(species, species_prompts[species])
    
    print(f"\n✅ NEXT STEPS")
    print("=" * 40)
    print("1. Install CLIP: pip install git+https://github.com/openai/CLIP.git")
    print("2. Run demo: python clip_mushroom_classifier.py --demo")
    print("3. Or run full evaluation: python clip_mushroom_classifier.py")
    print(f"\nYour prompts are ready for {len(species_prompts)} species! 🎉")


if __name__ == "__main__":
    main()