"""
CLIP-based Mushroom Classification with Text Prompts Generator

This script generates comprehensive text prompts for mushroom species
to be used with CLIP models for zero-shot classification.
"""

import pandas as pd
import json
from typing import Dict, List, Set
import re

class MushroomPromptGenerator:
    def __init__(self):
        # Mushroom characteristic templates
        self.mushroom_characteristics = {
            # Morphological features
            'cap': ['cap', 'pileus', 'hat', 'top'],
            'stem': ['stem', 'stipe', 'stalk'],
            'gills': ['gills', 'lamellae', 'underneath'],
            'spores': ['spores', 'spore print'],
            'ring': ['ring', 'annulus'],
            'volva': ['volva', 'cup', 'base'],
            
            # Colors commonly found in mushrooms
            'colors': [
                'white', 'cream', 'yellow', 'orange', 'red', 'pink', 'purple',
                'brown', 'tan', 'black', 'gray', 'blue', 'green', 'golden'
            ],
            
            # Textures and patterns
            'textures': [
                'smooth', 'rough', 'scaly', 'fibrous', 'slimy', 'dry', 'sticky',
                'velvety', 'hairy', 'warty', 'spotted', 'striped'
            ],
            
            # Habitats
            'habitats': [
                'forest', 'woodland', 'deciduous', 'coniferous', 'oak', 'pine',
                'birch', 'dead wood', 'soil', 'grass', 'moss', 'rotting wood'
            ],
            
            # Growth patterns
            'growth': [
                'solitary', 'clustered', 'scattered', 'fairy ring', 'bracket',
                'shelf-like', 'parasitic', 'saprobic'
            ]
        }
        
        # Species-specific knowledge base
        self.species_knowledge = self._build_species_knowledge()
    
    def _build_species_knowledge(self) -> Dict[str, Dict]:
        """Build comprehensive knowledge base for mushroom species"""
        return {
            # Amanita species - deadly and edible varieties
            'Amanita muscaria': {
                'common_names': ['fly agaric', 'fly amanita', 'red-capped amanita'],
                'key_features': ['bright red cap with white warts', 'white stem with ring', 'bulbous base with scales', 'white gills'],
                'colors': ['bright red cap', 'white spots', 'white stem', 'cream gills'],
                'habitat': ['birch and pine forests', 'mycorrhizal with conifers', 'acidic soil'],
                'toxicity': 'poisonous psychoactive',
                'size': 'medium to large',
                'distinctive': ['iconic red and white mushroom', 'fairy tale mushroom appearance', 'Christmas ornament colors']
            },
            'Amanita phalloides': {
                'common_names': ['death cap', 'deadly amanita'],
                'key_features': ['pale greenish cap', 'white gills free from stem', 'white ring on stem', 'bulbous base with cup-like volva'],
                'colors': ['pale green to yellowish cap', 'white gills', 'white stem'],
                'habitat': ['oak and chestnut trees', 'imported with non-native trees', 'deciduous forests'],
                'toxicity': 'extremely deadly poisonous',
                'size': 'medium to large',
                'distinctive': ['most dangerous mushroom', 'responsible for most mushroom poisoning deaths', 'deceptively mild appearance']
            },
            'Amanita rubescens': {
                'common_names': ['blusher', 'blushing amanita'],
                'key_features': ['reddish-brown cap with white patches', 'bruises pinkish-red when cut', 'white ring on stem', 'bulbous base'],
                'colors': ['reddish-brown cap', 'pinkish blush when bruised', 'white stem'],
                'habitat': ['mixed deciduous and coniferous forests', 'acidic soil', 'summer to fall'],
                'toxicity': 'edible when thoroughly cooked',
                'size': 'medium',
                'distinctive': ['turns pink when damaged', 'flesh changes color', 'requires cooking to be safe']
            },
            
            # Boletus species - prized edibles
            'Boletus edulis': {
                'common_names': ['porcini', 'king bolete', 'penny bun', 'cep'],
                'key_features': ['brown cap with white margin when young', 'thick white stem with fine network', 'white pores that don\'t stain', 'firm white flesh'],
                'colors': ['brown to chestnut cap', 'white to cream pores', 'white stem with darker network'],
                'habitat': ['coniferous and deciduous forests', 'mycorrhizal with trees', 'well-drained soil'],
                'toxicity': 'choice edible',
                'size': 'large',
                'distinctive': ['prized culinary mushroom', 'nutty flavor', 'firm texture', 'expensive commercial mushroom']
            },
            'Boletus reticulatus': {
                'common_names': ['summer king bolete', 'reticuled bolete'],
                'key_features': ['light brown cap with cracked surface', 'white stem with raised network', 'white pores', 'firm flesh'],
                'colors': ['light brown cap', 'white pores', 'white stem'],
                'habitat': ['oak and beech forests', 'calcareous soil', 'summer fruiting'],
                'toxicity': 'excellent edible',
                'size': 'large',
                'distinctive': ['early summer fruiting', 'cracked cap surface', 'strong reticulation on stem']
            },
            
            # Cantharellus species - golden treasures
            'Cantharellus cibarius': {
                'common_names': ['golden chanterelle', 'girolle', 'egg mushroom'],
                'key_features': ['bright golden-yellow funnel shape', 'forked ridges instead of true gills', 'solid stem', 'apricot scent'],
                'colors': ['bright golden-yellow', 'egg-yolk yellow', 'apricot orange'],
                'habitat': ['mossy coniferous forests', 'acidic soil', 'mycorrhizal with conifers'],
                'toxicity': 'choice edible',
                'size': 'medium',
                'distinctive': ['apricot fragrance', 'funnel-shaped cap', 'false gills are ridges', 'golden treasure of forest']
            },
            
            # Psilocybe species - psychoactive
            'Psilocybe cubensis': {
                'common_names': ['golden teacher', 'magic mushroom', 'golden caps', 'cubes'],
                'key_features': ['golden-brown cap with dark spores', 'bruises blue when damaged', 'purple-black spore print', 'ring on stem'],
                'colors': ['golden-brown to caramel cap', 'white stem', 'blue bruising', 'dark purple spores'],
                'habitat': ['cattle dung', 'rich grasslands', 'subtropical regions', 'cultivated indoors'],
                'toxicity': 'psychoactive hallucinogenic',
                'size': 'medium',
                'distinctive': ['blue bruising reaction', 'grows on dung', 'purple-black spores', 'consciousness-altering effects']
            },
            'Psilocybe azurescens': {
                'common_names': ['flying saucer mushroom', 'blue angels', 'azzies'],
                'key_features': ['caramel-colored cap with wavy margins', 'strong blue bruising', 'whitish stem', 'dark spores'],
                'colors': ['caramel to tawny cap', 'whitish stem', 'intense blue bruising'],
                'habitat': ['wood chips', 'coastal areas', 'cool climates', 'wood-loving'],
                'toxicity': 'very potent psychoactive',
                'size': 'medium',
                'distinctive': ['most potent psilocybe species', 'wavy cap margins', 'wood-loving habitat', 'intense blue staining']
            },
            
            # Ganoderma species - medicinal
            'Ganoderma tsugae': {
                'common_names': ['reishi', 'lingzhi', 'varnish shelf', 'hemlock reishi'],
                'key_features': ['shiny reddish-brown lacquered surface', 'woody bracket growth', 'white to brown pore surface', 'no stem or lateral stem'],
                'colors': ['reddish-brown lacquered cap', 'white to brown pores', 'mahogany shine'],
                'habitat': ['hemlock and other conifers', 'dead and dying trees', 'year-round bracket'],
                'toxicity': 'medicinal not culinary',
                'size': 'medium to large',
                'distinctive': ['glossy varnished appearance', 'medicinal properties', 'bitter taste', 'long-lasting bracket fungus']
            },
            
            # Laetiporus species - shelf fungus
            'Laetiporus sulphureus': {
                'common_names': ['chicken of the woods', 'sulfur shelf', 'chicken mushroom', 'chicken fungus'],
                'key_features': ['bright yellow and orange overlapping brackets', 'sulfur-yellow pore surface', 'no stem', 'soft when young'],
                'colors': ['bright orange caps', 'sulfur-yellow edges', 'yellow pore surface'],
                'habitat': ['oak and hardwood trees', 'dead and dying wood', 'large clusters on trunks'],
                'toxicity': 'edible when young and properly cooked',
                'size': 'large clusters',
                'distinctive': ['chicken-like texture and flavor', 'brilliant orange and yellow colors', 'large shelf-like growth', 'vegetarian chicken substitute']
            },
            
            # Trametes species - bracket fungi
            'Trametes versicolor': {
                'common_names': ['turkey tail', 'many-zoned polypore', 'cloud mushroom'],
                'key_features': ['thin bracket with concentric zones of color', 'velvety surface texture', 'white pore surface', 'no stem'],
                'colors': ['multiple zones of brown, tan, blue, green', 'white pore surface', 'colorful banded appearance'],
                'habitat': ['dead hardwood logs and branches', 'year-round', 'overlapping clusters'],
                'toxicity': 'medicinal not edible',
                'size': 'small to medium brackets',
                'distinctive': ['turkey tail appearance', 'colorful concentric zones', 'immune-boosting properties', 'very common decomposer']
            },
            
            # Lycoperdon species - puffballs
            'Lycoperdon perlatum': {
                'common_names': ['common puffball', 'gem-studded puffball', 'warted puffball'],
                'key_features': ['white round ball with small spines', 'opens at top to release spores', 'white flesh when young', 'attached to ground by root-like base'],
                'colors': ['white when young', 'brown when mature', 'white flesh turning yellow-green'],
                'habitat': ['grassy areas', 'open woodlands', 'disturbed soil', 'late summer to fall'],
                'toxicity': 'edible when young and white inside',
                'size': 'small to medium',
                'distinctive': ['pear-shaped puffball', 'spiny surface texture', 'puffs spores when mature', 'white flesh essential for edibility']
            },
            
            # Additional comprehensive species knowledge...
            'Armillaria mellea': {
                'common_names': ['honey mushroom', 'honey fungus', 'bootlace fungus'],
                'key_features': ['honey-colored caps with dark scales', 'white ring on stem', 'black rhizomorphs', 'grows in clusters'],
                'colors': ['honey to golden-brown caps', 'white to cream gills', 'brownish stem'],
                'habitat': ['parasitic on trees', 'dead and dying wood', 'large clusters at tree base'],
                'toxicity': 'edible when cooked but can cause gastric upset',
                'size': 'medium',
                'distinctive': ['honey coloration', 'black bootlace rhizomorphs', 'tree parasite', 'bioluminescent mycelium']
            },
            
            # Add more species with detailed information...
        }
    
    def parse_species_name(self, species_name: str) -> Dict[str, str]:
        """Parse scientific name into genus and species"""
        parts = species_name.strip().split()
        if len(parts) >= 2:
            return {
                'genus': parts[0],
                'species': parts[1],
                'full_name': species_name
            }
        return {'genus': species_name, 'species': '', 'full_name': species_name}
    
    def generate_basic_prompts(self, species_name: str) -> List[str]:
        """Generate basic prompt variations for a species"""
        parsed = self.parse_species_name(species_name)
        
        prompts = [
            f"a photo of {species_name}",
            f"a photograph of {species_name}",
            f"an image of {species_name}",
            f"a picture of {species_name}",
            f"{species_name} mushroom",
            f"{species_name} fungus",
            f"the mushroom {species_name}",
            f"the fungus {species_name}",
        ]
        
        # Add genus-level prompts
        if parsed['genus']:
            prompts.extend([
                f"a {parsed['genus']} mushroom",
                f"a {parsed['genus']} species",
                f"{parsed['genus']} fungus"
            ])
        
        return prompts
    
    def generate_descriptive_prompts(self, species_name: str) -> List[str]:
        """Generate rich descriptive prompts based on species knowledge"""
        prompts = []
        
        # Check if we have specific knowledge about this species
        if species_name in self.species_knowledge:
            knowledge = self.species_knowledge[species_name]
            
            # Common name prompts with descriptive context
            if 'common_names' in knowledge:
                for common_name in knowledge['common_names']:
                    prompts.extend([
                        f"a photo of {common_name}",
                        f"{common_name} mushroom in natural habitat",
                        f"the distinctive {common_name} fungus",
                        f"wild {common_name} growing in forest"
                    ])
            
            # Detailed feature-based prompts
            if 'key_features' in knowledge:
                for feature in knowledge['key_features']:
                    prompts.extend([
                        f"a mushroom with {feature}",
                        f"{species_name} showing {feature}",
                        f"close-up view of mushroom with {feature}"
                    ])
            
            # Rich color descriptions
            if 'colors' in knowledge:
                for color in knowledge['colors']:
                    prompts.extend([
                        f"mushroom displaying {color}",
                        f"{species_name} with distinctive {color}",
                        f"fungus showing {color} coloration"
                    ])
            
            # Habitat and ecological prompts
            if 'habitat' in knowledge:
                for habitat in knowledge['habitat']:
                    prompts.extend([
                        f"{species_name} growing naturally in {habitat}",
                        f"wild mushroom thriving in {habitat}",
                        f"fungus found growing on {habitat}",
                        f"mushroom habitat showing {habitat} environment"
                    ])
            
            # Distinctive characteristics
            if 'distinctive' in knowledge:
                for distinctive in knowledge['distinctive']:
                    prompts.extend([
                        f"mushroom known for {distinctive}",
                        f"{species_name} displaying {distinctive}",
                        f"fungus with characteristic {distinctive}"
                    ])
            
            # Size and growth pattern
            if 'size' in knowledge:
                prompts.extend([
                    f"{knowledge['size']} sized {species_name} mushroom",
                    f"mature {knowledge['size']} {species_name}",
                    f"{species_name} at full {knowledge['size']} size"
                ])
            
            # Safety and identification context
            if 'toxicity' in knowledge:
                toxicity = knowledge['toxicity']
                prompts.extend([
                    f"{species_name}, a {toxicity} mushroom species",
                    f"{toxicity} fungus {species_name}",
                    f"mushroom classified as {toxicity}"
                ])
        
        else:
            # Generate genus-level descriptive prompts for unknown species
            parsed = self.parse_species_name(species_name)
            genus = parsed['genus']
            
            # Add genus-specific contextual information
            genus_contexts = {
                'Amanita': ['deadly poisonous group', 'white gills and ring', 'bulbous base with volva'],
                'Boletus': ['pores instead of gills', 'often edible', 'thick fleshy stem'],
                'Psilocybe': ['psychoactive properties', 'bruises blue', 'dark spore print'],
                'Cantharellus': ['golden chanterelle family', 'forked ridges not gills', 'funnel shaped'],
                'Trametes': ['bracket fungus', 'grows on wood', 'leathery texture'],
                'Ganoderma': ['woody bracket fungus', 'medicinal properties', 'shiny surface'],
                'Armillaria': ['honey mushroom group', 'grows in clusters', 'parasitic on trees']
            }
            
            if genus in genus_contexts:
                for context in genus_contexts[genus]:
                    prompts.extend([
                        f"{species_name} mushroom with {context}",
                        f"{genus} species showing {context}",
                        f"fungus from {genus} genus with {context}"
                    ])
        
        return prompts
    
    def generate_contextual_prompts(self, species_name: str) -> List[str]:
        """Generate rich contextual prompts for natural settings and identification"""
        prompts = [
            # Natural habitat contexts
            f"{species_name} mushroom photographed in its natural forest habitat",
            f"wild {species_name} growing undisturbed in woodland environment",
            f"{species_name} fungus thriving in typical ecological niche",
            f"field photograph of {species_name} in natural setting",
            
            # Identification and field guide contexts
            f"field guide photograph of {species_name} for identification",
            f"mycological specimen of {species_name} showing key features",
            f"detailed identification photo of {species_name}",
            f"scientific documentation of {species_name} mushroom",
            
            # Growth stage and condition contexts
            f"mature fruiting body of {species_name}",
            f"fresh young {species_name} in prime condition",
            f"fully developed {species_name} showing all characteristics",
            f"pristine {species_name} specimen for study",
            
            # Photographic perspectives
            f"top-down view of {species_name} cap showing surface details",
            f"side profile of {species_name} revealing stem and cap structure",
            f"close-up macro photography of {species_name} textures",
            f"detailed view showing {species_name} gills and stem features",
            
            # Growth patterns and arrangements
            f"solitary {species_name} mushroom growing alone",
            f"cluster of {species_name} growing together in group",
            f"fairy ring formation of {species_name} mushrooms",
            f"multiple {species_name} specimens at different growth stages",
            
            # Substrate and ecological contexts
            f"{species_name} growing from decomposing organic matter",
            f"{species_name} emerging from forest floor substrate",
            f"{species_name} fruiting from dead wood and logs",
            f"{species_name} mycorrhizal association with tree roots",
            
            # Seasonal and environmental contexts
            f"{species_name} appearing in autumn forest conditions",
            f"{species_name} thriving in humid woodland environment",
            f"{species_name} photographed during peak fruiting season",
            f"{species_name} in typical weather conditions for growth"
        ]
        
        return prompts
    
    def generate_morphological_prompts(self, species_name: str) -> List[str]:
        """Generate scientific morphological prompts"""
        prompts = []
        
        # Parse genus and species for taxonomic prompts
        parsed = self.parse_species_name(species_name)
        genus = parsed['genus']
        
        # Generic morphological features
        morphological_terms = [
            f"{species_name} showing cap, stem and gill structure",
            f"mushroom fruiting body of {species_name}",
            f"{species_name} displaying pileus and stipe morphology",
            f"basidiomycete {species_name} with visible spore-bearing structures",
            f"{species_name} fungus showing hymenium and hymenophore",
            f"sporocarp of {species_name} in natural state"
        ]
        prompts.extend(morphological_terms)
        
        # Genus-specific morphological features
        genus_morphology = {
            'Amanita': [
                f"{species_name} amanita showing universal veil remnants",
                f"{species_name} with characteristic bulbous stem base and ring",
                f"amanita {species_name} displaying free gills and white spores"
            ],
            'Boletus': [
                f"{species_name} bolete showing pore surface instead of gills",
                f"{species_name} with characteristic thick fleshy stem",
                f"bolete {species_name} displaying tube layer and pores"
            ],
            'Cantharellus': [
                f"{species_name} chanterelle with forked ridges and decurrent attachment",
                f"{species_name} showing false gills and funnel-shaped morphology"
            ],
            'Psilocybe': [
                f"{species_name} with dark purple-brown spore print",
                f"{species_name} showing blue bruising oxidation reaction"
            ],
            'Trametes': [
                f"{species_name} bracket fungus with leathery consistency",
                f"{species_name} polypore showing white pore surface"
            ],
            'Ganoderma': [
                f"{species_name} woody bracket with lacquered surface texture",
                f"{species_name} perennial polypore with resinous appearance"
            ]
        }
        
        if genus in genus_morphology:
            prompts.extend(genus_morphology[genus])
        
        return prompts
    
    def generate_all_prompts(self, species_name: str) -> List[str]:
        """Generate comprehensive prompt variations for a species"""
        all_prompts = []
        
        # Basic prompts
        all_prompts.extend(self.generate_basic_prompts(species_name))
        
        # Rich descriptive prompts
        all_prompts.extend(self.generate_descriptive_prompts(species_name))
        
        # Detailed contextual prompts
        all_prompts.extend(self.generate_contextual_prompts(species_name))
        
        # Scientific morphological prompts
        all_prompts.extend(self.generate_morphological_prompts(species_name))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_prompts = []
        for prompt in all_prompts:
            if prompt not in seen:
                seen.add(prompt)
                unique_prompts.append(prompt)
        
        return unique_prompts
    
    def process_dataset(self, csv_file: str) -> Dict[str, List[str]]:
        """Process dataset and generate prompts for all species"""
        print(f"Processing {csv_file}...")
        
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Get unique species
        unique_species = df['label'].unique()
        print(f"Found {len(unique_species)} unique species")
        
        # Generate prompts for each species
        species_prompts = {}
        for i, species in enumerate(unique_species):
            if (i + 1) % 20 == 0:  # Progress update every 20 species
                print(f"Processing {i+1}/{len(unique_species)}: {species}")
            prompts = self.generate_all_prompts(species)
            species_prompts[species] = prompts
        
        return species_prompts
    
    def save_prompts(self, species_prompts: Dict[str, List[str]], output_file: str):
        """Save prompts to JSON file"""
        print(f"Saving prompts to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(species_prompts, f, indent=2, ensure_ascii=False)
        
        print(f"Saved prompts for {len(species_prompts)} species")
        
        # Print statistics
        total_prompts = sum(len(prompts) for prompts in species_prompts.values())
        avg_prompts = total_prompts / len(species_prompts)
        print(f"Total prompts generated: {total_prompts}")
        print(f"Average prompts per species: {avg_prompts:.1f}")


def main():
    """Main function to generate prompts for the mushroom dataset"""
    generator = MushroomPromptGenerator()
    
    # File paths - update these to point to your dataset
    train_csv = "/zfs/ai4good/datasets/mushroom/train.csv"
    prompts_json = "mushroom_prompts.json"
    
    # Generate prompts
    print("=== Generating Mushroom CLIP Prompts ===")
    species_prompts = generator.process_dataset(train_csv)
    
    # Save prompts
    generator.save_prompts(species_prompts, prompts_json)
    
    print("\n=== Process Complete ===")
    print("Files generated:")
    print(f"1. {prompts_json} - All prompts for each species")
    
    # Show sample prompts
    print("\n=== Sample Prompts ===")
    sample_species = list(species_prompts.keys())[:3]
    for species in sample_species:
        print(f"\n{species}:")
        for prompt in species_prompts[species][:5]:  # Show first 5 prompts
            print(f"  - {prompt}")
        print(f"  ... ({len(species_prompts[species])} total prompts)")


if __name__ == "__main__":
    main()