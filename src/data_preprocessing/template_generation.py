from itertools import product, combinations
import re
import argparse
import json
from pathlib import Path

base_template1 = "You are {gender} {nationality} {person_description} in the age range {age_range} responding to the following survey questions."

# Cultural prompting template - explicit cultural identity specification for improved cultural alignment
base_template_cultural = "You are {gender} {nationality} {person_description} in the age range {age_range}{cultural_identity}. Respond to the following survey questions from this cultural perspective."

person_descriptions = ["an average human being", "a typical human being", "an average person", "a typical person", "a person", "an average individual", "an individual", "a world citizen"]

genders = ["female", "male"]

nationalities = ["Spanish", "Greek", "English"]

age_ranges = ['15-29', '30-49', '50+']

# Cultural prompting options - cultural identities to improve cultural alignment
# These specify cultural backgrounds, values, and contexts
cultural_identities = [
    # Spanish cultural identities
    "raised in a traditional Spanish family",
    "with strong Mediterranean cultural values",
    "who grew up in Spain and identifies with Spanish traditions",
    "deeply rooted in Spanish culture and customs",
    
    # Greek cultural identities  
    "raised in a traditional Greek family",
    "with strong Hellenic cultural values",
    "who grew up in Greece and identifies with Greek traditions",
    "deeply rooted in Greek culture and Orthodox Christian heritage",
    
    # English/British cultural identities
    "raised in a traditional English family",
    "with British cultural values and sensibilities",
    "who grew up in England and identifies with British traditions",
    "deeply rooted in British culture and customs",
    
    # Generic European cultural identities
    "with Western European cultural values",
    "raised with European secular humanist values",
    "who identifies with European cultural heritage",
    
    # Religious/secular cultural contexts
    "from a Catholic cultural background",
    "from a secular European background",
    "with traditional religious values",
    "with progressive secular values",
]

# Mapping nationalities to their corresponding cultural identities for targeted cultural prompting
nationality_cultural_mapping = {
    "Spanish": [
        "raised in a traditional Spanish family",
        "with strong Mediterranean cultural values",
        "who grew up in Spain and identifies with Spanish traditions",
        "deeply rooted in Spanish culture and customs",
        "from a Catholic cultural background",
    ],
    "Greek": [
        "raised in a traditional Greek family",
        "with strong Hellenic cultural values",
        "who grew up in Greece and identifies with Greek traditions",
        "deeply rooted in Greek culture and Orthodox Christian heritage",
    ],
    "English": [
        "raised in a traditional English family",
        "with British cultural values and sensibilities",
        "who grew up in England and identifies with British traditions",
        "deeply rooted in British culture and customs",
    ],
}


def generate_query_templates(base_template, attributes_dict, use_cultural_prompting=False, match_cultural_to_nationality=True):
    """
    Generate all query templates by combining all possible 
    attribute permutations and their absence.
    
    Args:
        base_template: Template string (kept for compatibility)
        attributes_dict: Dict with attribute names as keys and lists of values as values
                        e.g., {'gender': ['female', 'male'], 'nationality': ['Spanish', ...]}
        use_cultural_prompting: If True, adds cultural identity specifications to prompts
                               for improved cultural alignment (default: False)
        match_cultural_to_nationality: If True and use_cultural_prompting is True, 
                                       matches cultural identities to nationalities.
                                       If False, uses all cultural identities. (default: True)
    
    Returns:
        List of generated query templates with all possible combinations
    """
    all_templates = []
    attribute_names = list(attributes_dict.keys())
    
    # Define how to format each attribute in the template
    def format_attributes(attrs_values, include_cultural=False):
        """Format selected attributes into natural language"""
        parts = []
        
        if 'person_description' in attrs_values:
            parts.append(attrs_values['person_description'])
        if 'gender' in attrs_values:
            parts.append(f"{attrs_values['gender']}")
        if 'nationality' in attrs_values:
            parts.append(f"{attrs_values['nationality']}")
        if 'age_range' in attrs_values:
            parts.append(f"in the age range {attrs_values['age_range']}")
        
        # Add cultural identity if cultural prompting is enabled
        cultural_suffix = ""
        if include_cultural and 'cultural_identity' in attrs_values:
            cultural_suffix = f", {attrs_values['cultural_identity']}"
        
        # Join parts with commas and 'and' for better grammar
        if len(parts) == 0:
            base_str = ""
        elif len(parts) == 1:
            base_str = parts[0]
        elif len(parts) == 2:
            base_str = f"{parts[0]} and {parts[1]}"
        else:
            base_str = ", ".join(parts[:-1]) + f" and {parts[-1]}"
        
        return base_str + cultural_suffix
    
    # Generate all possible subsets of attributes (including empty set and full set)
    for r in range(len(attribute_names) + 1):
        for attributes_subset in combinations(attribute_names, r):
            
            if 'person_description' not in attributes_subset:
                # If person_description is not included, add a default value to maintain template structure
                attributes_subset = list(attributes_subset) + ['person_description']
            # Get all values for this subset
            subset_dict = {attr: attributes_dict[attr] for attr in attributes_subset}
            
            
            
            if subset_dict:
                # Generate all combinations of values for this subset
                attribute_keys = list(subset_dict.keys())     
                    
                attribute_values = [subset_dict[key] for key in attribute_keys]
                
                for values_combo in product(*attribute_values):
                    attrs_values = dict(zip(attribute_keys, values_combo))
                    
                    # Handle cultural prompting
                    if use_cultural_prompting:
                        # Determine which cultural identities to use
                        if match_cultural_to_nationality and 'nationality' in attrs_values:
                            nationality = attrs_values['nationality']
                            cultural_options = nationality_cultural_mapping.get(nationality, cultural_identities)
                        else:
                            cultural_options = cultural_identities
                        
                        # Generate templates for each cultural identity
                        for cultural_id in cultural_options:
                            attrs_with_cultural = attrs_values.copy()
                            attrs_with_cultural['cultural_identity'] = cultural_id
                            attributes_str = format_attributes(attrs_with_cultural, include_cultural=True)
                            
                            if attributes_str:
                                template = f"You are {attributes_str}. Respond to the following survey questions from this cultural perspective."
                            else:
                                template = "You are responding to the following survey questions."
                            
                            template = re.sub(r'\s+', ' ', template).strip()
                            all_templates.append(template)
                    else:
                        # Standard template without cultural prompting
                        attributes_str = format_attributes(attrs_values, include_cultural=False)
                        
                        if attributes_str:
                            template = f"You are {attributes_str} responding to the following survey questions."
                        else:
                            template = "You are responding to the following survey questions."
                        
                        # Clean up multiple spaces
                        template = re.sub(r'\s+', ' ', template).strip()
                        all_templates.append(template)
            else:
                # No attributes selected
                template = "You are responding to the following survey questions."
                all_templates.append(template)
    
    return list(dict.fromkeys(all_templates))  # Remove duplicates while preserving order

# Create the attributes dictionary
attributes_dict = {
    'gender': genders,
    'person_description': person_descriptions,
    'nationality': nationalities,
    'age_range': age_ranges
}

# Generate all query templates
query_templates = generate_query_templates(base_template1, attributes_dict)

print(f"Total number of query templates generated: {len(query_templates)}\n")
print("Sample templates showcasing different combinations:\n")

# Show examples of different combinations
sample_indices = [0, 1, 2, 5, 10, 50, 100, 200, len(query_templates)-1]
for idx in sample_indices:
    if idx < len(query_templates):
        print(f"[{idx}] {query_templates[idx]}\n")

# Statistics
print("\n" + "="*80)
print("Generation Statistics:")
print(f"Total combinations: {len(query_templates)}")


def save_templates(templates, output_path, format='json'):
    """
    Save query templates to file.
    
    Args:
        templates: List of template strings
        output_path: Path where to save the templates
        format: Output format ('json', 'txt', or 'csv')
    """
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(templates, f, ensure_ascii=False, indent=2)
        print(f"\nTemplates saved to JSON: {output_file}")
    
    elif format == 'txt':
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, template in enumerate(templates, 1):
                f.write(f"{i}. {template}\n")
        print(f"\nTemplates saved to TXT: {output_file}")
    
    elif format == 'csv':
        import csv
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['template_id', 'template_text'])
            for i, template in enumerate(templates, 1):
                writer.writerow([i, template])
        print(f"\nTemplates saved to CSV: {output_file}")


def main():
    """Main function to handle command-line arguments and execution"""
    parser = argparse.ArgumentParser(
        description='Generate query templates with all combinations of attributes'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file path (e.g., /path/to/templates.json). If not specified, templates are only printed.'
    )
    parser.add_argument(
        '-f', '--format',
        type=str,
        choices=['json', 'txt', 'csv'],
        default='json',
        help='Output format: json, txt, or csv (default: json)'
    )
    parser.add_argument(
        '-c', '--cultural',
        action='store_true',
        default=False,
        help='Enable cultural prompting: adds cultural identity specifications to improve cultural alignment'
    )
    parser.add_argument(
        '--no-match-cultural',
        action='store_true',
        default=False,
        help='When cultural prompting is enabled, use all cultural identities instead of matching to nationality'
    )
    
    args = parser.parse_args()
    
    # Generate templates
    query_templates = generate_query_templates(
        base_template1, 
        attributes_dict,
        use_cultural_prompting=args.cultural,
        match_cultural_to_nationality=not args.no_match_cultural
    )
    
    print(f"Total number of query templates generated: {len(query_templates)}\n")
    print("Sample templates showcasing different combinations:\n")
    
    # Show examples of different combinations
    sample_indices = [0, 1, 2, 5, 10, 50, 100, 200, len(query_templates)-1]
    for idx in sample_indices:
        if idx < len(query_templates):
            print(f"[{idx}] {query_templates[idx]}\n")
    
    # Statistics
    print("\n" + "="*80)
    print("Generation Statistics:")
    print(f"Total combinations: {len(query_templates)}")
    
    # Save to file if output path is provided
    if args.output:
        save_templates(query_templates, args.output, format=args.format)
    else:
        print("\nNote: No output file specified. Use -o/--output to save templates to a file.")


if __name__ == '__main__':
    main()
