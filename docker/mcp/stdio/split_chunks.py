#!/usr/bin/env python3
"""
Script to split big_chunks.json into multiple JSON files.
Each output JSON file will contain chunks from exactly 2 source files.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

def load_big_chunks(file_path):
    """Load the big chunks JSON file."""
    print(f"Loading chunks from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def group_chunks_by_source(chunks):
    """Group chunks by their source_file parameter."""
    grouped = defaultdict(list)
    
    for chunk in chunks:
        source_file = chunk.get('source_file')
        if source_file:
            grouped[source_file].append(chunk)
    
    print(f"Found {len(grouped)} unique source files:")
    for source_file, chunk_list in grouped.items():
        print(f"  - {source_file}: {len(chunk_list)} chunks")
    
    return grouped

def create_output_files(grouped_chunks, output_dir):
    """Create output JSON files, each containing chunks from 2 source files."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    source_files = list(grouped_chunks.keys())
    file_counter = 1
    
    # Process source files in pairs
    for i in range(0, len(source_files), 2):
        # Get current pair of source files
        current_pair = source_files[i:i+2]
        
        # Collect all chunks from these source files
        combined_chunks = []
        source_info = []
        
        for source_file in current_pair:
            chunks = grouped_chunks[source_file]
            combined_chunks.extend(chunks)
            source_info.append(f"{source_file} ({len(chunks)} chunks)")
        
        # Create output filename
        output_filename = f"chunks_group_{file_counter:02d}.json"
        output_path = os.path.join(output_dir, output_filename)
        
        # Write the combined chunks to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"Created {output_filename} with {len(combined_chunks)} chunks from:")
        for info in source_info:
            print(f"  - {info}")
        
        file_counter += 1
    
    print(f"\nSuccessfully created {file_counter - 1} JSON files in {output_dir}")

def main():
    """Main function to orchestrate the splitting process."""
    script_dir = Path(__file__).parent
    input_file = script_dir / "big_chunks.json"
    output_dir = script_dir / "split_chunks"
    
    print("=" * 60)
    print("SPLITTING BIG CHUNKS INTO MULTIPLE JSON FILES")
    print("=" * 60)
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found!")
        return
    
    try:
        # Load the big chunks file
        chunks = load_big_chunks(input_file)
        print(f"Loaded {len(chunks)} total chunks")
        
        # Group chunks by source file
        grouped_chunks = group_chunks_by_source(chunks)
        
        print(f"\nCreating output files in {output_dir}...")
        print("-" * 40)
        
        # Create output files with 2 source files each
        create_output_files(grouped_chunks, output_dir)
        
        print("\n" + "=" * 60)
        print("SPLITTING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
