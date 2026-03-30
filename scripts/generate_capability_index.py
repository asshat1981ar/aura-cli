#!/usr/bin/env python3
import os
import re
import glob
import sys

def parse_retros(retro_dir):
    """
    Parses all retro markdown files for capability deltas.
    Returns a list of dicts: [{'id': 'AF-DELTA-0001', 'title': '...', 'source': 'S000 retro'}, ...]
    """
    deltas = []
    retro_files = glob.glob(os.path.join(retro_dir, "*.md"))
    
    for file_path in retro_files:
        filename = os.path.basename(file_path)
        # Extract sprint ID from filename, e.g., S000 from S000_retro.md
        sprint_match = re.match(r"^(S[0-9]{3})", filename)
        source = f"{sprint_match.group(1)} retro" if sprint_match else filename
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find the Capability Deltas Proposed section
        # Look for the header and then items until the next header or end of file
        section_match = re.search(r"## Capability Deltas Proposed\n\n(.*?)(?=\n##|\Z)", content, re.DOTALL)
        if section_match:
            section_content = section_match.group(1)
            # Find lines like: - **AF-DELTA-0001**: Add capability-gap index generation
            item_matches = re.finditer(r"^- \*\*(AF-DELTA-[0-9]{4})\*\*: (.*)", section_content, re.MULTILINE)
            for m in item_matches:
                deltas.append({
                    'id': m.group(1),
                    'title': m.group(2).strip(),
                    'source': source
                })
                
    return deltas

def get_existing_ids(index_content):
    """
    Extracts all AF-DELTA-XXXX IDs from the current index content to avoid duplicates.
    """
    return set(re.findall(r"AF-DELTA-[0-9]{4}", index_content))

def generate_index(index_path, new_deltas, dry_run=False):
    """
    Updates the capability_gap_index.yaml with new deltas.
    """
    if not os.path.exists(index_path):
        print(f"Error: Index file not found at {index_path}")
        return
    
    with open(index_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    existing_ids = get_existing_ids(content)
    to_add = [d for d in new_deltas if d['id'] not in existing_ids]
    
    if not to_add:
        print("No new capability deltas found.")
        return
    
    print(f"Found {len(to_add)} new deltas to add.")
    
    # We want to insert them into the 'proposed:' section.
    # Usually 'proposed:' is followed by a list of items.
    # We'll look for 'proposed:' and then either the first item or the end of the line.
    
    updated_content = content
    for delta in to_add:
        new_entry = f"""
  - id: {delta['id']}
    title: "{delta['title']}"
    source: "{delta['source']}"
    target_areas: []"""
        
        # Simple insertion: find 'proposed:' and insert after it.
        # This is a bit naive but works if 'proposed:' exists.
        if 'proposed: []' in updated_content:
            updated_content = updated_content.replace('proposed: []', 'proposed:' + new_entry)
        else:
            # Insert after the 'proposed:' line
            updated_content = re.sub(r"(proposed:.*?\n)", r"\1" + new_entry + "\n", updated_content, count=1)

    if dry_run:
        print("DRY RUN: The following changes would be made:")
        print(updated_content)
    else:
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"Updated {index_path} successfully.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate capability delta index from retros.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing.")
    parser.add_argument("--retro-dir", default=".aura_forge/retros", help="Directory containing retro markdown files.")
    parser.add_argument("--index-file", default=".aura_forge/indexes/capability_gap_index.yaml", help="Path to the capability gap index file.")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.retro_dir):
        print(f"Error: Retro directory {args.retro_dir} not found.")
        sys.exit(1)
        
    deltas = parse_retros(args.retro_dir)
    generate_index(args.index_file, deltas, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
