#!/usr/bin/env python3
"""
Generate all HDF5 map files from YAML definitions.

This script walks through the maps directory and generates .h5 files
from all .yaml map definitions. This allows maps to be version controlled
as human-readable YAML files while still providing the efficient HDF5
format for runtime use.

Usage:
    python generate_all_maps.py              # Generate all maps
    python generate_all_maps.py --clean      # Remove all .h5 files first
    python generate_all_maps.py --verbose    # Show detailed output
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import yaml2HDF5
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum.maps.yaml2HDF5 import generate_map_from_yaml


def find_all_yaml_maps(maps_dir="quantum/maps"):
    """Find all YAML map files in the maps directory."""
    maps_path = Path(maps_dir)
    yaml_files = []
    
    for yaml_file in maps_path.rglob("*.yaml"):
        # Skip template files
        if yaml_file.name == "template.yaml":
            continue
        yaml_files.append(yaml_file)
    
    return sorted(yaml_files)


def clean_h5_files(maps_dir="quantum/maps"):
    """Remove all .h5 and .hdf5 files from maps directory."""
    maps_path = Path(maps_dir)
    removed_count = 0
    
    for h5_file in maps_path.rglob("*.h5"):
        h5_file.unlink()
        removed_count += 1
        print(f"🗑️  Removed: {h5_file}")
    
    for hdf5_file in maps_path.rglob("*.hdf5"):
        hdf5_file.unlink()
        removed_count += 1
        print(f"🗑️  Removed: {hdf5_file}")
    
    return removed_count


def generate_all_maps(maps_dir="quantum/maps", verbose=False):
    """Generate all HDF5 maps from YAML definitions."""
    yaml_files = find_all_yaml_maps(maps_dir)
    
    if not yaml_files:
        print("⚠️  No YAML map files found!")
        return 0
    
    print(f"Found {len(yaml_files)} YAML map definition(s)")
    print("-" * 60)
    
    success_count = 0
    error_count = 0
    
    for yaml_file in yaml_files:
        try:
            # Output to same directory as YAML file
            output_dir = yaml_file.parent
            
            if verbose:
                print(f"\n📄 Processing: {yaml_file}")
                print(f"   Output dir: {output_dir}")
            
            # Resolve materials.yaml relative to this script
            script_dir = Path(__file__).parent.resolve()
            materials_path = script_dir.parent / "config" / "materials.yaml"
            
            generate_map_from_yaml(
                yaml_path=str(yaml_file),
                output_dir=str(output_dir),
                materials_path=str(materials_path)
            )
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {yaml_file}: {e}")
            error_count += 1
    
    print("-" * 60)
    print(f"\n✅ Successfully generated: {success_count}")
    if error_count > 0:
        print(f"❌ Errors: {error_count}")
    
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Generate HDF5 map files from YAML definitions"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove all existing .h5 files before generating"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--maps-dir",
        default="quantum/maps",
        help="Maps directory (default: quantum/maps)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Map Generation Tool")
    print("=" * 60)
    
    if args.clean:
        print("\n🧹 Cleaning existing HDF5 files...")
        removed = clean_h5_files(args.maps_dir)
        print(f"Removed {removed} file(s)\n")
    
    print("\n🔨 Generating maps from YAML...")
    count = generate_all_maps(args.maps_dir, args.verbose)
    
    if count > 0:
        print("\n✨ Map generation complete!")
    else:
        print("\n⚠️  No maps were generated.")
        sys.exit(1)


if __name__ == "__main__":
    main()
