#!/usr/bin/env python3
"""
Cleanup script for BPE visualization project.
Removes temporary files and organizes visualizations.
"""

import os
import glob
import shutil
import argparse

def cleanup(organize=True):
    """
    Clean up temporary files and optionally organize visualizations.
    
    Args:
        organize: If True, move all PNG files to the visualizations directory
    """
    # Get the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Remove temporary files
    temp_files = glob.glob(os.path.join(project_dir, "_temp_*"))
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            print(f"Removed temporary file: {temp_file}")
        except OSError as e:
            print(f"Error removing {temp_file}: {e}")
    
    # Organize visualizations if requested
    if organize:
        # Create visualizations directory if it doesn't exist
        vis_dir = os.path.join(project_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Move all PNG files to the visualizations directory
        png_files = glob.glob(os.path.join(project_dir, "*.png"))
        for png_file in png_files:
            # Skip files already in the visualizations directory
            if os.path.dirname(png_file) == vis_dir:
                continue
                
            # Move the file
            filename = os.path.basename(png_file)
            destination = os.path.join(vis_dir, filename)
            try:
                shutil.move(png_file, destination)
                print(f"Moved {filename} to visualizations directory")
            except OSError as e:
                print(f"Error moving {png_file}: {e}")
    
    print("Cleanup complete!")

def main():
    parser = argparse.ArgumentParser(description="Clean up temporary files and organize visualizations")
    parser.add_argument("--no-organize", action="store_true", 
                        help="Only remove temporary files, don't organize visualizations")
    args = parser.parse_args()
    
    cleanup(organize=not args.no_organize)

if __name__ == "__main__":
    main()
