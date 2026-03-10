import zipfile
import shutil
import os
from pathlib import Path

DELETE_ZIP_AFTER_EXTRACTION = False

root_folder = Path('.')

zip_files = list(root_folder.rglob('*.zip'))

if not zip_files:
    print("No .zip files found in the project.")
else:
    print(f"{len(zip_files)} archive(s) found. Starting extraction...\n")

    for zip_path in zip_files:
        target_folder = zip_path.parent 
        
        extraction_marker = target_folder / zip_path.stem
        if extraction_marker.exists():
            print(f"Skipping '{zip_path.name}', already extracted at '{extraction_marker}/'.")
            continue

        print(f"Extracting '{zip_path.name}' directly into '{target_folder}/'...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as archive:
                archive.extractall(path=target_folder)
            
            # --- MAC CLEANUP ---
            macosx_folder = target_folder / "__MACOSX"
            if macosx_folder.exists() and macosx_folder.is_dir():
                shutil.rmtree(macosx_folder)
                
            print(f"Success for {zip_path.name}!")
            
            if DELETE_ZIP_AFTER_EXTRACTION:
                os.remove(zip_path)
                print(f"The file {zip_path.name} has been deleted to free up space.")
                
        except zipfile.BadZipFile:
            print(f"Error: The file {zip_path.name} is corrupted or not a valid archive.")

print("\nAll zip files are decompressed.")