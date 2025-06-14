import json
import os
from pathlib import Path
import shutil
from datetime import datetime

def clean_type_names(data):
    """Recursively clean type names by removing trailing periods"""
    if isinstance(data, dict):
        cleaned_data = {}
        for key, value in data.items():
            if key == "type" and isinstance(value, str):
                # Remove trailing period from type values
                cleaned_data[key] = value.rstrip('.')
            else:
                # Recursively clean nested structures
                cleaned_data[key] = clean_type_names(value)
        return cleaned_data
    elif isinstance(data, list):
        # Clean each item in the list
        return [clean_type_names(item) for item in data]
    else:
        # Return as-is for non-dict, non-list values
        return data

def backup_file(filepath):
    """Create a backup of the original file"""
    backup_dir = Path(filepath).parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{Path(filepath).stem}_backup_{timestamp}.json"
    backup_path = backup_dir / backup_filename
    
    shutil.copy2(filepath, backup_path)
    return backup_path

def process_json_file(filepath, create_backup=True):
    """Process a single JSON file to clean type names"""
    try:
        print(f"Processing: {filepath}")
        
        # Create backup if requested
        if create_backup:
            backup_path = backup_file(filepath)
            print(f"  Backup created: {backup_path}")
        
        # Read the JSON file
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Clean type names
        cleaned_data = clean_type_names(data)
        
        # Count changes made
        changes = count_type_changes(data, cleaned_data)
        
        if changes > 0:
            # Write back the cleaned data
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(cleaned_data, file, indent=2, ensure_ascii=False)
            
            print(f"  ✅ Cleaned {changes} type names")
        else:
            print(f"  ℹ️  No type names needed cleaning")
        
        return True, changes
        
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON decode error: {e}")
        return False, 0
    except Exception as e:
        print(f"  ❌ Error processing file: {e}")
        return False, 0

def count_type_changes(original, cleaned):
    """Count how many type values were changed"""
    changes = 0
    
    def count_changes_recursive(orig, clean):
        nonlocal changes
        if isinstance(orig, dict) and isinstance(clean, dict):
            for key in orig:
                if key == "type" and isinstance(orig[key], str) and isinstance(clean[key], str):
                    if orig[key] != clean[key]:
                        changes += 1
                elif key in clean:
                    count_changes_recursive(orig[key], clean[key])
        elif isinstance(orig, list) and isinstance(clean, list):
            for orig_item, clean_item in zip(orig, clean):
                count_changes_recursive(orig_item, clean_item)
    
    count_changes_recursive(original, cleaned)
    return changes

def find_all_json_files(root_folder):
    """Recursively find all JSON files in the folder and subfolders"""
    json_files = []
    
    root_path = Path(root_folder)
    
    if not root_path.exists():
        print(f"❌ Folder not found: {root_folder}")
        return []
    
    # Find all JSON files recursively
    for json_file in root_path.rglob("*.json"):
        # Skip backup files
        if "backup" not in str(json_file).lower():
            json_files.append(str(json_file))
    
    return sorted(json_files)

def preview_changes(root_folder):
    """Preview what changes would be made without actually modifying files"""
    print("🔍 PREVIEW MODE - No files will be modified")
    print("=" * 60)
    
    json_files = find_all_json_files(root_folder)
    
    if not json_files:
        print(f"No JSON files found in {root_folder}")
        return
    
    total_files = 0
    total_changes = 0
    files_with_changes = 0
    
    for filepath in json_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            cleaned_data = clean_type_names(data)
            changes = count_type_changes(data, cleaned_data)
            
            relative_path = os.path.relpath(filepath, root_folder)
            
            if changes > 0:
                print(f"📄 {relative_path}: {changes} type names to clean")
                files_with_changes += 1
                total_changes += changes
            else:
                print(f"📄 {relative_path}: No changes needed")
            
            total_files += 1
            
        except Exception as e:
            print(f"❌ Error reading {filepath}: {e}")
    
    print("=" * 60)
    print(f"📊 PREVIEW SUMMARY:")
    print(f"   Total files found: {total_files}")
    print(f"   Files needing changes: {files_with_changes}")
    print(f"   Total type names to clean: {total_changes}")

def clean_json_types_in_folder(root_folder, create_backup=True, preview_only=False):
    """Main function to clean type names in all JSON files in a folder"""
    
    if preview_only:
        preview_changes(root_folder)
        return
    
    print("🧹 JSON Type Name Cleaner")
    print("=" * 60)
    print(f"📁 Target folder: {root_folder}")
    print(f"💾 Create backups: {'Yes' if create_backup else 'No'}")
    print("=" * 60)
    
    json_files = find_all_json_files(root_folder)
    
    if not json_files:
        print(f"No JSON files found in {root_folder}")
        return
    
    print(f"Found {len(json_files)} JSON files:")
    for file in json_files:
        relative_path = os.path.relpath(file, root_folder)
        print(f"  📄 {relative_path}")
    
    print("\n" + "=" * 60)
    print("🚀 Starting processing...")
    print("=" * 60)
    
    successful_files = 0
    failed_files = 0
    total_changes = 0
    
    for filepath in json_files:
        success, changes = process_json_file(filepath, create_backup)
        
        if success:
            successful_files += 1
            total_changes += changes
        else:
            failed_files += 1
    
    print("=" * 60)
    print("📊 PROCESSING SUMMARY:")
    print(f"   ✅ Successfully processed: {successful_files}")
    print(f"   ❌ Failed to process: {failed_files}")
    print(f"   🧹 Total type names cleaned: {total_changes}")
    
    if create_backup and successful_files > 0:
        print(f"   💾 Backups stored in: {Path(json_files[0]).parent / 'backups'}")
    
    print("=" * 60)
    print("✨ Processing complete!")

# Example usage functions
def main():
    """Example usage of the JSON type cleaner"""
    
    # Replace with your actual folder path
    data_folder = "."  # Current directory - change this to your folder path
    
    # Option 1: Preview changes without modifying files
    print("1. Preview mode:")
    clean_json_types_in_folder(data_folder, preview_only=True)
    
    print("\n" + "="*80 + "\n")
    
    # Option 2: Actually clean the files (with backups)
    print("2. Cleaning mode:")
    clean_json_types_in_folder(data_folder, create_backup=True)

if __name__ == "__main__":
    main()

# Direct usage examples:
# For current directory:
# clean_json_types_in_folder(".", create_backup=True, preview_only=False)

# For specific folder:
    clean_json_types_in_folder(r"C:\Users\hp\OneDrive\Desktop\Study Material\epchatbot-finalvr3\data", create_backup=True, preview_only=False)