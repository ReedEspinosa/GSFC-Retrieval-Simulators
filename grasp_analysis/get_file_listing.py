#!/usr/bin/env python3
"""
Scan GMAO Nature Run data folders and generate file lists within a date range.
"""

import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path


def parse_date(date_string):
    """
    Parse date string in YYYY-MM-DD format.
    
    Args:
        date_string: Date in 'YYYY-MM-DD' format
        
    Returns:
        datetime object
    """
    try:
        return datetime.strptime(date_string, '%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_string}. Use YYYY-MM-DD")


def date_range(start_date, end_date):
    """
    Generate all dates between start_date and end_date (inclusive).
    
    Args:
        start_date: datetime object
        end_date: datetime object
        
    Yields:
        datetime objects for each day in range
    """
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(days=1)


def format_size(size_bytes):
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB", "234.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} EB"


def scan_data_folders(root_dir, start_date, end_date, output_file):
    """
    Scan GMAO Nature Run folders for files within date range.
    
    Args:
        root_dir: Root directory path (GMAO_Nature_Run)
        start_date: Start date (datetime object)
        end_date: End date (datetime object)
        output_file: Output text file path
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    
    # Get the root directory name to include in paths
    root_name = root_path.name
    
    # Product folders to scan
    product_folders = ['SS450', 'GPM']
    
    file_list = []
    total_files = 0
    total_size = 0
    skipped_folders = []
    
    print(f"Scanning date range: {start_date.date()} to {end_date.date()}")
    print(f"Root directory: {root_path.absolute()}")
    print(f"Root name: {root_name}\n")
    
    # Iterate through product folders
    for product in product_folders:
        product_path = root_path / product
        
        if not product_path.exists():
            print(f"Warning: Product folder not found: {product}")
            continue
        
        # Check permissions on product folder
        try:
            # Test if we can list the directory
            _ = list(product_path.iterdir())
        except PermissionError:
            msg = f"Permission denied - cannot access: {root_name}/{product}"
            print(f"SKIPPED: {msg}")
            skipped_folders.append(f"{root_name}/{product}")
            continue
        
        print(f"Scanning product: {product}")
        
        # Find all Level subfolders
        try:
            level_folders = [d for d in product_path.iterdir() 
                            if d.is_dir() and d.name.startswith('Level')]
        except PermissionError:
            msg = f"Permission denied - cannot list contents: {root_name}/{product}"
            print(f"  SKIPPED: {msg}")
            skipped_folders.append(f"{root_name}/{product}")
            continue
        
        if not level_folders:
            print(f"  No Level folders found in {product}")
            continue
        
        # Iterate through each Level folder
        for level_folder in sorted(level_folders):
            level_name = level_folder.name
            
            # Check permissions on level folder
            try:
                # Test if we can list the directory
                _ = list(level_folder.iterdir())
            except PermissionError:
                relative_path = f"{root_name}/{product}/{level_name}"
                msg = f"Permission denied - cannot access: {relative_path}"
                print(f"  SKIPPED: {msg}")
                skipped_folders.append(relative_path)
                continue
            
            print(f"  Checking {level_name}...")
            
            level_file_count = 0
            level_size = 0
            
            # Iterate through date range
            for date in date_range(start_date, end_date):
                year_folder = f"Y{date.year}"
                month_folder = f"M{date.month:02d}"
                day_folder = f"D{date.day:02d}"
                
                # Construct full path: Level/YYYY/MM/DD
                date_path = level_folder / year_folder / month_folder / day_folder
                
                if not date_path.exists():
                    continue
                
                # Get all files in this date folder
                try:
                    files_in_folder = list(date_path.iterdir())
                except PermissionError:
                    relative_path = f"{root_name}/{product}/{level_name}/{year_folder}/{month_folder}/{day_folder}"
                    msg = f"Permission denied - cannot access: {relative_path}"
                    print(f"    SKIPPED: {msg}")
                    skipped_folders.append(relative_path)
                    continue
                
                # Process each file
                for file_path in files_in_folder:
                    try:
                        if file_path.is_file():
                            # Get file size
                            file_size = file_path.stat().st_size
                            
                            # Get relative path from root directory and prepend root name
                            relative_path = file_path.relative_to(root_path)
                            full_relative_path = f"{root_name}/{relative_path}"
                            file_list.append(full_relative_path)
                            
                            level_file_count += 1
                            level_size += file_size
                            total_files += 1
                            total_size += file_size
                    except PermissionError:
                        relative_path = file_path.relative_to(root_path)
                        full_relative_path = f"{root_name}/{relative_path}"
                        msg = f"Permission denied - cannot access file: {full_relative_path}"
                        print(f"    SKIPPED: {msg}")
                        skipped_folders.append(full_relative_path)
                        continue
                    except OSError as e:
                        # Handle other file system errors (e.g., broken symlinks)
                        relative_path = file_path.relative_to(root_path)
                        full_relative_path = f"{root_name}/{relative_path}"
                        msg = f"Error accessing file: {full_relative_path} - {e}"
                        print(f"    SKIPPED: {msg}")
                        continue
            
            if level_file_count > 0:
                print(f"    Found {level_file_count} files ({format_size(level_size)}) in {level_name}")
    
    # Sort file list
    file_list.sort()
    
    # Write to output file
    print(f"\n{'='*70}")
    print(f"Writing {total_files} file paths to: {output_file}")
    with open(output_file, 'w') as f:
        for file_path in file_list:
            f.write(f"{file_path}\n")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total files found: {total_files:,}")
    print(f"Total size: {format_size(total_size)} ({total_size:,} bytes)")
    print(f"Output file: {output_file}")
    
    if skipped_folders:
        print(f"\nWarning: {len(skipped_folders)} folder(s)/file(s) were skipped due to permission errors.")
        print("See output above for details.")
    
    return file_list, total_size


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Scan GMAO Nature Run folders and generate file lists within a date range.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan entire year 2006
  %(prog)s /path/to/GMAO_Nature_Run 2006-01-01 2006-12-31 -o file_list.txt
  
  # Scan single month
  %(prog)s /path/to/GMAO_Nature_Run 2006-01-01 2006-01-31 -o jan_2006.txt
  
  # Scan single day
  %(prog)s /path/to/GMAO_Nature_Run 2006-01-15 2006-01-15 -o single_day.txt
        """
    )
    
    parser.add_argument(
        'root_dir',
        help='Root directory path (GMAO_Nature_Run)'
    )
    
    parser.add_argument(
        'start_date',
        type=parse_date,
        help='Start date in YYYY-MM-DD format'
    )
    
    parser.add_argument(
        'end_date',
        type=parse_date,
        help='End date in YYYY-MM-DD format (inclusive)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='file_list.txt',
        help='Output file path (default: file_list.txt)'
    )
    
    args = parser.parse_args()
    
    # Validate date range
    if args.end_date < args.start_date:
        parser.error("End date must be >= start date")
    
    try:
        scan_data_folders(args.root_dir, args.start_date, args.end_date, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
