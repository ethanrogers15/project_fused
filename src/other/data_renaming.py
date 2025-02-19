# Renaming data files for neatness

import os
from pathlib import Path


def main():
    data_path = '/project_fused/data'
    for walk in os.walk(data_path):
        parent_path = walk[0]
        category = Path(parent_path).name
        file_list = walk[2]
        if not file_list:
            continue
        i = 1
        for filename in file_list:
            file_path = Path(parent_path) / filename
            new_filename = f"{category}_image_{i}.png"
            new_file_path = file_path.with_name(new_filename)
            file_path.rename(new_file_path)
            i += 1
        
        
if __name__ == '__main__':
    main()