import os
import json
import fnmatch
def trainpathcsv_list(folder_path,
                      pattern='*_data_1st.csv'):
    trainpathcsv_list=[]
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if fnmatch.fnmatch(filename, pattern):
                trainpathcsv=os.path.join(root, filename)
                trainpathcsv_list.append(trainpathcsv)
    return trainpathcsv_list

def processed_json_list(folder_path,pattern="_data_1st_processed.json"):
    processed_json_list=[]
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if fnmatch.fnmatch(filename, pattern):
                processed_json=os.path.join(root, filename)
                processed_json_list.append(processed_json)
    merged_data = {}
    for file_name in processed_json_list:
        with open(os.path.join(file_name), "r") as f:
            file_data = json.load(f)
            merged_data.update(file_data)
    return merged_data