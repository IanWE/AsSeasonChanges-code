import json
import os
import glob
from datetime import datetime
from typing import Dict, Any

# Process the raw VT reports into the style that euphony can read and process.
def process_single_json(input_file: str, output_file: str) -> bool:
    """
    processing single json file
    :param input_file: 
    :param output_file: 
    :return: Success with True，Failure with False returned
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_content = f.read().strip()
        
        try:
            data = json.loads(raw_content)  
        except json.JSONDecodeError:
            lines = [line.strip() for line in raw_content.splitlines() if line.strip()]
            if not lines:
                print(f"❌ {input_file}：empty or invalid JSON ")
                return False
            data = json.loads(lines[0])
        pos = 0
        scans = dict()
        for vendor in data['data']['attributes']['last_analysis_results']:
            scan = data['data']['attributes']['last_analysis_results'][vendor]
            if scan['result']:
                item = dict()
                pos += 1
                item['result'] = scan['result']
                item['version'] = scan['engine_version']
                item['update'] = scan['engine_update']
                item['detected'] = True
                scans[vendor] = item
        target_data: Dict[str, Any] = {
            "positives": pos,#data.get("positives", pos),
            "resource": data.get("data", "").get("id",""),
            "verbose_msg": data.get("verbose_msg", "Scan finished, information embedded"),
            "scans": scans,
            "sha1": data['data']['attributes']['sha1'],
            "total": len(data['data']['attributes']['last_analysis_results']),
            "scan_id": data.get("data", "").get("id",""),
            "permalink": data['data']['links']['self'],
            "sha256": data['data']['attributes']['sha256'],
            "scan_date": datetime.fromtimestamp(data['data']['attributes']['last_analysis_date']).strftime("%Y-%m-%d %H:%M:%S"),
            "md5": data['data']['attributes']['sha1'],
            "response_code": data.get("response_code", 1)
        }
        print(f"✅ Success：{os.path.basename(input_file)} -> {os.path.basename(output_file)}")
        return target_data
    except Exception as e:
        print(f"❌ Failed {os.path.basename(input_file)}：{str(e)}")
        return False

def batch_process_json_folder(input_folder: str, output_file: str = "reports.vt") -> None:
    json_files = glob.glob(os.path.join(input_folder, "*.json"))  
    
    if not json_files:
        print(f"⚠️ cannot find any json in {input_folder}")
        return
    
    total = len(json_files)
    success = 0
    fail = 0
    
    print(f"\n🚀 Start， {total} JSONs in total...")
    processed_data = []
    for input_file in json_files:
        target_data = process_single_json(input_file, output_file)
        processed_data.append(target_data)
        if target_data:
            success += 1
        else:
            fail += 1
    if processed_data:
        with open(output_file, 'w', encoding='utf-8') as f:
            for data in processed_data:
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")  #
        file_size = os.path.getsize(output_file) / 1024  
        print(f"\n✅ All processed file in {os.path.abspath(output_file)}")
        print(f"   File size:{file_size:.2f} KB | {len(processed_data)} valid item")
    else:
        print(f"\n❌ no valid data")
    print(f"\n📊 Finished!")
    print(f"   Total:{total} | Success: {success} | Failure: {fail}")
    print(f"   All processed files are saved into {os.path.abspath(output_file)}")

INPUT_FOLDER = "../VTCollection/reports/"  
OUTPUT_FILE = "./reports.vt"
batch_process_json_folder(
    input_folder=INPUT_FOLDER,
    output_file=OUTPUT_FILE  
)
os.system("java -jar euphony.jar  -r reports.vt  -CPEO")
