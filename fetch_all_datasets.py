import os
import sys
import urllib.request
import pandas as pd
import shutil

print("==================================================")
print("  FETCHING AND ORGANIZING CTG & FETAL DATASETS    ")
print("==================================================")

os.makedirs('datasets', exist_ok=True)
os.makedirs('datasets/uci_ctg', exist_ok=True)
os.makedirs('datasets/physionet_ctu_uhb', exist_ok=True)

# 1. UCI CTG Dataset
print("\n[1/2] Verifying UCI Cardiotocography Dataset...")
uci_xls = 'data/CTG.xls'
if os.path.exists(uci_xls):
    shutil.copy(uci_xls, 'datasets/uci_ctg/CTG.xls')
    if os.path.exists('data/CTG_cleaned.csv'):
        shutil.copy('data/CTG_cleaned.csv', 'datasets/uci_ctg/CTG_cleaned.csv')
    print("  -> UCI CTG dataset organized in datasets/uci_ctg/ (2,126 recordings with 3-class target: Normal, Suspect, Pathologic).")

# 2. PhysioNet CTU-UHB Intrapartum CTG Database
print("\n[2/2] Fetching PhysioNet CTU-UHB Intrapartum CTG Database records & sample cases...")
base_url = "https://physionet.org/files/ctu-uhb-ctgdb/1.0.0/"
headers = {'User-Agent': 'Mozilla/5.0'}

try:
    # Download RECORDS list
    rec_url = base_url + "RECORDS"
    req = urllib.request.Request(rec_url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        records_txt = resp.read().decode('utf-8')
        record_ids = [r.strip() for r in records_txt.splitlines() if r.strip()]
        
    with open('datasets/physionet_ctu_uhb/RECORDS.txt', 'w') as f:
        f.write('\n'.join(record_ids))
    print(f"  Total records available in CTU-UHB: {len(record_ids)} cases")
    
    # Download sample cases (headers and data)
    sample_ids = ['1001', '1002', '1003']
    for rid in sample_ids:
        for ext in ['.hea', '.dat']:
            furl = f"{base_url}{rid}{ext}"
            out_file = f"datasets/physionet_ctu_uhb/{rid}{ext}"
            if not os.path.exists(out_file):
                req = urllib.request.Request(furl, headers=headers)
                with urllib.request.urlopen(req, timeout=20) as resp, open(out_file, 'wb') as out_f:
                    out_f.write(resp.read())
        print(f"    -> Downloaded sample continuous case {rid} (.hea and .dat)")
        
    # Read the clinical outcomes from record 1001 header
    with open('datasets/physionet_ctu_uhb/1001.hea', 'r') as f:
        hea_content = f.read()
    print("\n  Sample Record 1001 Clinical Metadata:")
    print("  " + "\n  ".join(hea_content.splitlines()[:15]))
    
except Exception as e:
    print("  Error fetching PhysioNet CTU-UHB:", e)

print("\n==================================================")
print("  ALL DATASETS READY IN 'datasets/' DIRECTORY!   ")
print("==================================================")
