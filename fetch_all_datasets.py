import os
import sys
import urllib.request
import zipfile
import io
import pandas as pd
import shutil

print("==================================================")
print("  FETCHING AND ORGANIZING CTG & FETAL DATASETS    ")
print("==================================================")

os.makedirs('datasets', exist_ok=True)
uci_dir = os.path.join('datasets', 'uci_ctg')
ctu_dir = os.path.join('datasets', 'physionet_ctu_uhb')
os.makedirs(uci_dir, exist_ok=True)
os.makedirs(ctu_dir, exist_ok=True)

# 1. UCI CTG Dataset
print("\n[1/2] Verifying UCI Cardiotocography Dataset...")
uci_xls = os.path.join(uci_dir, 'CTG.xls')
if not os.path.exists(uci_xls):
    uci_url = "https://archive.ics.uci.edu/static/public/193/cardiotocography.zip"
    print(f"  Downloading from {uci_url}...")
    try:
        req = urllib.request.Request(uci_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            for name in z.namelist():
                if name.endswith('.xls') or name.endswith('.xlsx') or name.endswith('.csv'):
                    z.extract(name, uci_dir)
                    print(f"    -> Extracted {name} to {uci_dir}/")
    except Exception as e:
        print("  Error downloading UCI CTG:", e)

print(f"  -> UCI CTG dataset ready in {uci_dir}/ (2,126 recordings with 3-class target: Normal, Suspect, Pathologic).")

# 2. PhysioNet CTU-UHB Intrapartum CTG Database
print("\n[2/2] Fetching PhysioNet CTU-UHB Intrapartum CTG Database records & sample cases...")
base_url = "https://physionet.org/files/ctu-uhb-ctgdb/1.0.0/"
headers = {'User-Agent': 'Mozilla/5.0'}

try:
    rec_url = base_url + "RECORDS"
    req = urllib.request.Request(rec_url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        records_txt = resp.read().decode('utf-8')
        record_ids = [r.strip() for r in records_txt.splitlines() if r.strip()]
        
    records_file = os.path.join(ctu_dir, 'RECORDS.txt')
    with open(records_file, 'w') as f:
        f.write('\n'.join(record_ids))
    print(f"  Total records available in CTU-UHB: {len(record_ids)} cases")
    
    # Download sample cases (headers and data)
    sample_ids = ['1001', '1002', '1003']
    for rid in sample_ids:
        for ext in ['.hea', '.dat']:
            furl = f"{base_url}{rid}{ext}"
            out_file = os.path.join(ctu_dir, f"{rid}{ext}")
            if not os.path.exists(out_file):
                req = urllib.request.Request(furl, headers=headers)
                with urllib.request.urlopen(req, timeout=20) as resp, open(out_file, 'wb') as out_f:
                    out_f.write(resp.read())
        print(f"    -> Sample continuous case {rid} ready in {ctu_dir}/")
        
except Exception as e:
    print("  Error fetching PhysioNet CTU-UHB:", e)

print("\n==================================================")
print("  ALL DATASETS READY IN 'datasets/' DIRECTORY!   ")
print("==================================================")
