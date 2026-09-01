import os
import sys
import urllib.request
import zipfile
import io
import pandas as pd

target_dir = os.path.join('datasets', 'uci_ctg')
os.makedirs(target_dir, exist_ok=True)

# Try downloading directly from UCI repo
uci_url = "https://archive.ics.uci.edu/static/public/193/cardiotocography.zip"
print(f"Downloading from {uci_url}...")
try:
    req = urllib.request.Request(uci_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        print("Zip contents:", z.namelist())
        for name in z.namelist():
            if name.endswith('.xls') or name.endswith('.xlsx') or name.endswith('.csv'):
                z.extract(name, target_dir)
                print(f"Extracted {name} to {target_dir}/")
except Exception as e:
    print("UCI Direct download error:", e)

print(f"Files in {target_dir}:", os.listdir(target_dir))
