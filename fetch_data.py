import urllib.request
import zipfile
import io
import os
import pandas as pd

os.makedirs('data', exist_ok=True)
csv_path = os.path.join('data', 'CTG.csv')

# Method 1: Try downloading directly from UCI repo
uci_url = "https://archive.ics.uci.edu/static/public/193/cardiotocography.zip"
print(f"Downloading from {uci_url}...")
try:
    req = urllib.request.Request(uci_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        print("Zip contents:", z.namelist())
        for name in z.namelist():
            if name.endswith('.xls') or name.endswith('.xlsx') or name.endswith('.csv'):
                z.extract(name, 'data')
                print(f"Extracted {name} to data/")
except Exception as e:
    print("UCI Direct download error:", e)

# Check what was extracted
print("Files in data:", os.listdir('data'))
