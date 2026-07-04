import os
import urllib.request

DOCS_PATH = "documents"
os.makedirs(DOCS_PATH, exist_ok=True)

PDFS = {
    "nsa-health-insurance-basics.pdf": "https://www.cms.gov/files/document/nsa-health-insurance-basics.pdf",
    "sbc-sample.pdf": "https://www.cms.gov/cciio/resources/forms-reports-and-other-resources/downloads/sbc-sample-completed-mm-508-fixed-4-12-16.pdf",
    "medicare-chapter15.pdf": "https://www.cms.gov/regulations-and-guidance/guidance/manuals/downloads/bp102c15.pdf"
}

for filename, url in PDFS.items():
    filepath = os.path.join(DOCS_PATH, filename)
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded {filename}")
    else:
        print(f"Already exists: {filename}")
