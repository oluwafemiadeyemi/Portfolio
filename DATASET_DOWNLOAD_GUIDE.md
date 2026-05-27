# Dataset Download Guide — All 10 Projects

> **Read this first.** Each section tells you exactly what to download, where to save it, and what commands to run. Datasets marked ✅ FREE & INSTANT require no account. Those marked 📝 REGISTER need a free account first.

---

## Step 0: Install Kaggle CLI (needed for 7 of 10 projects)

```bash
pip install kaggle
```

Then create your Kaggle API token:
1. Go to https://www.kaggle.com → Account → API → "Create New Token"
2. Download `kaggle.json`
3. Save to: `C:\Users\Artstanding\.kaggle\kaggle.json`
4. Verify: `kaggle datasets list`

---

## Project 1: Brand Intelligence Platform
**Folder**: `AirBnb reviews Sentimental Analysis\data\raw\`

### Dataset A: Yelp Open Dataset 2022 (6.9M reviews) 📝 REGISTER
1. Go to: https://business.yelp.com/data/resources/open-dataset/
2. Fill out the short form (name, email, use case)
3. Download the ZIP file (~4GB)
4. Extract and save these files to `data/raw/`:
   - `yelp_academic_dataset_review.json` ← most important
   - `yelp_academic_dataset_business.json`
   - `yelp_academic_dataset_user.json`

### Dataset B: Yelp via Kaggle (alternative, same data) ✅ FREE
```bash
kaggle datasets download -d yelp-dataset/yelp-dataset ^
  -p "AirBnb reviews Sentimental Analysis\data\raw\" --unzip
```

### Dataset C: Existing Airbnb data
Already in project root. No action needed.

**If you skip this**: The code auto-generates 2,000 synthetic hotel/restaurant reviews for demo.

---

## Project 2: Fraud Detection Platform
**Folder**: `Credit Card Default Prediction\data\raw\`

### Dataset A: Credit Card Fraud Detection 2023 ✅ FREE (Kaggle)
```bash
kaggle datasets download -d nelgiriyewithana/credit-card-fraud-detection-dataset-2023 ^
  -p "Credit Card Default Prediction\data\raw\" --unzip
```
Expected file: `creditcard_2023.csv` (568,630 transactions, 31 columns)

### Dataset B: IEEE-CIS Fraud Detection (590k transactions) 📝 REGISTER
1. Go to: https://www.kaggle.com/competitions/ieee-fraud-detection
2. Click "Join Competition" → accept rules
3. Then run:
```bash
kaggle competitions download -c ieee-fraud-detection ^
  -p "Credit Card Default Prediction\data\raw\" --unzip
```
Expected files: `train_transaction.csv`, `train_identity.csv`

**If you skip this**: The code auto-generates 500,000 synthetic transactions with realistic 3.5% fraud rate.

---

## Project 3: Fair Mortgage Decisioning Platform
**Folder**: `Loan Approval Prediction\data\raw\`

### HMDA 2022 Dataset (14M+ US Mortgage Applications) ✅ FREE INSTANT
Direct download — no account needed:

```bash
# Option 1: Download 2022 Combined LAR (all lenders, ~6GB)
# Go to: https://ffiec.cfpb.gov/data-publication/modified-lar/2022
# Click "Download Nationwide" → saves as 2022_public_lar.csv.zip

# Option 2: Use the API for a state subset (faster, 100k records)
# Replace STATE with your state code (e.g., CA, TX, NY, FL)
curl "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=CA&years=2022" ^
  -o "Loan Approval Prediction\data\raw\hmda_2022_CA.csv"
```

**Recommended**: Download just California or Texas data (~500k records each) for development.

**HMDA Data Browser** (easiest): https://ffiec.cfpb.gov/data-browser/
- Select: Year=2022, State=Texas (or any large state), click Download CSV

**If you skip this**: The code auto-generates 100,000 synthetic mortgage applications.

---

## Project 4: People Analytics Platform
**Folder**: `Employee Attrition Prediction\data\raw\`

### IBM HR Analytics Dataset ✅ FREE (Kaggle)
```bash
kaggle datasets download -d pavansubhasht/ibm-hr-analytics-attrition-dataset ^
  -p "Employee Attrition Prediction\data\raw\" --unzip
```
Expected file: `WA_Fn-UseC_-HR-Employee-Attrition.csv` (1,470 rows, 35 columns)

**Note**: The code automatically extends this to 50,000 synthetic employees using the IBM distributions.

**Also check**: The file may already exist in the project root. The code searches multiple locations.

---

## Project 5: Digital Biomarker Platform
**Folder**: `Parkinsons Disease Detection\data\raw\`

### Dataset A: UCI Parkinson's Telemonitoring (5,875 recordings) ✅ FREE INSTANT
```bash
# Direct download
curl "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data" ^
  -o "Parkinsons Disease Detection\data\raw\parkinsons_telemonitoring.csv"
```

### Dataset B: mPower Mobile Parkinson's Study (9,500+ participants) 📝 REGISTER
1. Create account at: https://www.synapse.org/register
2. Go to: https://www.synapse.org/#!Synapse:syn4993293
3. Click "Request Access" → complete certification quiz (15 min)
4. Install Synapse client: `pip install synapseclient`
5. Download:
```python
import synapseclient
syn = synapseclient.Synapse()
syn.login()  # prompts for credentials
syn.get('syn4993293', downloadLocation='Parkinsons Disease Detection/data/raw/')
```

### Dataset C: PPMI (Parkinson's Progression Markers Initiative) 📝 REGISTER
1. Register at: https://www.ppmi-info.org/access-data-specimens/download-data
2. Complete IRB-equivalent online training
3. Download from the data portal

**If you skip all**: Code auto-generates 6,000 multi-modal recordings (1,000 subjects × 6) with realistic PD biomarker distributions.

---

## Project 6: Supply Chain Risk Intelligence
**Folder**: `Bankruptcy Prediction\data\raw\`

### Dataset A: Financial Distress Prediction ✅ FREE (Kaggle)
```bash
kaggle datasets download -d shebrahimi/financial-distress ^
  -p "Bankruptcy Prediction\data\raw\" --unzip
```
Expected file: `Financial Distress.csv` (3,672 firms, 83 features)

### Dataset B: SEC EDGAR Financial Statements ✅ FREE INSTANT
```bash
# Downloads structured financial data for all public US companies
# Quarterly updates: https://www.sec.gov/cgi-bin/browse-edgar

# Install sec-edgar-downloader
pip install sec-edgar-downloader

# Or download pre-structured data directly:
# https://www.sec.gov/dera/data/financial-statements (ZIP files by quarter)
```

**If you skip this**: Code auto-generates 5,000 synthetic US companies (2016-2023) with realistic financial ratios.

---

## Project 7: Retail Operations Intelligence
**Folder**: `Object Detection\data\raw\`

### Dataset A: SKU-110K (110,628 retail product images) ✅ FREE
```bash
# Download from GitHub releases (~2GB)
# Go to: https://github.com/eg4000/SKU110K_CVPR19/releases
# Download: SKU110K_fixed.tar.gz
# Extract to: Object Detection\data\raw\SKU110K\
```

### Dataset B: Open Images V7 ✅ FREE (selective download)
```bash
pip install openimages
# Download just the "Supermarket" and "Retail" categories
python -c "
from openimages.download import download_category
download_category('Convenience store', 'Object Detection/data/raw/openimages', limit=5000)
"
```

### Dataset C: Existing Roboflow Dataset
Already in project (check Dataset/ folder). Use this first.

**If you skip this**: Code downloads 50 sample retail images and runs demo detection with pretrained YOLOv8.

---

## Project 8: Ergonomics & Injury Prevention
**Folder**: `Pose Estimation\data\raw\`

### Dataset A: COCO Keypoints 2023 ✅ FREE INSTANT
```bash
# Annotations only (~240MB) — no need for full 40GB image set
curl "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" ^
  -o "Pose Estimation\data\raw\coco_annotations.zip"
# Extract person_keypoints_train2017.json from the zip
```

### Dataset B: NTU RGB+D 120 (114,480 action samples) 📝 REGISTER
1. Fill request form at: https://rose1.ntu.edu.sg/dataset/actionRecognition/
2. Receive download link via email (usually within 24h)
3. Very large dataset (~2TB total) — request specific subsets

### Dataset C: Existing Activity Videos
Already in project (Frames/ or Videos/ directory). MediaPipe runs directly on these.

**Note**: MediaPipe Pose works out-of-box on any person image/video — no training data needed for inference. REBA/RULA algorithm is rule-based. Training data only needed to fine-tune custom pose model.

---

## Project 9: CLV & Retention Platform
**Folder**: `Customer Churn\data\raw\`

### Dataset A: KKBox Music Streaming Churn (2.6M users) 📝 REGISTER (Kaggle Competition)
1. Go to: https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge
2. Accept competition rules
3. Run:
```bash
kaggle competitions download -c kkbox-churn-prediction-challenge ^
  -p "Customer Churn\data\raw\" --unzip
```
Expected files: `train.csv`, `members_v3.csv`, `transactions_v2.csv`, `user_logs_v2.csv`

### Dataset B: IBM Telco Customer Churn (7,043 customers) ✅ FREE (Kaggle)
```bash
kaggle datasets download -d blastchar/telco-customer-churn ^
  -p "Customer Churn\data\raw\" --unzip
```
Expected file: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

**Also check**: Existing `customer_churn.csv` in project root — code searches there first.

**If you skip this**: Code generates 200,000 synthetic streaming users with realistic churn patterns.

---

## Project 10: PPE Safety Compliance System
**Folder**: `Face Detection\data\raw\`

### Dataset A: Hard Hat Detection ✅ FREE (Kaggle)
```bash
kaggle datasets download -d andrewmvd/hard-hat-detection ^
  -p "Face Detection\data\raw\" --unzip
```
Expected: `images/` + `annotations/` folders (7,041 images)

### Dataset B: Construction Site Safety (Roboflow) ✅ FREE (Kaggle)
```bash
kaggle datasets download -d snehilsanyal/construction-site-safety-image-dataset-roboflow ^
  -p "Face Detection\data\raw\" --unzip
```

### Dataset C: Roboflow Universe (PPE, best option) ✅ FREE
1. Go to: https://universe.roboflow.com/joseph-nelson/hard-hat-workers
2. Click "Download Dataset"
3. Select format: YOLOv8 → Download ZIP
4. Extract to: `Face Detection\data\raw\ppe_dataset\`

**If you skip this**: Code runs with pretrained YOLOv8 in demo mode on any uploaded image.

---

## Master Install Script

Run this once to install all dependencies for all 10 projects:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm shap ^
  fastapi uvicorn streamlit plotly joblib pydantic python-multipart ^
  nltk scipy matplotlib seaborn imbalanced-learn optuna ^
  transformers datasets sentence-transformers ^
  lifelines lifetimes ^
  networkx ^
  ultralytics opencv-python pillow supervision ^
  mediapipe ^
  kaggle sec-edgar-downloader ^
  loguru evidently fairlearn ^
  requests aiofiles
```

Or install per-project (recommended):
```bash
cd "Project Folder"
pip install -r requirements.txt
```

---

## Quick Start (After Data Download)

For each project:
```bash
cd "Project Folder"

# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (trains model + saves to data/models/)
python src/train.py

# 3. Start the API
uvicorn api.main:app --reload --port 8000

# 4. Start the dashboard (separate terminal)
streamlit run dashboard/app.py
```

---

## Priority Download Order

If you're short on time, download in this order (highest ROI first):

| Priority | Dataset | Size | Time | Project |
|---|---|---|---|---|
| 1 | IBM HR Attrition (Kaggle) | 300KB | 30 sec | Employee Attrition |
| 2 | Credit Card Fraud 2023 (Kaggle) | 150MB | 2 min | Fraud Detection |
| 3 | Financial Distress (Kaggle) | 1MB | 30 sec | Bankruptcy |
| 4 | IBM Telco Churn (Kaggle) | 1MB | 30 sec | Customer Churn |
| 5 | Hard Hat Detection (Kaggle) | 200MB | 3 min | PPE Safety |
| 6 | Yelp Dataset (Yelp.com) | 4GB | 20 min | Brand Intelligence |
| 7 | HMDA 2022 (CFPB) | 500MB per state | 10 min | Loan/Mortgage |
| 8 | IEEE-CIS Fraud (Kaggle competition) | 2GB | 15 min | Fraud Detection |
| 9 | KKBox Churn (Kaggle competition) | 500MB | 10 min | Customer Churn |
| 10 | UCI Telemonitoring | 2MB | 30 sec | Parkinson's |

*All projects work in demo mode without any downloads — synthetic data is auto-generated.*
