# AsSeasonChanges-code

> **Implementation for ACM CCS 2026 paper:** > *As Seasons Change: Temporal Feature Powers Malware Classification*

---

## 🚀 Core Algorithm
If you are primarily interested in the **G-MoE** implementation, please refer to:
* `core/moe/moe.py`: Core MoE architecture.
* `core/nn.py`: Neural network architectures and G-MoE training settings.

---
> [!TIP]
> For binary classification tasks, we report F1 score of the positive class `F1 (average='binary')`. For multi-class tasks, we report macro-averaged `F1 (average='macro')`. When comparing other works, please make sure the used metrics are consistent.

---

## 🛠 Environments & Setup

The repository is built on **Ubuntu 24.04** with **Python 3.13.5**.

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install git+[https://github.com/elastic/ember.git](https://github.com/elastic/ember.git)
    ```
2.  **Initialize directories:**
    ```bash
    bash init.sh
    ```

---

## 📊 Datasets 
If there are already datasets under `dataset/`, you can skip step 1, 2, 3. 

### 1. Download Sources
Before reproduction, please download the following dataset:
* [ANDROID (AndroZoo)](https://androzoo.uni.lu/) (Please download refer to sha256 list in `dataset/2024-GP-meta.json`)

### 2. Metadata Information
We provide pre-processed metadata in the `dataset/` directory:
* `2024-GP-meta.json`: Full dataset metadata.
* `hypercube_metadata.json`: Hypercube dataset metadata.
* `proposed.json`: Family information processed by **Euphony**.

### 3. Feature Extraction & Processing
* **DREBIN Features:** Ensure you use the newer version of [baksmali](https://github.com/baksmali/smali/releases) (we are using v2.5.2) to avoid significant feature loss.
* **Extraction Code:** Located in `feature-extraction/`. After processing, save them in the same format as in `data_utils.load_gp_data` and `data_utils.load_hypercube` for easy loading.
* **Loading Data:** Use `data_utils.load_gp_data(NAME)` or `data_utils.load_hypercube(NAME)`.

> [!TIP]
> **Quick Start:** You can also use datasets provided by existing works like [Transcendent](https://github.com/s2labres/transcendent-release) for evaluation.

---

## 📂 File Descriptions

### Scripts for Reproducing Main Experiments
| Script | Description | Expected Results |
| :--- | :--- | :--- |
| `verify_negative_effect.py` | Verifies the distribution conflict effect by gradually removing old samples (Figure 3). | Performance gains from deleting old training samples
| `next_year_prediction.py` | Trains on all data prior to the testing year to predict future samples (Table 2). | G-MoE demonstrates better performance |
| `sustainability_verification.py` | Evaluates model sustainability (Training on 2014, testing on the next 9 years, Table 4). | G-MoE demonstrates better performance |
| `active_learning.py` | Verifies effectiveness in an Active Learning context (Table 5). | G-MoE demonstrates better performance |

### Data Processing
* `process_apigraph.py`: Converts DREBIN features into **APIGraph** format.
* `process_data_bundle.py`: Processes datasets using **Subspace Compression with Binarization (SCB)**.
* `process_data_bundle_original.py`: Similar to above, but allows separate processing for train/test sets.

### Core Modules
* `core/data_utils.py`: Dataset loading utilities.
* `core/model_utils.py`: Utilities for training, saving, and loading models.
* `code/nn.py`: Configurations for NN, G-MoE, T-MoE, and standard MoE.
* `code/moe/`: Implementations of `G-MoE` (`moe.py`), `T-MoE` (`moe_wg.py`), and `MoE` (`moe_o.py`).
* `core/utils.py`: Miscellaneous helper functions.

---

## 🧪 Testing on Other Datasets

Ensure the corresponding environment and respective datasets are ready before running:

* **[EMBER](https://github.com/elastic/ember/):** `ember_verification.py`
* **[AnoShift](https://github.com/bit-ml/AnoShift/tree/main):** `Anoshift/test_on_anoshift.py`
* **[Kyoto (G-MoE + DeepSVDD)](https://github.com/bit-ml/AnoShift/tree/main):** `Anoshift/kyoto.py`
* **[WildTime](https://github.com/huaxiuyao/Wild-Time):** `wildtime/wildtime.ipynb`

