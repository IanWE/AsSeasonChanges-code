# AsSeasonChanges-code

> **Implementation for paper:** > *As Seasons Change: Temporal Feature Powers Malware Classification*

---

## 🚀 Core Algorithm
If you are primarily interested in the **G-MoE** implementation, please refer to:
* `core/moe/moe.py`: Core MoE architecture.
* `core/nn.py`: Neural network architectures and G-MoE training settings.

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

### 1. Download Sources
Before reproduction, please download the following datasets:
* [ANDROID (AndroZoo)](https://androzoo.uni.lu/) (Please download refer to sha256 list in `dataset/2024-GP1-meta.json`)
* [EMBER](https://github.com/elastic/ember/)
* [AnoShift](https://github.com/bit-ml/AnoShift/tree/main)
* [WildTime](https://github.com/huaxiuyao/Wild-Time)

### 2. Metadata Information
We provide pre-processed metadata in the `dataset/` directory:
* `2024-GP1-meta.json`: Full dataset metadata.
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

### Main Scripts
| Script | Description |
| :--- | :--- |
| `verify_negative_effect.py` | Verifies the distribution conflict effect by gradually removing old samples. |
| `next_year_prediction.py` | Trains on all data prior to the testing year to predict future samples. |
| `sustainability_verification.py` | Evaluates model sustainability (Training on 2014, testing on the next 9 years). |
| `active_learning.py` | Verifies effectiveness in an Active Learning context. |

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

Ensure the environment and respective datasets are ready before running:

* **EMBER:** `ember_verification.py`
* **AnoShift:** `Anoshift/test_on_anoshift.py`
* **Kyoto (G-MoE + DeepSVDD):** `Anoshift/kyoto.py`
* **WildTime:** `wildtime/wildtime.ipynb` (Jupyter Notebook)
