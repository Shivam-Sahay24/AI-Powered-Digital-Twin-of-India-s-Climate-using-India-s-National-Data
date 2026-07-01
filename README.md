# CLIMATE-TWIN: Baseline Prototype

An AI-Powered Digital Twin prototype for India's climate dynamics. This project utilizes a custom PyTorch `DualStream_AttnUConvLSTM` architecture to forecast highly localized monsoon precipitation and thermal anomalies across the Western Ghats grid.

## ⚙️ Engineering Overview
When forecasting localized climate events, standard regression networks suffer from a "dry bias," collapsing into a local minimum where they predict zero rainfall to cheat average accuracy metrics. 

This Digital Twin bypasses standard statistical averaging by separating heat and moisture into independent processing streams. It predicts spatiotemporal trajectories using a 14-day sliding window of historical multi-dimensional telemetry:
* Gridded Precipitation ($0.25^\circ \times 0.25^\circ$)
* Maximum Temperature ($1.0^\circ \times 1.0^\circ$)
* Minimum Temperature ($1.0^\circ \times 1.0^\circ$)

## 🧠 Machine Learning Architecture
* **Competitors:** Standard Spatiotemporal CNNs vs. `DualStream_AttnUConvLSTM`
* **Winner:** The Dual-Stream network successfully preserves thermodynamic and hydrological physical laws, merging them via a Multi-Head Cross-Attention Bridge before temporal propagation.
* **Optimization:** Optimized using `Optuna` Bayesian tuning and guided by a custom `PerfectClimateLoss` function, which applies a 10x penalty weight on active storm pixels to forcefully prevent dry-bias overfitting.

## 📈 Model Training Convergence

![Dual-Stream Foundation Model Training Convergence](loss_curve_ppt.jpg)

*The training curve (100 epochs) demonstrates stable convergence of the `DualStream_AttnUConvLSTM` architecture. Early validation volatility (red dashed line) is standard due to the highly sparse nature of localized precipitation anomalies. However, guided by the custom `PerfectClimateLoss` (Huber + SSIM), the validation loss successfully stabilizes and converges tightly alongside the training loss post-epoch 80, proving the model successfully generalizes without falling into a dry-bias trap.*

This Digital Twin bypasses standard statistical averaging by separating heat and moisture into independent processing streams. It predicts spatiotemporal trajectories using a 14-day sliding window of historical multi-dimensional telemetry:
* Gridded Precipitation ($0.25^\circ \times 0.25^\circ$)
* Maximum Temperature ($1.0^\circ \times 1.0^\circ$)
* Minimum Temperature ($1.0^\circ \times 1.0^\circ$)

## Machine Learning Architecture
* **Competitors:** Standard Spatiotemporal CNNs vs. `DualStream_AttnUConvLSTM`
* **Winner:** The Dual-Stream network successfully preserves thermodynamic and hydrological physical laws, merging them via a Multi-Head Cross-Attention Bridge before temporal propagation.
* **Optimization:** Optimized using `Optuna` Bayesian tuning and guided by a custom `PerfectClimateLoss` function, which applies a 10x penalty weight on active storm pixels to forcefully prevent dry-bias overfitting.

##  How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/CLIMATE-TWIN.git](https://github.com/YOUR_USERNAME/CLIMATE-TWIN.git)
cd CLIMATE-TWIN
```
### 2. Download the IMD Dataset
```bash
Before running the notebook, you must acquire the raw gridded data from the Indian Meteorological Department (IMD):Visit the IMD Pune Data Portal (or your designated access node).Download the high-resolution daily gridded files (.GRD or .nc format) for Rainfall ($0.25^\circ$) and Temperature ($1.0^\circ$).Place these files into the root directory under a new folder named data/raw/.
```
### 3. Install the dependencies
```bash
. Install the dependencies
```
### 4. Launch the Prototype Environment
```bash
Start the Jupyter environment to execute the baseline preprocessing and model pipeline:
jupyter notebook baseline_preprocess.ipynb
```
