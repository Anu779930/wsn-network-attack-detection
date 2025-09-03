# WSN Network Attack Detection  

ML-based detection of Wormhole & Sybil attacks in Wireless Sensor Networks (WHASA dataset) using **Random Forest, XGBoost, AdaBoost, and KNN**.  
Built with Python (Tkinter GUI), Scikit-learn, XGBoost, Pandas, and Seaborn.  

---

## 🚀 Project Highlights  
- Detection of multiple **network layer attacks** (Sybil, Wormhole, etc.) from WSN traffic  
- GUI-based interface built with **Tkinter** for dataset upload, preprocessing, model training & testing  
- Evaluation of **4 ML classifiers** (Random Forest, XGBoost, AdaBoost, KNN)  
- **Confusion matrix visualization** and performance metrics for each algorithm  
- Saved trained models for reuse with Pickle (`.txt` format)  

---

## 🛠️ Tech & Skills Demonstrated  
- **Python ML Libraries**: Scikit-learn, XGBoost, AdaBoost, Random Forest, KNN  
- **GUI**: Tkinter-based interface with real-time results display  
- **Data Processing**: Pandas, NumPy, Label Encoding, Normalization, PCA  
- **Visualization**: Seaborn (confusion matrices), Matplotlib (metrics comparison)  
- **Deployment Ready**: Requirements file, modular source code, dataset integration  

---

## 📂 Files in Repo  

- `data/` → Dataset CSV files (Sybil, Wormhole attack traffic)  
- `src/gui_app.py` → Main GUI + ML logic  
- `images/` → GUI screenshots, confusion matrices, and result captures  
- `requirements.txt` → Python dependencies  
- `packages.txt` → List of required packages  
- `README.md` → Project documentation (this file)  
- `LICENSE` → MIT license  

---

## 🖼️ GUI & Results Preview  

### Start Screen  
![Start Screen](https://raw.githubusercontent.com/Anu779930/wsn-network-attack-detection/main/images/gui_start_screen.png)  

### Upload Dataset  
![Upload Dataset](https://raw.githubusercontent.com/Anu779930/wsn-network-attack-detection/main/images/dataset_upload_in_gui.png)  

### Preprocessing Dataset  
![Preprocessing](https://raw.githubusercontent.com/Anu779930/wsn-network-attack-detection/main/images/dataprocessing_in_gui.png)  

---

### Confusion Matrix – XGBoost  
![XGBoost](https://raw.githubusercontent.com/Anu779930/wsn-network-attack-detection/main/images/XGBoost_graph.png)  
![XGBoost in GUI](https://raw.githubusercontent.com/Anu779930/wsn-network-attack-detection/main/images/XGBoost_in_gui.png)  

### Confusion Matrix – KNN  
![KNN](https://raw.githubusercontent.com/Anu779930/wsn-network-attack-detection/main/images/KNN_graph.png)  
![KNN in GUI](https://raw.githubusercontent.com/Anu779930/wsn-network-attack-detection/main/images/KNN_in_gui.png)  

### Confusion Matrix – AdaBoost  
![AdaBoost](https://raw.githubusercontent.com/Anu779930/wsn-network-attack-detection/main/images/AdaBoost_graph.png)  
![AdaBoost in GUI](https://raw.githubusercontent.com/Anu779930/wsn-network-attack-detection/main/images/AdaBoost_in_gui.png)  

### Confusion Matrix – RandomForest  
![RandomForest](https://raw.githubusercontent.com/Anu779930/wsn-network-attack-detection/main/images/RandomForest_graph.png)  
![RandomForest in GUI](https://raw.githubusercontent.com/Anu779930/wsn-network-attack-detection/main/images/RandomForest_in_gui.png)  

---

### Results Table  
![Results Table 1](https://raw.githubusercontent.com/Anu779930/wsn-network-attack-detection/main/images/Result1.png)  
![Results Table 2](https://raw.githubusercontent.com/Anu779930/wsn-network-attack-detection/main/images/Result2.png)  

---

## 🔮 Future Improvements  
- Expand to additional **network attack types**  
- Add **real-time streaming detection**  
- Optimize PCA and feature engineering for higher accuracy  
- Build a **lighter model version** for IoT/embedded deployment  

---

## 📜 License  
This project is licensed under the MIT License.  
