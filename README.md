# Multiple Object Tracking for the FIRST Robotics Competition

This project focuses on applying **state-of-the-art Multi-Object Tracking (MOT)** methods to video footage from the **FIRST Robotics Competition (FRC)**. The goal is to enable **automatic tracking of robots** to support objective and scalable performance evaluation, replacing the currently **manual scoring process** .


## 📌 Project Objectives

* Fine-tune a **YOLO object detector** on custom-labeled FRC video frames
* Implement and evaluate **multiple state-of-the-art MOT models** on FRC footage
* Compare tracking performance across models using trajectory statistics
  

## 📁 Dataset & Annotation

* Video data is taken from official **FIRST Robotics Competition matches**
* Robot annotations were created using **Roboflow**
* These annotations were used to **fine-tune a YOLO detector** to improve detection quality
* This is critical since **tracking performance is highly dependent on detection quality**



## 🧠 Evaluated Tracking Models

The following five MOT models were implemented and compared:

### 1. Regular MOT (Baseline)

* Kalman filter with constant velocity
* IoU-based association
* Baseline for performance comparison

### 2. StrongSORT

* Improved appearance embeddings using **ResNeSt50 + Bag of Tricks**
* Confidence-aware Kalman filter
* More robust against noisy detections and identity switches

### 3. OC-SORT (Observation-Centric SORT)

* Uses YOLO detections
* **Observation-Centric Re-Update (ORU)** to correct drift after occlusions
* **Observation-Centric Momentum (OCM)**: angle-based motion consistency for association
* Designed for **robust tracking under occlusions and non-linear motion**

### 4. MeMOT (Lightweight Variant)

* Memory-based identity preservation
* Uses appearance embeddings
* Good short-term consistency, but sensitive to long occlusions

### 5. ByteTrack

* Two-stage association using **high- and low-confidence detections**
* No appearance model
* Very robust to occlusion and well-suited for **low-quality, fast-motion FRC footage**



## 📊 Model Evaluation

The models are evaluated based on:

* Total number of tracked IDs
* Average track lifetime
* Maximum track lifetime
* Gap count (track interruptions)

Trajectory visualizations and comparative statistics were generated for:

* Regular MOT
* ByteTrack
* MeMOT
* OC-SORT
* StrongSORT

The evaluation shows clear differences in **identity stability, occlusion robustness, and track continuity** across models .



## ⚠️ Reproducibility Note (Important)

> **All experiments were conducted in Google Colab.**
> When reproducing this project locally, **file paths, dataset locations, and model checkpoints must be updated accordingly**, as Colab uses a different filesystem structure than local machines.



## 📚 References

*Buric, Matija, Marina Ivasic-Kos, and Miran Pobar. "Player tracking in sports videos." 2019 IEEE International 
Conference on Cloud Computing Technology and Science (CloudCom). IEEE, 2019.*

*Cao, Jinkun, et al. "Observation-centric sort: Rethinking sort for robust multi-object tracking." Proceedings of the 
IEEE/CVF conference on computer vision and pattern recognition. 2023.*

*Cai, Jiarui, et al. "Memot: Multi-object tracking with memory." Proceedings of the IEEE/CVF conference on 
computer vision and pattern recognition. 2022.*

*Du, Yunhao, et al. "Strongsort: Make deepsort great again." IEEE Transactions on Multimedia 25 (2023): 8725-
8737.*

*Zhang, Yifu, et al. "Bytetrack: Multi-object tracking by associating every detection box." European conference on 
computer vision. Cham: Springer Nature Switzerland, 2022*

*FIRST Robotics Competition: [https://www.firstinspires.org/programs/frc](https://www.firstinspires.org/programs/frc)*


