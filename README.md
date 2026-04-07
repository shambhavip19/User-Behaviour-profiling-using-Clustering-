Here you go:

---

# User Behavior Profiling using K-Means Clustering and PCA
Grouping mobile phone users based on their digital behavior patterns using unsupervised learning.

## Dataset
Mobile Device Usage and User Behavior Dataset with 700 user records and 11 features including app usage time, screen time, battery drain, and demographic info.

Source: **Kaggle**

Features used: `App Usage Time (min/day)`, `Screen On Time (hours/day)`, `Battery Drain (mAh/day)`, `Number of Apps Installed`, `Data Usage (MB/day)`, `Age`

## Workflow
1. Load and explore the data, shape, missing values, column types
2. Select relevant behavioral features and drop non-informative columns
3. Preprocessing, feature scaling using StandardScaler
4. Experiment with K=3, K=4, K=5 to find the right number of clusters
5. Apply K-Means and assign cluster labels to each user
6. Visualize clusters in 2D using PCA
7. Interpret each cluster by analyzing average feature values

## Models Used
**K-Means Clustering**
- Scaled features using `StandardScaler` before clustering
- Tested K=3, K=4, K=5 and compared cluster interpretability

**PCA (Principal Component Analysis)**
- Used only for visualization, reduces 6 features to 2D
- Does not affect the actual clustering

## Results
| Cluster | App Usage | Screen Time | Data Usage | Avg Age | Profile |
|---|---|---|---|---|---|
| 0 | 321 min | 6.0 hrs | 1039 MB | 49 | Moderate Older Users |
| 1 | 541 min | 10.1 hrs | 1975 MB | 38 | Power Users |
| 2 | 97 min | 2.3 hrs | 331 MB | 38 | Minimal Users |
| 3 | 307 min | 5.8 hrs | 1008 MB | 27 | Active Young Users |

K=4 gave the most interpretable clusters. K=3 collapsed age differences entirely, and K=5 began splitting groups that had no meaningful behavioral difference.

## Libraries
- `pandas`, `numpy` for data handling
- `matplotlib` for visualization
- `scikit-learn` for clustering, PCA, and preprocessing
- `streamlit`, `joblib` for web app and model export

## What I Learned
- How unsupervised learning differs from supervised, no labels, no right answer
- Why feature scaling is critical before running K-Means
- How PCA works as a visualization tool without affecting the model
- How to export a trained model and build a working web app around it
