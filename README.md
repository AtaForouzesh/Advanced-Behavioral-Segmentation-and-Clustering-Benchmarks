# Advanced-Behavioral-Segmentation-and-Clustering-Benchmarks

Applying Unsupervised Machine Learning to extract business insights. Features K-Means and Agglomerative Hierarchical clustering for customer and driver risk segmentation.

---

## 🛠️ Mathematical Concepts & Frameworks Implemented

### 1. K-Means Clustering
* **Optimization:** Leveraged the `k-means++` initialization technique to resolve the random initialization trap, ensuring faster and globally stable convergence.
* **Objective:** Minimizes the Within-Cluster Sum of Squares (WCSS) to partition data into $K$ spherical clusters.

### 2. Agglomerative Hierarchical Clustering
* **Linkage Criteria:** Implemented **Ward’s Linkage Method**, which minimizes the total within-cluster variance at each step of the hierarchy merge.
* **Distance Metric:** Utilized standard **Euclidean Distance** to quantify point-to-point dissimilarity.
* **Dendrogram Analysis:** Generated tree-structured taxonomies to visually determine optimal cut-off thresholds for cluster selection.

---

## 📊 Comparative Analysis

### Case Study 1: Instagram User Engagement Segmentation
* **Dataset:** `Instagram visits clustering.csv`
* **Features Analysed:** `Instagram Visit Score` vs. `Spending Rank`
* **Algorithm Outcomes:**
  * **K-Means ($k=5$):** Generated balanced, hyper-spherical groupings optimal for discrete customer tier categorization.
  * **Hierarchical ($k=5$):** Constructed a bottom-up taxonomy. The generated **Dendrogram** confirmed that splitting the user base into 5 clusters captures the most natural divergence in consumer spending behavior.
* **Business Use-Case:** Enables marketing teams to differentiate between *High-Engagement/Low-Spenders* (brand awareness targets) and *High-Engagement/High-Spenders* (premium conversion targets).

### Case Study 2: Fleet Logistics Fleet Risk Management
* **Dataset:** `driver-data.csv`
* **Features Analysed:** `Mean Distance Driven per Day` vs. `Mean Over-Speed Percentage`
* **Algorithm Outcomes:**
  * **K-Means ($k=4$):** Grouped corporate fleet drivers evenly into neat quadrants.
  * **Hierarchical ($k=4$):** Built a hierarchical tree structure that isolates extreme driving anomalies, verifying that 4 primary risk behavior sets exist.
* **Business Use-Case:** Allows insurance and operations managers to pinpoint high-risk, long-haul commercial drivers needing immediate safety interventions.
