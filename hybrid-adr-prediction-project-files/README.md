# Predicting Adverse Drug Reactions Using Machine Learning: A Hybrid Model Stacking Approach

**This is the official repository for the paper "Predicting Adverse Drug Reactions Using Machine Learning: A Hybrid Model Stacking Approach," published by IEEE.**

---

## Publication
This work has been published by IEEE. For a detailed explanation of the methodology and a comprehensive discussion of the results, please refer to the official paper.

**[Click here to view the publication on IEEE Xplore](https://ieeexplore.ieee.org/document/11086311)**

---

## 1. Problem Statement
Adverse Drug Reactions (ADRs) are a significant cause of morbidity and mortality worldwide, posing a major challenge to patient safety and placing a heavy burden on healthcare systems. The ability to prospectively identify high-risk combinations of drugs and patient indications can be a crucial step in mitigating these risks. This project addresses the critical need for an accurate and robust computational tool to predict the probability of an ADR, thereby enabling better clinical decision-making.

---

## 2. Overview
This project introduces a novel hybrid machine learning model to predict the probability of an ADR based on a given drug and its medical indication. Our solution is a **stacked ensemble (or hybrid) model** that combines the predictive capabilities of three powerful gradient boosting algorithms—LightGBM, XGBoost, and CatBoost. The outputs of these base models are then used to train a final Random Forest meta-model, which produces a more accurate and generalized prediction than any single model could achieve alone.

### Key Features
* [cite_start]**Advanced Feature Engineering**: The model's predictive power is enhanced by sophisticated feature engineering, including TF-IDF vectorization for text data and the creation of novel co-occurrence metrics to capture the complex relationships between drugs, indications, and side effects[cite: 68, 87, 91].
* [cite_start]**Hybrid Stacking Ensemble**: Our approach leverages a stacking architecture to combine diverse, high-performing models, effectively reducing bias and variance to create a more robust final predictor[cite: 443, 444].
* [cite_start]**Data-Driven Approach**: The model is trained on a comprehensive dataset constructed by merging multiple real-world data sources, providing a rich foundation for identifying patterns in ADR reporting[cite: 589].

---

## 3. Technology Stack
This project was implemented in Python 3 and utilizes the following key libraries and frameworks:
* **Data Manipulation & Analysis**: Pandas, NumPy
* **Machine Learning & Modeling**: Scikit-learn, LightGBM, XGBoost, CatBoost
* **Data Visualization**: Matplotlib, Seaborn
* **Model Persistence**: Joblib

---

## 4. Methodology
The methodology follows a multi-stage pipeline, from data preparation to model evaluation, as outlined in our paper.

### 4.1. Data Processing and EDA
[cite_start]The initial phase involved constructing a cohesive dataset from three separate sources: drug names, medical indications, and side effects[cite: 522, 531, 568]. [cite_start]To address the inherent imbalance in the data, a downsampling strategy was employed: records of rare side effects were kept, while common side effects were randomly sampled to create a more balanced and computationally efficient dataset for training[cite: 51, 56, 58].

### 4.2. Feature Engineering
Several features were engineered to provide the models with rich, predictive signals:
* [cite_start]**TF-IDF Vectorization**: The textual `side_effect_name` and `meddra_indication_name` columns were transformed into 50-dimensional numerical vectors using Term Frequency-Inverse Document Frequency (TF-IDF)[cite: 68, 69, 70].
* **Frequency and Co-occurrence Metrics**:
    * [cite_start]`side_effect_freq`: The normalized frequency of each side effect in the dataset[cite: 76, 82].
    * [cite_start]`indication_side_effect_relevance`: A metric capturing how often a specific side effect and indication appear together[cite: 83, 87].
    * [cite_start]`drug_side_effect_relevance`: A metric for the co-occurrence of a specific drug and side effect[cite: 88, 91].
* [cite_start]**Target Variable**: The final target variable, `adverse_reaction_probability`, was synthesized from a weighted combination of these engineered features[cite: 95, 98, 99, 100].

### 4.3. Modeling: The Stacked Ensemble
Our hybrid model consists of two layers:
1.  [cite_start]**Level 0 (Base Models)**: Three distinct gradient boosting models were trained independently on the feature set[cite: 128, 168, 198]:
    * [cite_start]`LGBMRegressor` [cite: 120]
    * [cite_start]`XGBRegressor` [cite: 163]
    * [cite_start]`CatBoostRegressor` [cite: 197]
2.  [cite_start]**Level 1 (Meta-Model)**: The predictions from these three base models were then stacked together and used as the input features to train a final `RandomForestRegressor`, which serves as the meta-model[cite: 443, 444, 445]. This model learns how to best combine the base predictions to generate the final, more accurate output.

---

## 5. Results
The stacked meta-model demonstrated vastly superior performance across all evaluation metrics when compared to the individual base models. The results below clearly show the effectiveness of the stacking approach.

| Model | R-Squared ($R^2$) Score | Mean Squared Error (MSE) | Mean Absolute Error (MAE) |
| :--- | :---: | :---: | :---: |
| LightGBM | 0.456 | 0.00371 | 0.0420 |
| XGBoost | 0.451 | 0.00374 | 0.0422 |
| CatBoost | 0.537 | 0.00315 | 0.0387 |
| **Meta-Model (Stacking)** | **0.879** | **0.00083** | **0.0183** |

As shown, the meta-model increased the **$R^2$ score to 0.879**, explaining nearly 88% of the variance in the data. Furthermore, it reduced the **Mean Squared Error by over 73%** compared to the best-performing base model (CatBoost), indicating a significant improvement in predictive accuracy.

---

## 6. How to Replicate

### Option A: Run with Docker (Recommended)
1. Ensure you have [Docker](https://docs.docker.com/get-docker/) installed.
2. Clone the repository:
    ```bash
    git clone https://github.com/<your-username>/adr-prediction-stacked-ensemble.git
    cd adr-prediction-stacked-ensemble
    ```
3. Build the image:
    ```bash
    docker compose build
    ```
4. Run the container:
    ```bash
    docker compose up
    ```
5. Open [http://localhost:8501](http://localhost:8501) in your browser.

The Docker build process will automatically extract the compressed project files (models, TF-IDF, encoders, scalers, data, etc.), so no manual unzipping is required.

---

### Option B: Manual Setup (Alternative)
1. Ensure Git LFS is installed on your system, as the zip file is tracked using Git LFS.
2. Clone the repository:
    ```bash
    git clone https://github.com/<your-username>/adr-prediction-stacked-ensemble.git
    cd adr-prediction-stacked-ensemble
    ```
3. Manually extract the `hybrid-adr-prediction.zip` file into the root of the cloned repository. The extracted folder (`hybrid-adr-prediction-fresh/`) must remain intact with the following structure:
    ```
    hybrid-adr-prediction-fresh/
    ├── app.py
    ├── app_inference_utils.py
    ├── requirements.txt
    ├── models/
    ├── tfidfs/
    ├── one_hot_encoder/
    ├── scaler/
    ├── data/
    └── notebooks/
    ```
4. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

---

## 7. Future Work
While this model provides a strong foundation, future research could explore several promising directions:
* **Integration of Additional Data Sources**: Incorporating patient demographic data, genomic information, or drug-drug interaction databases could further enhance predictive accuracy.
* **Exploration of Deep Learning**: Advanced deep learning architectures, such as Transformers or Graph Neural Networks, could be explored to better capture the complex relationships in the data.
* **Real-time API Deployment**: Deploying the model as a scalable, real-time API would allow for its integration into clinical decision support systems and other healthcare applications.

---

## 8. Citation
If you use the code or findings from this project in your research, please cite our paper.

T. A and B. P, "Predicting Adverse Drug Reactions Using Machine Learning: A Hybrid Model Stacking Approach," 2025 Second International Conference on Cognitive Robotics and Intelligent Systems (ICC - ROBINS), Coimbatore, India, 2025, pp. 645-650, doi: 10.1109/ICC-ROBINS64345.2025.11086311.
keywords: {Measurement;Drugs;Stacking;Medical services;Predictive models;Prediction algorithms;Data models;Reliability;Pharmaceutical industry;Random forests;Adverse drug reactions;machine learning;feature engineering;ensemble learning;LightGBM;XGBoost;CatBoost;stacking;meta model;random forest regressor},

---

## 9. Disclaimer

This project is intended **solely for research and educational purposes**. While it leverages real-world datasets and rigorous machine learning techniques, the predictions generated by this system **should not be used for clinical decision-making, diagnosis, or treatment of patients**. Users are advised to consult qualified healthcare professionals for any medical decisions. The authors are not responsible for any outcomes resulting from the use of this software in real-world medical applications.

---

## 10. License

This project is licensed under the **Apache License 2.0**. You may obtain a copy of the license at:

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an **"AS IS" BASIS**, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
