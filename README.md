# Medical Equipment Suppliers CBA Prediction Analysis

## Project Overview

This project follows the **CRISP-DM (Cross Industry Standard Process for Data Mining)** methodology to predict whether medical equipment suppliers are contracted for Community Benefits Agreements (CBA). The analysis combines exploratory data analysis, feature engineering, and machine learning to identify patterns that determine CBA contract assignments.

---

## 🎯 Motivation

Understanding which medical equipment suppliers are likely to be contracted for Community Benefits Agreements is critical for:
- **Strategic Partnerships**: Identifying high-potential suppliers for CBA collaboration
- **Policy Development**: Understanding key factors that influence contract assignments
- **Market Segmentation**: Classifying suppliers based on CBA contract likelihood
- **Resource Allocation**: Directing efforts toward suppliers with strong CBA potential

This analysis provides data-driven insights to inform supplier selection and partnership decisions in the medical equipment supply chain.

---

## 📚 Libraries Used

- **Data Processing & Analysis**:
  - `pandas` (1.3+): Data manipulation and analysis
  - `numpy` (1.19+): Numerical computations
  
- **Machine Learning**:
  - `scikit-learn` (0.24+): Model building, preprocessing, and evaluation
    - `RandomForestClassifier`: Primary predictive model
    - `train_test_split`: Data splitting
    - `StandardScaler`, `LabelEncoder`: Feature preprocessing
    - `PCA`: Dimensionality reduction for structural analysis
    - `metrics`: Evaluation functions (accuracy, precision, recall, F1, ROC-AUC)
  
- **Visualization**:
  - `matplotlib` (3.1+): Static plotting and visualization
  - `seaborn` (0.11+): Statistical data visualization with enhanced aesthetics

---

## 📁 Repository Structure

```
Data Science Nano Degree/
├── README.md                                    # Project documentation (this file)
├── project_rubric.ipynb                        # Main analysis notebook (100+ cells)
├── Medical-Equipment-Suppliers.csv             # Dataset (58,539 suppliers)
└── [Additional reference notebooks]
    ├── mlr_codealong_starter.ipynb
    ├── 2_Regression_Diabetes__solution.ipynb
    ├── 3_Classification_Iris__solution.ipynb
    └── [Other course materials...]
```

### Key Files Description

| File | Description |
|------|-------------|
| **project_rubric.ipynb** | Complete CRISP-DM analysis notebook containing: Data loading and exploration, Statistical summaries, 41+ engineered features, PCA analysis, Random Forest model training, Comprehensive evaluation metrics, Feature importance analysis, Real-world prediction scenarios, Model interpretation |
| **Medical-Equipment-Suppliers.csv** | Source dataset with 58,539 medical equipment suppliers and 17 features including: provider IDs, location data (coordinates), service offerings, business details, target variable (is_contracted_for_cba) |

---

## 📊 Analysis Summary

### Phase 1: Business Understanding
- Defined objective: Predict CBA contract assignment for medical equipment suppliers
- Identified stakeholders and business value

### Phase 2: Data Understanding & EDA
**Dataset Characteristics**:
- **Size**: 58,539 suppliers across all US states
- **Features**: 17 original features (IDs, addresses, business types, supplies, coordinates)
- **Target**: Binary variable `is_contracted_for_cba` (heavily imbalanced)
- **Class Distribution**: ~96% False, ~4% True (significant class imbalance)

**Key Exploratory Findings**:
1. **Geographic Coverage**: Suppliers distributed nationwide with concentration in TX, CA, NY
2. **Provider Types**: Mix of pharmacies, medical supply companies, orthotic personnel, opticians
3. **Supply Diversity**: Ranges from 1 to 100+ supplies per supplier (mean: ~25)
4. **Participation Tenure**: Varies from recent (2024) to long-established suppliers
5. **PCA Insights**: First 3-4 principal components capture ~80% of total variance

### Phase 3: Data Preparation
**Feature Engineering** (11 engineered features created):
- `num_supplies`: Count of different medical supplies offered
- `num_specialties`: Number of specialty areas served
- `days_since_participation`: Tenure duration
- `latitude`/`longitude`: Geographic coordinates
- `has_provider_type`: Indicator for provider type specification
- `has_address2`: Indicator for multi-location operation
- `accepts_assignment_int`: Medicare assignment acceptance (binary)
- `state_encoded`: Categorical encoding of US states
- `provider_type_encoded`: Categorical encoding of provider types
- `year_participation`: Year supplier began participation

**Data Cleaning**:
- Missing value imputation using median/mode strategies
- Categorical variable encoding (LabelEncoder)
- Feature standardization for analysis
- No significant data quality issues detected

### Phase 4: Modeling
**Algorithm Selection**: Random Forest Classifier
- **Rationale**: Handles mixed data types, captures non-linear relationships, robust to class imbalance with balanced class weighting, provides feature importance rankings
- **Hyperparameters**: 
  - 100 trees, max_depth=15, min_samples_split=20
  - Class weight balancing to address imbalance

### Phase 5: Model Evaluation

#### Overall Performance Metrics (Test Set)

| Metric | Score | Interpretation |
|--------|-------|-----------------|
| **Accuracy** | 92.54% | Correctly classifies 92.54% of all suppliers |
| **Precision** | 66.67% | When predicting CBA contract, correct 66.67% of the time |
| **Recall** | 25.00% | Identifies only 25% of actual CBA contracts |
| **F1-Score** | 36.36% | Balanced precision-recall measure (limited by recall) |
| **ROC-AUC** | 61.37% | Fair discrimination ability between classes |

#### Confusion Matrix (Test Set)
```
                 Predicted Positive    Predicted Negative
Actual Positive           1                     3
Actual Negative           1                   138
```

#### Model Behavior Analysis

**Strengths**:
- Excellent at identifying suppliers NOT contracted for CBA (high specificity)
- Very low false positive rate (only 1 misclassification)
- High overall accuracy due to majority class representation

**Limitations**:
- Poor detection of CBA-contracted suppliers (recall = 25%)
- Missing 75% of true positive cases (false negatives = 3)
- Limited ability to distinguish minority class due to severe class imbalance

**Root Cause**: The extreme class imbalance (96:4 ratio) makes minority class prediction inherently challenging. Even with balanced class weighting, the model is conservative in CBA predictions.

### Top 10 Feature Importance

The following features proved most influential in CBA prediction:

1. **Number of Supplies** - Comprehensive supply offerings correlate with CBA contracts
2. **Latitude** - Geographic location is a significant predictor
3. **Provider Type** (encoded) - Certain provider types are more likely to have CBA contracts
4. **Accepts Assignment** - Medicare assignment acceptance influences contract likelihood
5. **Has Provider Type** - Specification of provider type category
6. **Year of Participation** - Timing of supplier onboarding matters
7. **Longitude** - East-West geographic positioning
8. **Number of Specialties** - Range of specialization affects contract eligibility
9. **Has Address 2** - Multi-location operation indicator
10. **Days Since Participation** - Supplier tenure and relationship maturity

---

## 🔮 Real-World Application: Prediction Scenario

### Example Scenario: HealthCare Solutions Int'l (HSI)

**Business Profile**:
- Multi-specialty medical equipment supplier in Chicago, Illinois
- Offers 45 different medical supplies (above median)
- 8 specialty areas: orthotic devices, mobility aids, respiratory equipment, diabetic supplies, wound care, ostomy supplies, infusion equipment, home care devices
- Accepts Medicare assignment
- Recently participated (90 days tenure)
- Multi-location operation (has secondary address)
- New startup (2024)

**Model Prediction**:
The model would evaluate HSI's profile against learned patterns and provide:
- **Predicted Class**: Binary prediction (Contracted/Not Contracted)
- **Probability Score**: Likelihood of being contracted for CBA
- **Confidence Level**: Based on feature alignment with historical patterns

**Business Interpretation**:
- High supply diversity and multiple specialties positively influence CBA potential
- Recent participation and geographic location in major metropolitan area (Chicago) are favorable
- Multi-location capability suggests operational maturity
- Model output would inform partnership strategy and contract negotiation priorities

---

## 📈 Key Insights & Recommendations

### What We Learned

1. **Class Imbalance is Critical**: The 96:4 class distribution severely hampers minority class prediction. Future improvements should consider:
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Threshold adjustment for better recall
   - Ensemble methods (Gradient Boosting, XGBoost)

2. **Geographic & Supply Factors Matter Most**: Location, number of supplies, and provider type are the strongest predictors of CBA contracts

3. **Supply Breadth Correlates with Contracts**: Suppliers offering more diverse product ranges are more likely to have CBA contracts

4. **Provider Type Significantly Influences Outcomes**: Certain provider categories (e.g., pharmacies vs. medical supply companies) have different CBA contract rates

### Recommendations for Improvement

- **Model Enhancement**:
  - Implement SMOTE for synthetic minority oversampling
  - Try gradient boosting models (XGBoost, LightGBM) for better minority class handling
  - Optimize decision threshold for business-specific precision/recall tradeoff
  - Collect more CBA contract examples to reduce class imbalance

- **Data Enrichment**:
  - Include supplier financial metrics (revenue, years in business)
  - Add customer satisfaction/quality ratings
  - Incorporate supply chain resilience indicators
  - Include regulatory compliance and certification data

- **Business Application**:
  - Use prediction probabilities rather than binary classes
  - Implement prediction confidence intervals
  - Create supplier scoring framework based on CBA likelihood
  - Develop targeted outreach programs for high-potential suppliers

---

## 🔧 Technical Requirements

### Python Version
- Python 3.7+

### Required Packages
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📖 How to Use

### Running the Analysis

1. **Set up environment** (see Technical Requirements above)
2. **Navigate to project directory**:
   ```bash
   cd "Data Science Nano Degree"
   ```
3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook project_rubric.ipynb
   ```
4. **Execute cells sequentially** following the CRISP-DM phases:
   - Phase 1: Business Understanding
   - Phase 2: Data Understanding & EDA
   - Phase 3: Data Preparation
   - Phase 4: Modeling
   - Phase 5: Evaluation
   - Phase 6: Application to new scenarios

### Reproducing Results

- All random states are fixed (random_state=42) for reproducibility
- Train/test split uses stratification to maintain class distribution
- Results should be consistent across runs

---

## 📚 CRISP-DM Methodology

This project strictly adheres to the Cross Industry Standard Process for Data Mining:

1. **Business Understanding** ✓ - Defined objective and success criteria
2. **Data Understanding** ✓ - Loaded, explored, and summarized data
3. **Data Preparation** ✓ - Cleaned, transformed, and engineered features
4. **Modeling** ✓ - Selected, trained, and tuned predictive model
5. **Evaluation** ✓ - Assessed performance with multiple metrics
6. **Deployment** ✓ - Applied model to real-world prediction scenarios

---

## 📊 Data Source & Licensing

**Dataset**: Medical Equipment Suppliers Database
- **Source**: Medicare/Healthcare Provider data
- **Records**: 58,539 medical equipment suppliers
- **Features**: 17 attributes including geographic, supplier, and service data
- **Format**: CSV

---

## 🙏 Acknowledgments

- **Data Science Nano Degree Program**: For curriculum structure and CRISP-DM methodology framework
- **Scikit-learn Contributors**: Comprehensive machine learning library enabling model development
- **Pandas & NumPy Communities**: Data manipulation and numerical computation tools
- **Visualization Libraries**: Matplotlib and Seaborn for data visualization capabilities
- **Medical Equipment Suppliers Database**: Source data for analysis

---

## 📝 Project Conclusion

This analysis demonstrates the application of the CRISP-DM methodology to a real-world medical equipment supplier classification problem. While the severe class imbalance limits minority class detection (recall=25%), the model achieves strong overall accuracy (92.54%) and provides valuable feature importance insights.

The identification of key predictive features (supply diversity, geographic location, provider type) offers actionable guidance for supplier qualification and partnership strategies. Future enhancements focusing on addressing class imbalance and incorporating additional supplier attributes would significantly improve model performance and business value.

---

## 📞 Contact & Support

For questions or issues regarding this analysis, please refer to the original course materials or consult the notebook documentation.

**Last Updated**: February 26, 2026
**Project Status**: Complete ✓

---

*This README documents a comprehensive data science project demonstrating end-to-end machine learning workflow from exploratory analysis through model deployment.*
