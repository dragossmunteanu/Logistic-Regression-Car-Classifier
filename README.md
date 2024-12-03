### README.md

# 🚗 Car Evaluation with Machine Learning Models

This project showcases the application of machine learning algorithms to classify car evaluation ratings using the **Car Evaluation Dataset** from the UCI Machine Learning Repository. The implementation uses **Random Forest Classifier** and **Logistic Regression** models, accompanied by preprocessing techniques like One-Hot Encoding and Label Encoding. 📊

---

## 📂 Dataset Overview

The **Car Evaluation Dataset** contains categorical features that describe the acceptability of cars based on various attributes. The dataset is directly fetched using the `ucimlrepo` library.

- **Source**: UCI Machine Learning Repository  
- **ID**: 19  
- **Features**:  
  - Buying price  
  - Maintenance cost  
  - Number of doors  
  - Passenger capacity  
  - Luggage boot size  
  - Safety  

- **Target**: Car acceptability (e.g., unacceptable, acceptable, good, very good)  

For detailed metadata and variable descriptions, the script includes methods to display this information.  

---

## 🛠 Features of the Project

1. **Data Loading**:  
   - The dataset is imported and split into features (`X`) and target labels (`y`).  

2. **Preprocessing**:  
   - Handles categorical data using two techniques:  
     - **One-Hot Encoding**: Converts categorical variables into binary format.  
     - **Label Encoding**: Assigns numerical labels to categories.  

3. **Modeling**:  
   - **Random Forest Classifier** 🌳  
     - Ensemble method with 100 estimators (configurable).  
   - **Logistic Regression** ➡️  
     - A baseline classifier for comparison.  

4. **Evaluation Metrics**:  
   - **Accuracy** ✅  
   - **Precision** 🎯  
   - **Recall** 🔄  
   - **F1-Score** 🏆  

5. **Performance Comparison**:  
   - Compares the performance of Random Forest and Logistic Regression on the dataset.  

---

## 🚀 How to Run

### Prerequisites
Make sure you have the following libraries installed:  
- `ucimlrepo`  
- `scikit-learn`  
- `pandas`  

Install them using pip:  
```bash
pip install ucimlrepo scikit-learn pandas
```

### Steps
1. Clone this repository.  
2. Run the script in your Python environment:  
   ```bash
   python car_evaluation.py
   ```

---

## 📈 Results

### Random Forest Classifier
- **Accuracy**: 93%  
- **Precision**: 92%  
- **Recall**: 91%  
- **F1-Score**: 91%  

### Logistic Regression
- **Accuracy**: 87%  
- **Precision**: 86%  
- **Recall**: 85%  
- **F1-Score**: 85%  

*Note: Exact results may vary depending on the random state and preprocessing technique used.*

---

## 🤖 Future Improvements

- Experiment with other classification algorithms like SVM or Neural Networks.  
- Perform hyperparameter tuning for better results.  
- Add exploratory data analysis (EDA) for deeper insights into the dataset.  

---

## 📜 License
This project is open-source and free to use under the MIT License.  

---

Feel free to reach out if you have questions or suggestions! 😊
