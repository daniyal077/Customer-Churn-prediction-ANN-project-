# Customer Churn Prediction App

## Overview
This project is a machine learning application designed to predict whether a customer is likely to exit (churn) or stay with a company. The app uses a neural network model trained on the "Churn_Modelling" dataset. The predictions are presented through an interactive Streamlit web application.

---

## Features
1. **Data Preprocessing**:
   - Handles categorical and numerical columns using OneHotEncoding and MinMaxScaler, respectively.
   - Balances imbalanced data using SMOTE (Synthetic Minority Oversampling Technique).
2. **Neural Network Model**:
   - Built using TensorFlow/Keras.
   - Includes multiple layers with dropout regularization to prevent overfitting.
   - Optimized with Adam optimizer and early stopping to improve training efficiency.
   - Achieves 80% accuracy on training data
3. **Streamlit Web Application**:
   - Allows users to input customer details via an intuitive sidebar interface.
   - Provides predictions on whether the customer will exit or stay.
4. **Saved Artifacts**:
   - Preprocessing pipeline (`preprocessor.pkl`).
   - Trained neural network model (`customer_churn_model.keras`).

---

## Installation
### Prerequisites
- Python 3.8 or above
- Libraries: TensorFlow, Keras, pandas, scikit-learn, imbalanced-learn, pickle, matplotlib, Streamlit

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/daniyal077/Customer-Churn-prediction-ANN-project-.git
   cd Churn-prediction-ANN-project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## Dataset
- The dataset used is `Churn_Modelling.csv`.
- Features include:
  - **Demographic**: Gender, Geography, Age.
  - **Financial**: CreditScore, Balance, EstimatedSalary.
  - **Behavioral**: Tenure, Number of Products, Active Membership, Credit Card ownership.
- Target: `Exited` (1 = customer churned, 0 = customer stayed).

---

## File Structure
```
.
├── Churn_Modelling.csv        # Dataset
├── app.py                     # Streamlit app
├── model_training.ipynb          # Model training script
├── preprocessor.pkl           # Preprocessing pipeline
├── customer_churn_model.keras # Trained model
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
```

---

## Usage
1. **Train the Model**:
   - Run `model_training.ipynb` to preprocess the data, train the neural network model, and save the artifacts.
     
2. **Run the Web App**:
   - Execute `app.py` to launch the Streamlit interface.
   - Input customer details on the sidebar and view predictions in real time.
   - Access the web application directly at: https://customer-churn-prediction-1o57.onrender.com/.
  


---

## How It Works
1. **Model Training**:
   - Preprocesses data by encoding categorical variables and scaling numerical ones.
   - Balances the dataset using SMOTE.
   - Trains a neural network model with dropout layers for regularization.
2. **Prediction**:
   - The Streamlit app accepts user input.
   - Input data is transformed using the saved preprocessor.
   - The trained model predicts the likelihood of churn.

---


## Screenshots
### Streamlit App UI
- **Input Sidebar**:
  
  ![image](https://github.com/user-attachments/assets/d0eb0ed0-0d6f-420f-9528-566335343dbd)
- **Prediction Result**:
  ![image](https://github.com/user-attachments/assets/015e7157-6bf1-49e6-bfbb-a5b72e275238)


---

## Future Improvements
- Add more features like visualizations of customer data.
- Improve the model using advanced architectures like XGBoost or ensemble methods.
- Deploy the app on platforms like AWS, GCP, or Heroku for broader accessibility.

---


## License
This project is licensed under the MIT License.

---

## Acknowledgments
- Dataset source: Kaggle
- Libraries: TensorFlow, Keras, Streamlit, Scikit-learn, Imbalanced-learn

