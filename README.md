# Stock Price Prediction Web App (LSTM)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

---

## Project Overview

This project is a Stock Price Prediction Web Application built using Machine Learning (LSTM) and Streamlit.

It predicts future stock prices based on historical data and provides interactive visualizations.



## Features

* Predict stock prices using historical data
* Future trend visualization
* Upload custom dataset (CSV format)
* Interactive graphs using Plotly
* Simple web interface using Streamlit

---

## Tech Stack

* Python
* TensorFlow / Keras
* NumPy & Pandas
* Matplotlib / Plotly
* Scikit-learn
* Streamlit

---

## Project Structure

```
├── app.py
├── model.py
├── AAPL_sample.csv
├── requirements.txt
├── screenshots/
│   ├── home.png
│   ├── prediction.png
│   └── upload.png
└── README.md
```

---

## How It Works

1. Load historical stock data
2. Preprocess data (scaling and sequence creation)
3. Train LSTM model
4. Generate predictions
5. Visualize results

---

## How to Run the Project

```bash
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction
```

```bash
python -m venv venv
venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

---

## Model Details

* Model: LSTM
* Input: Historical stock prices
* Output: Predicted future prices
* Evaluation Metric: Mean Squared Error (MSE)

---

## Future Improvements

* Add live stock API integration
* Support multiple stocks
* Deploy using cloud platforms
* Improve model accuracy

---

## Author

Regunathan

---

## License

This project is licensed under the MIT License.

---

## Support

If you find this project useful, consider giving it a star.
