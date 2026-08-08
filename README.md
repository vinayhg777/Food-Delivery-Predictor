# Food Delivery Time Prediction

A Flask web application that estimates delivery time for a courier service using a pre-trained machine learning model. The app accepts inputs such as distance, traffic level, preparation time, courier experience, weather, time of day, and vehicle type.

## Project Structure

- `app.py` - Flask application with prediction logic.
- `templates/` - HTML templates for the web pages.
- `static/` - CSS, JavaScript, and library assets.
- `delivery_time_model.pkl` - Trained delivery time prediction model.
- `traffic_encoder.pkl` - Encoder for traffic level values.
- `model_columns.pkl` - Model column order metadata.
- `delivery_sacler.pkl` - Scaler used to normalize input features.

## Prerequisites

- Python 3.8+ (or compatible Python 3.x)
- `pip` package manager

## Installation

1. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install required packages:

```bash
pip install flask pandas joblib
```

## Running the App

From the project root directory, run:

```bash
python app.py
```

By default, the app will start on `http://0.0.0.0:4000`. Open your browser and navigate to:

```text
http://127.0.0.1:4000/
```

## Usage

1. Go to the home page.
2. Enter the delivery details on the prediction form.
3. Submit the form to receive the estimated delivery time.

## Notes

- The prediction code loads the required artifacts from the project root:
  - `delivery_time_model.pkl`
  - `traffic_encoder.pkl`
  - `model_columns.pkl`
  - `delivery_sacler.pkl`
- Ensure these files are present in the same folder as `app.py`.

## Troubleshooting

- If the app fails to start, verify that all required package dependencies are installed.
- If predictions fail, confirm the model artifact files are available and properly named.

## License

This project does not include a license file. Add one if you want to publish or share it publicly.
