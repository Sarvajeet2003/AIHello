# AIHello

# Reinforcement Learning for Price Optimization

## Overview
This project aims to optimize the pricing of a product using reinforcement learning (RL). The goal is to predict the ideal price for maximizing sales and conversions, while also encouraging exploration beyond historical pricing trends.

## Features
- **Historical Data Processing**: Clean and normalize pricing data with missing value handling.
- **Time-Series Forecasting**: Use ARIMA to forecast sales for future dates.
- **Reinforcement Learning**:
  - Custom environment built using OpenAI's Gym.
  - Reward function incentivizes higher sales, organic conversion, and price exploration.
- **Visualization**: Graphs of rewards over steps to evaluate performance.

---

## Dataset
### Input Format
The dataset should be a CSV file containing the following columns:
- **Report Date**: The date of the record.
- **Product Price**: Price of the product on that date.
- **Organic Conversion Percentage**: Conversion rate of the product.
- **Ad Conversion Percentage**: Ad conversion rate.
- **Total Profit**: Total profit for the day.
- **Total Sales**: Total sales on that day.
- **Predicted Sales**: Predicted sales for future dates.

Example:
| Report Date | Product Price | Organic Conversion Percentage | Ad Conversion Percentage | Total Profit | Total Sales | Predicted Sales |
|-------------|---------------|-------------------------------|---------------------------|--------------|-------------|-----------------|
| 2025-01-01  | 15.00         | 0.12                          | 0.08                      | 200.00       | 50.00       | 60.00           |

---

## Installation

### Prerequisites
- Python 3.8+
- Required Libraries:
  - `gym`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `statsmodels`
  - `scikit-learn`

### Setup
1. Clone the repository:
   ```bash
   git clone (http://github.com/Sarvajeet2003/AIHello)
   cd AIHello
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset in the project directory.

---

## How It Works

### 1. Preprocessing
- Handles missing values using forward and backward filling.
- Normalizes numeric columns for RL compatibility.

### 2. Forecasting
- Uses ARIMA to predict sales for future dates based on historical data.

### 3. RL Environment
- **State Space**: Product price, organic and ad conversion rates, total sales, and predicted sales.
- **Action Space**: Increment or decrement the price by up to 10%.
- **Reward Function**:
  - Rewards higher sales, organic conversion, and ad conversion.
  - Encourages exploration by rewarding prices above the historical median.

### 4. Visualization
- Displays reward trends over time to evaluate the agent's performance.

---

## Usage

### Running the Code
1. Execute the script:
   ```bash
   python3 a.py
   python3 b.py
   ```

2. The script will:
   - Preprocess the dataset.
   - Forecast sales for tomorrow.
   - Train and evaluate the RL agent.
   - Plot a reward trend graph.

---

## Results
- The RL agent dynamically adjusts prices to maximize sales and conversions.
- Rewards trends show the effectiveness of the exploration-exploitation balance.

---

## Future Improvements

1. Replace random actions with an advanced RL algorithm (e.g., Q-learning, PPO).
2. Incorporate dynamic reward weighting to balance between exploration and exploitation.
3. Use more sophisticated forecasting models like LSTM for better sales prediction.

---
