import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import gym
from gym import spaces

# Preprocessing Function
def preprocess_data(data):
    # Convert 'Report Date' to datetime format
    data['Report Date'] = pd.to_datetime(data['Report Date'])
    
    # Handle NaN values: Fill with interpolated values or mean
    data.fillna(method='ffill', inplace=True)  # Forward fill
    data.fillna(method='bfill', inplace=True)  # Backward fill
    
    # Normalize numeric columns
    numeric_columns = ['Product Price', 'Organic Conversion Percentage',
                       'Ad Conversion Percentage', 'Total Profit',
                       'Total Sales', 'Predicted Sales']
    
    scaler = MinMaxScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    return data, scaler

# Load data
data_path = 'soapnutshistory.csv'  # Replace with your file path
data = pd.read_csv(data_path)

# Preprocess data
cleaned_data, scaler = preprocess_data(data)

# Forecasting Function Using ARIMA
def forecast_sales(data, column, steps=1):
    series = data[column].dropna().values  # Ensure no NaNs
    model = ARIMA(series, order=(5, 1, 0))  # ARIMA(p, d, q)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Forecast sales for tomorrow
forecasted_sales = forecast_sales(cleaned_data, 'Total Sales', steps=1)
print(f"Forecasted Sales for Tomorrow: {forecasted_sales}")

# Define the RL Environment
class PriceOptimizationEnv(gym.Env):
    def __init__(self, data, scaler):
        super(PriceOptimizationEnv, self).__init__()
        self.data = data.reset_index(drop=True)  # Reset index
        self.scaler = scaler
        self.current_step = 0

        # Define action and observation space
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(5,),  # Product Price, Organic, Ad, Total Sales, Predicted Sales
            dtype=np.float32,
        )

    def reset(self):
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)[0]
        new_price = self.data.loc[self.current_step, "Product Price"] + action
        new_price = np.clip(new_price, 0, 1)
        self.data.loc[self.current_step, "Product Price"] = new_price

        # Fetch current data
        organic_conversion = self.data.loc[self.current_step, "Organic Conversion Percentage"]
        ad_conversion = self.data.loc[self.current_step, "Ad Conversion Percentage"]
        predicted_sales = self.data.loc[self.current_step, "Predicted Sales"]
        historical_median = self.data["Product Price"].median()

        # Reward: Sales + Conversion + Exploration
        sales = predicted_sales * (1 + (new_price - self.data.loc[self.current_step, "Product Price"]))
        reward = sales + organic_conversion + ad_conversion
        if new_price > historical_median:
            reward += 0.1  # Reward for exploring higher price points

        # Check if done
        self.current_step += 1
        done = self.current_step >= len(self.data)

        return self._get_observation() if not done else None, reward, done, {}

    def _get_observation(self):
        row = self.data.loc[self.current_step]
        return np.array([
            row["Product Price"],
            row["Organic Conversion Percentage"],
            row["Ad Conversion Percentage"],
            row["Total Sales"],
            row["Predicted Sales"]
        ])

# Initialize the environment
env = PriceOptimizationEnv(cleaned_data, scaler)

# Test the environment
obs = env.reset()
done = False
rewards = []
while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, done, _ = env.step(action)
    rewards.append(reward)

# Visualization of rewards
plt.plot(rewards)
plt.title("Rewards Over Steps")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.show()
