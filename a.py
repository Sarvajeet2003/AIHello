import numpy as np
import pandas as pd
import gym
from gym import spaces

# Load the historical data
soapnut_data_path = 'soapnutshistory.csv'  # Replace with your file path
woolball_data_path = 'woolballhistory.csv'  # Replace with your file path

# Data Preprocessing
def preprocess_data(data):
    # Convert 'Report Date' to datetime format
    data['Report Date'] = pd.to_datetime(data['Report Date'])
    
    # Drop rows with NaN values in critical columns
    data = data.dropna(subset=['Product Price', 'Total Sales', 'Predicted Sales'])
    
    # Normalize the columns for RL model (Min-Max Scaling)
    numeric_columns = ['Product Price', 'Organic Conversion Percentage',
                       'Ad Conversion Percentage', 'Total Profit',
                       'Total Sales', 'Predicted Sales']
    
    for col in numeric_columns:
        min_val = data[col].min()
        max_val = data[col].max()
        if max_val > min_val:  # Avoid division by zero
            data.loc[:, col] = (data[col] - min_val) / (max_val - min_val)
    return data

# Load and preprocess data
soapnut_data = pd.read_csv(soapnut_data_path)
woolball_data = pd.read_csv(woolball_data_path)
soapnut_data_cleaned = preprocess_data(soapnut_data)
woolball_data_cleaned = preprocess_data(woolball_data)

# Define the RL environment
class PriceOptimizationEnv(gym.Env):
    def __init__(self, data):
        super(PriceOptimizationEnv, self).__init__()
        self.data = data.reset_index(drop=True)  # Reset index for sequential access
        self.current_step = 0

        # Define action and observation space
        # Actions: Increase or decrease price by a percentage [-0.1, 0.1]
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(1,), dtype=np.float32)

        # Observations: Product price, Organic Conversion, Ad Conversion, Total Sales, Predicted Sales
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(5,),
            dtype=np.float32,
        )

    def reset(self):
        """Reset the environment to the initial state."""
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        """Take an action and return the new state, reward, and done flag."""
        # Adjust price based on action
        action = np.clip(action, self.action_space.low, self.action_space.high)[0]
        new_price = self.data.loc[self.current_step, "Product Price"] + action

        # Clamp the price to the range [0, 1]
        new_price = np.clip(new_price, 0, 1)

        # Simulate the impact of price change on sales and conversions
        organic_conversion = self.data.loc[self.current_step, "Organic Conversion Percentage"]
        ad_conversion = self.data.loc[self.current_step, "Ad Conversion Percentage"]
        predicted_sales = self.data.loc[self.current_step, "Predicted Sales"]

        # Reward function
        sales = predicted_sales * (1 + (new_price - self.data.loc[self.current_step, "Product Price"]))
        reward = sales + organic_conversion + ad_conversion

        # Update state
        self.data.loc[self.current_step, "Product Price"] = new_price

        # Increment step and check if done
        self.current_step += 1
        done = self.current_step >= len(self.data)
        return self._get_observation() if not done else None, reward, done, {}

    def _get_observation(self):
        """Return the current state as an observation."""
        row = self.data.loc[self.current_step]
        return np.array([
            row["Product Price"],
            row["Organic Conversion Percentage"],
            row["Ad Conversion Percentage"],
            row["Total Sales"],
            row["Predicted Sales"]
        ])

# Initialize the environment
env = PriceOptimizationEnv(soapnut_data_cleaned)

# Test the environment
obs = env.reset()
done = False
while not done:
    # Sample random action
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    reward = 0 if np.isnan(reward) else reward
    obs = np.nan_to_num(obs)
    if done:
        print("Episode finished.")
        break
    print(f"Action: {action}, Reward: {reward}, Observation: {obs}")
