# SnakeRL: Reinforcement Learning Snake Game

## Overview
SnakeRL is a Python-based implementation of the classic Snake game, powered by a Deep Q-Learning (DQN) reinforcement learning algorithm. The project demonstrates how a neural network can be trained to play the Snake game by learning optimal actions through interaction with a custom game environment. The agent navigates a grid, collects food, avoids collisions, and dynamically adjusts rewards based on the snake's length to enhance learning efficiency.

This project is ideal for those interested in reinforcement learning, neural networks, or game AI development. It includes a modular environment (`Env.py`), an RL agent (`Agent.py`), and a main script (`main.py`) to train and test the agent.

## Features
- **Custom Snake Environment**: A grid-based environment with configurable dimensions, supporting snake movement, food collection, and collision detection.
- **Deep Q-Learning Agent**: Utilizes a neural network with Keras to learn optimal actions using an Epsilon-Greedy or Boltzmann policy.
- **Dynamic Reward System**: Adjusts rewards for food collection and collisions based on snake length to balance difficulty.
- **Visualization**: Real-time game visualization using Matplotlib and logging of rewards for performance analysis.
- **Configurable Hyperparameters**: Easily tweak learning rate, gamma, epsilon, temperature, and network architecture.
- **Model Persistence**: Save and load trained models for continued training or testing.

## Prerequisites
To run SnakeRL, ensure you have the following installed:
- Python 3.8+
- Required Python packages:
  ```bash
  pip install numpy tensorflow keras matplotlib
  ```

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/SnakeRL.git
   cd SnakeRL
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Create a `requirements.txt` file with:
   ```
   numpy
   tensorflow
   keras
   matplotlib
   ```

3. **Directory Setup**:
   Ensure a directory exists at `C:\Users\F15\Desktop\Python\AI\Reinforcement Learning\Snape Game\` for saving models and logs, or modify the paths in `Agent.py` to suit your environment.

## Usage
1. **Run the Main Script**:
   The `main.py` script initializes the environment, creates the agent, trains it, and tests its performance. Run it with:
   ```bash
   python main.py
   ```

2. **Training**:
   - The agent trains for 888 episodes using the Boltzmann policy (`Policy='B'`).
   - Training progress, including epsilon and temperature values, is printed to the console.
   - Rewards are logged to `Rewards.txt` in the specified directory.
   - Optionally, enable `trShow=True` in `Agent.py` for real-time visualization during training.

3. **Testing**:
   - After training, the agent runs 5 test episodes using a greedy policy (`Policy='G'`).
   - Test results, including total rewards and accuracy, are appended to `Rewards.txt`.
   - Set `teShow=True` in `Agent.py` to visualize test episodes.

4. **Plotting Results**:
   - Use `Agent.PlotEpsilons()` or `Agent.PlotTemperatures()` to visualize the decay of exploration parameters.
   - Use `Agent.PlotActionLog(L)` or `Agent.PlotEpisodeLog(L)` to plot rewards with a Simple Moving Average (SMA) over `L` steps or episodes.

## Project Structure
```
SnakeRL/
├── Agent.py          # RL agent with Deep Q-Learning implementation
├── Env.py           # Snake game environment
├── main.py          # Main script to run training and testing
├── requirements.txt  # Python dependencies
└── README.md        # Project documentation
```

## How It Works
- **Environment (`Env.py`)**: Defines a grid-based Snake game with states (free, snake, food, head, out) and actions (up, right, down, left). Rewards are assigned for food (+14), collisions (-12), and proximity to food (±2.5).
- **Agent (`Agent.py`)**: Implements a Deep Q-Learning agent with a neural network (SELUs activation, configurable dense layers). Supports Epsilon-Greedy and Boltzmann exploration policies, with dynamic alpha and temperature decay.
- **Dynamic Rewards**: Food and collision rewards scale with snake length to reflect increasing difficulty.
- **State Representation**: Uses an embedding technique to encode the game state as a vector of ±1 values, normalized head and food positions, ensuring inputs are in [-1, +1] for stable training.

## Hyperparameters
Key hyperparameters in `main.py`:
- **Grid Size**: 20x20 (`H=20`, `W=20`)
- **Episodes**: 888 (`nEpisode=888`)
- **Max Steps per Episode**: 128 (`mStep=128`)
- **Learning Rate**: 0.001 (`LR=0.001`)
- **Gamma**: 0.95 (`Gamma=0.95`)
- **Memory Size**: 1024 (`sMemory=1024`)
- **Batch Size**: 64 (`sBatch=64`)
- **Network Architecture**: [512, 256] dense layers with SELU activation

Modify these in `main.py` or `Agent.py` to experiment with different configurations.

## Future Improvements
- Add support for convolutional neural networks (CNNs) to process raw game grids.
- Implement additional RL algorithms like Dueling DQN or A2C.
- Enhance visualization with a GUI (e.g., Pygame or Tkinter).
- Optimize hyperparameters using grid search or automated tuning.
- Add support for different grid sizes and game modes.
