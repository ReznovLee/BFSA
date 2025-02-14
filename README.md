# Radar Network Scheduling and Target Tracking

## Introduction

This project focuses on radar network scheduling and target tracking. It includes various algorithms and models to optimize the allocation of radar resources for tracking multiple targets. The main goal is to improve the efficiency and accuracy of radar systems in tracking targets by using advanced scheduling algorithms.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/ReznovLee/BFSA.git
   cd BFSA
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the main script and generate results, follow these steps:

1. Ensure you have the necessary configuration files and data generators in the `data` directory.

2. Run the main script:
   ```bash
   python main.py
   ```

3. The results will be generated and saved in the `results` directory.

## Project Structure

The project is organized into the following directories:

- `core`: Contains the main algorithms and models used in the project.
  - `algorithms`: Implementation of scheduling algorithms.
  - `models`: Definition of radar network and target models.
  - `utils`: Utility functions and classes for constraints, Kalman filter, and metrics.

- `data`: Contains configuration files and data generators for creating scenarios.
  - `configs`: Configuration files for different scenarios.
  - `generators`: Scripts for generating radar and target data.

- `visualization`: Contains scripts for plotting and visualizing results.

- `results`: Directory where the generated results are saved.

## Contributing

We welcome contributions to the project. To contribute, follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch with a descriptive name:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit them with clear and concise commit messages.
4. Push your changes to your forked repository:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Create a pull request on the main repository, describing your changes and the motivation behind them.

Please ensure your code follows the project's coding standards and includes appropriate tests.

