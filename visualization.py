from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def save_to_csv(state_changes: List[Dict], export_path: str) -> None:
    """
    Save the car tracking and counting data to a CSV file.

    Parameters:
        state_changes (List[Dict]): The car state changes to save.
        export_path (str): The path to save the CSV file to.
    """
    df = pd.DataFrame(state_changes)
    df.to_csv(export_path, index=False)


def visualize_data(csv_path: str, save: bool = False) -> None:
    """
    Generate visualizations for the car tracking and counting data.

    Parameters:
        csv_path (str): The path to the CSV file containing the car tracking and counting data.
        save (bool): Whether to save the plots. Default is False.
        save_path (str): The path to save the plots. Required if save is True.
    """
    # Load the data
    df = pd.read_csv(csv_path)

    # Convert 'timestamp' to datetime format for easier manipulation
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Visualizing the number of cars passing in each direction over time
    plt.figure(figsize=(10, 6))

    # Ensure the plot displays time in seconds
    locator = mdates.SecondLocator()
    formatter = mdates.DateFormatter('%H:%M:%S')

    for direction in df['state'].unique():
        # Filter data for each direction
        direction_data = df[df['state'] == direction]

        # Plot each timestamp directly without resampling
        plt.scatter(direction_data['timestamp'], direction_data['state'].apply(lambda x: direction), label=direction, alpha=0.6)

    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.title('Number of Cars Passing in Each Direction Over Time')
    plt.xlabel('Time')
    plt.ylabel('Direction')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        plt.savefig('./number_of_cars.png')
    plt.show()

    # Visualizing the total count of cars in each direction
    plt.figure(figsize=(8, 5))
    df['state'].value_counts().plot(kind='bar')
    plt.title('Total Count of Cars in Each Direction')
    plt.xlabel('Direction')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    if save:
        plt.savefig('./total_count_of_cars.png')
    plt.show()
