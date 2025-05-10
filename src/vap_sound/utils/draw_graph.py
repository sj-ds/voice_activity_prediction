import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

LOG_FILE = "/Users/shanujha/Desktop/voice_activity_prediction/pred.txt"
WINDOW_SIZE = 20  # Show only the latest 20 seconds

def read_predictions():
    """Reads the latest predictions from the log file."""
    timestamps, predictions = [], []
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            # print(lines)
            for line in lines:
                # print(line)
                parts = line.strip().split("Prediction: ")
                # print(parts, len(parts))
                if len(parts) == 2:
                    # print(parts)
                    try:
                        timestamp = float(parts[0])  # Extract timestamp
                        # print(timestamp)
                        prediction = float(parts[1])  # Extract prediction value
                        # print(prediction)
                        timestamps.append(timestamp)
                        predictions.append(prediction)
                    except ValueError:
                        continue  # Skip any malformed lines
    except FileNotFoundError:
        pass  # If the file doesn't exist yet, just return empty lists
    return timestamps, predictions


a, b = read_predictions()
print(a, b)

# Setup Matplotlib figure
fig, ax = plt.subplots()
x_data, y_data = [], []
line, = ax.plot([], [], 'b-', label="Prediction")

def update(frame):
    """Fetches new data and updates the graph with a sliding window."""
    global x_data, y_data
    timestamps, predictions = read_predictions()

    if timestamps and predictions:
        # Get the latest timestamp
        current_time = timestamps[-1]

        # Filter data to keep only the last WINDOW_SIZE seconds
        x_data = [t for t in timestamps if t >= current_time - WINDOW_SIZE]
        y_data = predictions[-len(x_data):]  # Keep only corresponding y-values

        line.set_data(x_data, y_data)

        # Set x-axis limits based on the sliding window
        ax.set_xlim(current_time - WINDOW_SIZE, current_time)
        ax.set_ylim(min(y_data) - 0.05, max(y_data) + 0.05)  # Adjust y-axis dynamically

    return line,

# Set up animation
ani = animation.FuncAnimation(fig, update, interval=500)

# Graph settings
ax.set_xlabel("Time (UNIX Timestamp)")
ax.set_ylabel("Prediction")
ax.set_title("Live Prediction Graph (Sliding Window: 20s)")
ax.legend()
plt.show()
