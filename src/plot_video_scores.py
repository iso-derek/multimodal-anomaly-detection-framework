import pandas as pd
import matplotlib.pyplot as plt

csv_path = r"outputs\video_ucsd\UCSDped1_Test_Test001_scores.csv"
df = pd.read_csv(csv_path)

plt.figure(figsize=(10,4))
plt.plot(df["frame_index"], df["score"])
plt.title("UCSDped1 Test001 - Frame Difference Anomaly Score")
plt.xlabel("Frame Index")
plt.ylabel("Normalized Score (0-1)")
plt.grid(True)
plt.show()
