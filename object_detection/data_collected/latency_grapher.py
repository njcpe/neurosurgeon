import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

frame = pd.read_csv('ns_latency.txt')

frame['mobile_processing'] = frame['mobile_processing'].div(1000)
frame['server_processing'] = frame['server_processing'].div(1000)
frame['transmission'] = frame['transmission'].div(1000)
frame['end_to_end'] = frame['end_to_end'].div(1000)

final_data = frame.filter(["layer_name","mobile_processing", "transmission", "server_processing"]).set_index(["layer_name"])

plot = final_data.plot.bar(stacked=True, title="NS Implemenation Delay Breakdown")

plt.show()

