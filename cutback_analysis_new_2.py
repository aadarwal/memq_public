import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec

# Directory setup - modify this path to your local directory containing the CSV files
DATA_DIRECTORY = '/Users/debroccolie/Desktop/memq/dolores_19_0p5'  # Update this to your CSV files location
directory_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith('.csv')][:2]  # Take only first two CSV files

# Lists to store the data
signal_data = []
wavelength_data = []
lengths = []

# Load and process CSV files
for file_name in directory_files:
    csv_file_path = os.path.join(DATA_DIRECTORY, file_name)
    data = []
    # Length (in micrometers) extracted from analyzed file
    length_string = file_name.replace('.csv', '').replace('L', '')
    lengths.append(int(length_string))
    
    print(f"Processing file: {file_name}")
    
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        # Convert all rows to appropriate data types
        for row in csv_reader:
            converted_row = []
            for item in row:
                try:
                    converted_item = float(item)
                except ValueError:
                    converted_item = item
                converted_row.append(converted_item)
            data.append(converted_row)
    
    # Storing the wavelength and signal data
    wv_column = data[16][1:]  # Skip first column
    signal_column = data[17][1:]
    
    # Downsample data (every 10th point)
    wv_column = wv_column[::10]
    signal_column = signal_column[::10]
    
    signal_data.append(signal_column)
    wavelength_data.append(wv_column)

# Converting micrometers to cm
lengths = [length/10**4 for length in lengths]
print(f"Processed lengths (cm): {lengths}")

# Define linear fit function
def line(x, a, b):
    return a*x + b

# Initialize lists for analysis
WV = []
slopes = []
y1 = []
y2 = []
uncertainties = []

# Process all signals
signals = []
for i in range(len(signal_data)):
    signals.append(signal_data[i])

# Get wavelength data
WV = wavelength_data[0]

# Calculate slopes and uncertainties
x = lengths
devices = [f"{value}cm" for value in x]

for signal_points in zip(*signals):
    data_points = list(signal_points)
    # Initial guess for curve_fit - modified for two points
    p0 = [(data_points[1]-data_points[0])/(x[1]-x[0]), data_points[0]]
    
    try:
        params, covar = curve_fit(line, x, data_points, p0=p0)
        slope = params[0]
        slopes.append(slope)
        perr = np.sqrt(np.diag(covar))
        uncertainty = perr[0]
        uncertainties.append(uncertainty)
    except:
        print(f"Fitting failed for wavelength point")
        slopes.append(np.nan)
        uncertainties.append(np.nan)

# Calculate uncertainty bounds
for uncert, value in zip(uncertainties, slopes):
    y2.append(value + uncert)
    y1.append(value - uncert)

# Calculate statistics
peak_loss = np.nanmax(slopes)
position = np.nanargmax(slopes)
peak_uncertainties = uncertainties[max(0, position-100):min(len(uncertainties), position+100)]
mean_uncertainty = np.nanmean(uncertainties)
peak_uncertainty = np.nanmean(peak_uncertainties)
slope_diff = np.nanmax(slopes) - np.nanmin(slopes)

# Create plots - modified layout for clearer visualization with 2 devices
fig = plt.figure(figsize=(15, 8))
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])  # Loss/cm over wavelength
ax2 = fig.add_subplot(gs[0, 1])  # Raw signals
ax3 = fig.add_subplot(gs[1, :])  # Combined minima and average

# Plot 1: Loss/cm over wavelength
ax1.plot(WV, slopes, label='Loss/cm')
ax1.fill_between(WV, y2, y1, alpha=0.3, label='Uncertainty')
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Loss (dB/cm)')
ax1.set_title('Loss/cm over wavelength range')
ax1.grid(True)
ax1.legend()

# Add statistics annotation
stats_text = (f"Loss difference: {slope_diff:.2f} dB/cm\n"
             f"Default loss: {np.nanmin(slopes):.2f} dB/cm\n"
             f"Mean uncertainty: {mean_uncertainty:.2f} dB/cm\n"
             f"Peak uncertainty: {peak_uncertainty:.2f} dB/cm")
ax1.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=8, va='top', bbox=dict(boxstyle='round', fc='w', alpha=0.8))

# Plot 2: Raw signals
colors = plt.cm.rainbow(np.linspace(0, 1, len(signals)))
for i, (signal, color) in enumerate(zip(signals, colors)):
    ax2.plot(WV, signal, color=color, label=devices[i])
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Signal (dB)')
ax2.set_title('Raw Signals')
ax2.legend(fontsize=8)
ax2.grid(True)

# Plot 3: Combined minima and average analysis
minima = [np.min(signal) for signal in signals]
averages = [np.mean(signal) for signal in signals]

ax3.scatter(x, minima, label='Minima', color='blue')
ax3.scatter(x, averages, label='Averages', color='red')
ax3.set_xlabel('Length (cm)')
ax3.set_ylabel('Signal (dB)')
ax3.set_title('Signal Minima and Averages vs Length')
for i, (xi, yi_min, yi_avg) in enumerate(zip(x, minima, averages)):
    ax3.annotate(devices[i], (xi, yi_min), xytext=(5, 5), textcoords='offset points')
    ax3.annotate(devices[i], (xi, yi_avg), xytext=(5, 5), textcoords='offset points')
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()

# Print summary statistics
print("\nAnalysis Summary:")
print(f"Loss difference: {slope_diff:.2f} dB/cm")
print(f"Default loss: {np.nanmin(slopes):.2f} dB/cm")
print(f"Mean uncertainty: {mean_uncertainty:.2f} dB/cm")
print(f"Peak uncertainty: {peak_uncertainty:.2f} dB/cm")
