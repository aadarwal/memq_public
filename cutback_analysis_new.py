import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec

# Directory setup - modify this path to your local directory containing the CSV files
DATA_DIRECTORY = '/Users/debroccolie/Desktop/memq/dolores_19_0p5' 
directory_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith('.csv')]

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
    # Initial guess for curve_fit
    p0 = [(data_points[2]-data_points[0])/(x[2]-x[0]), data_points[0]]
    
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

# Create plots
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, :-1])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[1, 2])

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

# Plot 2: Uncertainties
ax2.plot(WV, uncertainties)
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Uncertainty (dB/cm)')
ax2.set_title('Fit Uncertainties')
ax2.grid(True)

# Plot 3: Raw signals
colors = plt.cm.rainbow(np.linspace(0, 1, len(signals)))
for i, (signal, color) in enumerate(zip(signals, colors)):
    ax3.plot(WV, signal, color=color, label=devices[i])
ax3.set_xlabel('Wavelength (nm)')
ax3.set_ylabel('Signal (dB)')
ax3.set_title('Raw Signals')
ax3.legend(fontsize=8)
ax3.grid(True)

# Plot 4: Minima analysis
minima = [np.min(signal) for signal in signals]
ax4.scatter(x, minima)
ax4.set_xlabel('Length (cm)')
ax4.set_ylabel('Minimum Signal (dB)')
ax4.set_title('Signal Minima vs Length')
for i, (xi, yi) in enumerate(zip(x, minima)):
    ax4.annotate(devices[i], (xi, yi), xytext=(5, 5), textcoords='offset points')
ax4.grid(True)

# Plot 5: Average analysis
averages = [np.mean(signal) for signal in signals]
ax5.scatter(x, averages)
ax5.set_xlabel('Length (cm)')
ax5.set_ylabel('Average Signal (dB)')
ax5.set_title('Average Signal vs Length')
for i, (xi, yi) in enumerate(zip(x, averages)):
    ax5.annotate(devices[i], (xi, yi), xytext=(5, 5), textcoords='offset points')
ax5.grid(True)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\nAnalysis Summary:")
print(f"Loss difference: {slope_diff:.2f} dB/cm")
print(f"Default loss: {np.nanmin(slopes):.2f} dB/cm")
print(f"Mean uncertainty: {mean_uncertainty:.2f} dB/cm")
print(f"Peak uncertainty: {peak_uncertainty:.2f} dB/cm")
