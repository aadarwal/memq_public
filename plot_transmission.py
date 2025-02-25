import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import c  # speed of light

def freq_to_wavelength(freq):
    """Convert frequency in Hz to wavelength in nm"""
    return (c / freq) * 1e9  # Convert to nm

def process_spectrum_data(data_str):
    """Convert spectrum data string to numpy array"""
    return np.array([float(x) for x in data_str.split(',')])

def plot_transmission():
    # Read the CSV file
    df = pd.read_csv('frequency_power_sweep.csv')
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Process each measurement
    for power in df['Power_dBm'].unique():
        power_data = df[df['Power_dBm'] == power]
        
        for _, row in power_data.iterrows():
            # Get frequency range
            start_freq = row['Spectrum_Start_Freq']
            stop_freq = row['Spectrum_Stop_Freq']
            points = row['Spectrum_Points']
            
            # Create frequency array
            freqs = np.linspace(start_freq, stop_freq, points)
            
            # Convert to wavelengths
            wavelengths = freq_to_wavelength(freqs)
            
            # Process spectrum data
            spectrum = process_spectrum_data(row['Spectrum_Data'])
            
            # Plot
            plt.plot(wavelengths, spectrum, 
                    label=f"Power: {power}dBm, Freq: {row['Frequency_Hz']/1e6:.1f}MHz")
    
    # Customize plot
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmission (dB)')
    plt.title('Transmission vs Wavelength')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('transmission_vs_wavelength.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'transmission_vs_wavelength.png'")
    
    plt.show()

if __name__ == "__main__":
    plot_transmission() 