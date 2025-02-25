
from pymeasure.instruments.keithley import Keithley2450
from mlpPyAPI.api import connect_to_api
import time
import numpy as np
import os
import logging

def validate_results_path(results_path):
    """Ensure the results directory exists and is valid"""
    try:
        # Remove any quotes and extra spaces
        results_path = results_path.strip().strip('"').strip("'")
        
        # Convert to absolute path
        results_path = os.path.abspath(results_path)
        
        if not os.path.exists(results_path):
            os.makedirs(results_path)
            print(f"Created directory: {results_path}")
        else:
            print(f"Using existing directory: {results_path}")
            
        return results_path
    
    except Exception as e:
        print(f"Error with directory path: {str(e)}")
        print("Please enter a valid directory path (e.g., C:\\Experiments\\Test1)")
        raise

def setup_optical_instruments():
    """Initialize and configure the optical measurement setup"""
    try:
        mlp = connect_to_api()
        laser = mlp.laser
        laser_sweep = mlp.laser_sweep
        
        if not laser.is_connected():
            laser.set_connection_param('gbiAddress', 1)
            laser.connect()
        
        if not laser.is_on():
            laser.turn_on()
            time.sleep(1)
        
        if not laser_sweep.can_start():
            raise RuntimeError("Laser sweep cannot start. Check instrument connections.")
        
        return mlp, laser, laser_sweep
    except Exception as e:
        print(f"Error setting up optical instruments: {str(e)}")
        raise

def perform_measurement(voltage, laser_sweep, results_path, wait_time=1.0):
    """Perform a single measurement at given voltage"""
    if laser_sweep.can_start():
        print(f"Starting Laser Sweep at {voltage:.2f}V...")
        laser_sweep.start()
        laser_sweep.set_save_results_path(results_path)
        device_label = f"Voltage_{voltage:.2f}V_"
        laser_sweep.save_results_to_csv(f"{device_label}.csv", use_results_path=True)
        time.sleep(wait_time)
        print(f"Measurement at {voltage:.2f}V completed.")
        return True
    print(f"Laser sweep not ready at {voltage:.2f}V. Measurement skipped.")
    return False

def main():
    # Configuration
    CONFIG = {
        'voltage_start': -10,
        'voltage_end': 10,
        'voltage_step': 2,
        'compliance_current': 0.1,
        'voltage_range': 10,
        'measurement_delay': 1.0
    }
    
    keithley = None
    
    try:
        # Initialize Keithley
        print("Initializing Keithley 2450...")
        keithley = Keithley2450("USB0::0x05E6::0x2450::04462512::INSTR")
        keithley.reset()
        time.sleep(1)
        keithley.apply_voltage(compliance_current=CONFIG['compliance_current'])
        keithley.source_voltage_range = CONFIG['voltage_range']
        keithley.source_voltage = 0
        
        # Setup optical instruments
        print("\nSetting up optical instruments...")
        mlp, laser, laser_sweep = setup_optical_instruments()
        
        # Get and validate results path
        while True:
            try:
                print("\nEnter the path where results should be saved.")
                print("Example: C:\\Experiments\\Test1")
                results_path = input('Path: ')
                results_path = validate_results_path(results_path)
                break
            except:
                print("Invalid path. Please try again.")
                continue
        
        print("\nEnabling Keithley source output...")
        keithley.enable_source()
        time.sleep(1)
        
        # Generate voltage sequence
        voltages = np.arange(CONFIG['voltage_start'], 
                           CONFIG['voltage_end'] + CONFIG['voltage_step'], 
                           CONFIG['voltage_step'])
        
        print("\nStarting voltage sweep with measurements...")
        print(f"Will test voltages: {voltages}")
        input("Press Enter to begin sweep (or Ctrl+C to abort)...")
        
        for voltage in voltages:
            print(f"\nSetting voltage to {voltage:.2f}V")
            keithley.source_voltage = voltage
            time.sleep(CONFIG['measurement_delay'])
            
            measured_voltage = keithley.voltage
            print(f"Set voltage: {voltage:.2f}V")
            print(f"Measured voltage: {measured_voltage:.3f}V")
            
            success = perform_measurement(voltage, laser_sweep, results_path)
            if not success:
                print(f"Warning: Measurement failed at {voltage:.2f}V")
                response = input("Press Enter to continue, or 'q' to quit: ")
                if response.lower() == 'q':
                    break
    
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise
    
    finally:
        print("\nExperiment ending - performing safe shutdown...")
        if keithley is not None:
            try:
                print("Ramping Keithley voltage to 0...")
                keithley.ramp_to_voltage(0, steps=5)
                keithley.shutdown()
                print("Keithley shutdown complete")
            except Exception as e:
                print(f"Error during Keithley shutdown: {str(e)}")
        
        print("Measurement sequence completed.")

if __name__ == "__main__":
    print("=== Keithley 2450 and Optical Measurement Script ===")
    print("\nThis script will:")
    print("1. Initialize the Keithley and optical instruments")
    print("2. Perform voltage sweep with optical measurements")
    print("3. Save results for each measurement")
    print("\nMake sure:")
    print("- Keithley is powered on and USB connected")
    print("- Optical measurement setup is ready")
    print("- Your device is properly connected")
    
    input("\nPress Enter to begin...")
    main()
