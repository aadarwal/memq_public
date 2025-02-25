from pymeasure.instruments.keithley import Keithley2450
import time
import numpy as np
def test_voltage_control():
    keithley = Keithley2450("USB0::0x05E6::0x2450::04462512::INSTR")
    try:
        print("Initializing Keithley 2450...")
        keithley.reset()
        keithley.apply_voltage(compliance_current=0.1)  # 100mA compliance
        keithley.source_voltage_range = 10
        keithley.source_voltage = 0
        print("Enabling source output...")
        keithley.enable_source()
        time.sleep(1)
        test_voltages = [0, 1, 2, 1, 0, -1, -2, -1, 0]  # Safe test sequence
        print("\nStarting voltage test sequence...")
        for target_voltage in test_voltages:
            print(f"\nSetting voltage to {target_voltage}V")
            keithley.source_voltage = target_voltage
            time.sleep(1)  # Wait for voltage to stabilize
            measured_voltage = keithley.voltage
            print(f"Set voltage: {target_voltage}V")
            print(f"Measured voltage: {measured_voltage:.3f}V")
            response = input("Press Enter to continue to next voltage (or 'q' to quit): ")
            if response.lower() == 'q':
                break
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        print("\nTest ending - returning to 0V...")
        keithley.ramp_to_voltage(0, steps=5)  # Safely ramp down
        keithley.shutdown()
        print("Keithley shutdown complete")
if __name__ == "__main__":
    print("=== Keithley 2450 Voltage Control Test ===")
    print("This script will:")
    print("1. Initialize the Keithley")
    print("2. Step through a sequence of test voltages")
    print("3. Show measured voltage at each step")
    print("4. Allow you to verify each step")
    print("\nIMPORTANT:")
    print("- Press Enter to proceed to next voltage")
    print("- Press 'q' and Enter to quit at any time")
    print("- The script will safely return to 0V when done")
    print("\nMake sure:")
    print("- Keithley is powered on")
    print("- USB cable is connected")
    print("- Your device is properly connected")
    input("\nPress Enter to begin the test...")
    test_voltage_control()