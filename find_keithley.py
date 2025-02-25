import pyvisa
def list_instruments():
    try:
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        print("\nFound devices:")
        for idx, resource in enumerate(resources):
            print(f"{idx+1}. {resource}")
            try:
                inst = rm.open_resource(resource)
                try:
                    id_string = inst.query('*IDN?')
                    print(f"   Device ID: {id_string.strip()}")
                except:
                    print("   Could not get device ID")
                inst.close()
            except:
                print("   Could not open device")
        return resources
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure NI-VISA is installed")
        print("2. Verify the device is powered on")
        print("3. Try disconnecting and reconnecting the USB cable")
        return []
if __name__ == "__main__":
    print("Searching for connected instruments...")
    resources = list_instruments()
    if not resources:
        print("\nNo instruments found.")
    else:
        print("\nTo use these addresses in your code, use the exact string shown above.")






