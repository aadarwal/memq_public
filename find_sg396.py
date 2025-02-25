import pyvisa
import platform
import subprocess
import socket

def get_local_ip():
    """Get the local IP address of the computer"""
    try:
        # Create a dummy connection to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return None

def check_network_connection(ip):
    """Test basic network connectivity"""
    try:
        if platform.system().lower() == "windows":
            result = subprocess.run(["ping", "-n", "1", "-w", "1000", ip],
                                  capture_output=True, text=True)
        else:
            result = subprocess.run(["ping", "-c", "1", "-W", "1", ip],
                                  capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def list_sg396():
    """
    Find and test connection to SG396 via LAN
    """
    try:
        rm = pyvisa.ResourceManager()
        sg_address = 'TCPIP0::192.168.1.205::inst0::INSTR'
        
        print("\nChecking network configuration:")
        local_ip = get_local_ip()
        print(f"Local IP: {local_ip}")
        print(f"Target SG396 IP: 192.168.1.205")
        
        # Check basic network connectivity first
        print("\nTesting network connectivity...")
        if not check_network_connection("192.168.1.205"):
            print("⚠️  Network ping failed! Checking if VISA can still connect...")
        
        print(f"\nTrying to connect to SG396 at {sg_address}")
        try:
            inst = rm.open_resource(sg_address)
            try:
                id_string = inst.query('*IDN?')
                print(f"✓ Successfully connected!")
                print(f"Device ID: {id_string.strip()}")
                
                # Test basic communication
                inst.write('*CLS')  # Clear status
                inst.query('*OPC?')  # Check operation complete
                print("✓ Basic communication test passed!")
                
            except Exception as query_error:
                print(f"✗ Could not query device: {query_error}")
            finally:
                inst.close()
                
        except Exception as conn_error:
            print(f"✗ Connection failed: {conn_error}")
            print("\nTroubleshooting steps:")
            print("1. Verify the SG396 is powered on")
            print("2. Check the network cable connections")
            print("3. Verify the IP address (192.168.1.205) on the device's front panel")
            print("4. Ensure your computer and the SG396 are on the same subnet")
            print("   - Your IP should be like 192.168.1.xxx")
            print("   - Subnet mask should be 255.255.255.0")
            print("5. Try accessing the device's web interface (if available)")
            print("6. Check if any firewall is blocking the connection")
            
    except Exception as e:
        print(f"\nError during device search: {str(e)}")
        print("\nBasic troubleshooting:")
        print("1. Make sure NI-VISA is installed")
        print("2. Check network adapter settings")
        print("3. Try rebooting both the SG396 and computer")

if __name__ == "__main__":
    print("Searching for SG396 Signal Generator...")
    list_sg396()