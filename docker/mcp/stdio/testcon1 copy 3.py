#!/usr/bin/env python3
"""
Network Connectivity Checker for SQL Server vs837
Checks connectivity to vs837 using multiple methods and addresses
"""

import socket
import subprocess
import platform
import time
import sys
from typing import List, Tuple, Dict

class NetworkConnectivityChecker:
    def __init__(self):
        self.targets = [
            ("vs837", 1433),
            ("vs837.vll.se", 1433),
            ("10.19.50.53", 1433)
        ]
        self.ping_targets = [
            "vs837",
            "vs837.vll.se", 
            "10.19.50.53"
        ]
        
    def check_port_connectivity(self, host: str, port: int, timeout: int = 5) -> bool:
        """Check if a specific port is reachable on a host"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as e:
            print(f"    Error checking {host}:{port} - {e}")
            return False
    
    def ping_host(self, host: str, timeout: int = 3) -> bool:
        """Ping a host to check basic connectivity"""
        try:
            # Determine ping command based on OS
            if platform.system().lower() == "windows":
                cmd = ["ping", "-n", "1", "-w", str(timeout * 1000), host]
            else:
                cmd = ["ping", "-c", "1", "-W", str(timeout), host]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 2)
            return result.returncode == 0
        except Exception as e:
            print(f"    Error pinging {host} - {e}")
            return False
    
    def resolve_hostname(self, hostname: str) -> List[str]:
        """Resolve hostname to IP addresses"""
        try:
            result = socket.gethostbyname_ex(hostname)
            return result[2]  # List of IP addresses
        except Exception as e:
            print(f"    Error resolving {hostname} - {e}")
            return []
    
    def check_sql_server_specific(self, host: str, port: int = 1433) -> Dict[str, bool]:
        """Check SQL Server specific connectivity"""
        results = {}
        
        # Check default SQL Server port
        results['sql_port_1433'] = self.check_port_connectivity(host, 1433, 3)
        
        # Check dynamic port (if 1433 fails, try SQL Browser port)
        if not results['sql_port_1433']:
            results['sql_browser_1434'] = self.check_port_connectivity(host, 1434, 3)
        
        return results
    
    def run_comprehensive_check(self) -> Dict[str, Dict]:
        """Run comprehensive connectivity checks"""
        print("=" * 60)
        print("SQL Server vs837 Network Connectivity Checker")
        print("=" * 60)
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
        print()
        
        results = {}
        
        # 1. DNS Resolution Check
        print("1. DNS Resolution Check:")
        print("-" * 30)
        for target in ["vs837", "vs837.vll.se"]:
            ips = self.resolve_hostname(target)
            results[f"{target}_dns"] = len(ips) > 0
            if ips:
                print(f"✅ {target} resolves to: {', '.join(ips)}")
            else:
                print(f"❌ {target} - DNS resolution failed")
        print()
        
        # 2. Ping Connectivity Check
        print("2. Ping Connectivity Check:")
        print("-" * 30)
        for host in self.ping_targets:
            is_reachable = self.ping_host(host)
            results[f"{host}_ping"] = is_reachable
            if is_reachable:
                print(f"✅ {host} - Ping successful")
            else:
                print(f"❌ {host} - Ping failed")
        print()
        
        # 3. SQL Server Port Connectivity
        print("3. SQL Server Port Connectivity Check:")
        print("-" * 40)
        for host, port in self.targets:
            sql_results = self.check_sql_server_specific(host)
            results[f"{host}_sql"] = sql_results
            
            print(f"Host: {host}")
            for check, success in sql_results.items():
                port_num = check.split('_')[-1]
                if success:
                    print(f"  ✅ Port {port_num} - Connection successful")
                else:
                    print(f"  ❌ Port {port_num} - Connection failed")
            print()
        
        # 4. Network Interface Check
        print("4. Network Interface Information:")
        print("-" * 35)
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            print(f"Local hostname: {hostname}")
            print(f"Local IP: {local_ip}")
            
            # Check if we're in the same subnet as target
            if local_ip.startswith("10.19."):
                print("✅ Appears to be on internal network (10.19.x.x subnet)")
                results['internal_network'] = True
            else:
                print("⚠️  Not on expected internal network subnet")
                results['internal_network'] = False
        except Exception as e:
            print(f"❌ Error getting network info: {e}")
            results['internal_network'] = False
        print()
        
        return results
    
    def generate_summary(self, results: Dict) -> str:
        """Generate a summary of connectivity status"""
        print("5. Connectivity Summary:")
        print("-" * 25)
        
        # Check if any method succeeded
        sql_success = False
        ping_success = False
        dns_success = False
        
        for key, value in results.items():
            if 'sql' in key and isinstance(value, dict):
                if any(value.values()):
                    sql_success = True
            elif 'ping' in key and value:
                ping_success = True
            elif 'dns' in key and value:
                dns_success = True
        
        summary = []
        
        if sql_success:
            summary.append("✅ SQL Server connectivity: SUCCESS")
            status = "CONNECTED"
        else:
            summary.append("❌ SQL Server connectivity: FAILED")
            status = "NOT CONNECTED"
        
        if ping_success:
            summary.append("✅ Network ping: SUCCESS")
        else:
            summary.append("❌ Network ping: FAILED")
        
        if dns_success:
            summary.append("✅ DNS resolution: SUCCESS")
        else:
            summary.append("❌ DNS resolution: FAILED")
        
        if results.get('internal_network', False):
            summary.append("✅ Internal network: DETECTED")
        else:
            summary.append("⚠️  Internal network: NOT DETECTED")
        
        for line in summary:
            print(line)
        
        print()
        print("=" * 60)
        print(f"OVERALL STATUS: {status}")
        print("=" * 60)
        
        # Recommendations
        if not sql_success:
            print("\nTroubleshooting Recommendations:")
            print("- Check if SQL Server service is running on vs837")
            print("- Verify firewall settings allow port 1433")
            print("- Confirm you're connected to the internal network")
            print("- Try connecting with SQL Server Management Studio")
            if not ping_success:
                print("- Check network connectivity (ping failed)")
            if not dns_success:
                print("- Check DNS settings or use IP address directly")
        
        return status

def main():
    """Main function to run the connectivity checker"""
    try:
        checker = NetworkConnectivityChecker()
        results = checker.run_comprehensive_check()
        status = checker.generate_summary(results)
        
        # Exit with appropriate code
        sys.exit(0 if status == "CONNECTED" else 1)
        
    except KeyboardInterrupt:
        print("\n\nCheck interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()