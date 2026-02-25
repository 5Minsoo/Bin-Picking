import os
import platform
import subprocess
import socket
import time

try:
    import serial
except ImportError:
    print("[오류] pyserial 모듈이 없습니다. 터미널에서 'pip install pyserial'을 실행하세요.")
    exit(1)

# ==========================================
# ⚙️ 설정 (본인의 환경에 맞게 수정하세요)
# ==========================================
# 로봇 (LAN) 설정
ROBOT_IP = "192.168.137.100"  # 로봇의 IP 주소
ROBOT_PORT = 12345          # 로봇 통신 포트 (UR은 보통 30001, 30002 등)

# 그리퍼 (USB) 설정
# (USB 소켓 통신이라고 하셨지만, 보통 /dev/ttyUSB0 형태의 시리얼 통신을 의미하는 경우가 많습니다)
GRIPPER_USB_PORT = "/dev/ttyUSB0" 
GRIPPER_BAUDRATE = 115200
# ==========================================

def check_robot_lan(ip, port):
    print(f"\n[1] 로봇 LAN 통신 점검 (IP: {ip}, Port: {port})")
    print("-" * 50)
    
    # 1. Ping 테스트
    param = '-n' if platform.system().lower()=='windows' else '-c'
    command = ['ping', param, '1', ip]
    
    if subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        print("✅ [Ping] 성공: 로봇과 물리적/네트워크로 연결되어 있습니다.")
    else:
        print("❌ [Ping] 실패: 로봇 IP에 도달할 수 없습니다.")
        print("   -> 해결책: 랜선 연결을 확인하고, PC와 로봇의 IP 대역이 같은지 확인하세요.")
        return # 핑이 안 되면 소켓 테스트는 의미가 없으므로 종료

    # 2. Socket(포트) 개방 테스트
    try:
        sock = socket.create_connection((ip, port), timeout=3)
        print(f"✅ [Socket] 성공: 로봇의 {port}번 포트가 열려있고 통신 준비가 되었습니다.")
        sock.close()
    except socket.timeout:
        print(f"❌ [Socket] 시간 초과: 로봇은 켜져 있으나, {port}번 포트로 응답하지 않습니다.")
    except ConnectionRefusedError:
        print(f"❌ [Socket] 연결 거부: 핑은 가지만 {port}번 포트가 닫혀있습니다. 로봇 컨트롤러에서 서버를 켰는지 확인하세요.")
    except Exception as e:
        print(f"❌ [Socket] 기타 오류: {e}")


def check_gripper_usb(usb_port, baudrate):
    print(f"\n[2] 그리퍼 USB 통신 점검 (Port: {usb_port})")
    print("-" * 50)

    # 1. USB 장치 존재 여부 확인 (리눅스 기준)
    if not os.path.exists(usb_port):
        print(f"❌ [USB] 인식 실패: {usb_port} 장치를 찾을 수 없습니다.")
        print("   -> 해결책: USB를 뺐다 다시 꽂아보거나, 'ls /dev/ttyUSB*' 로 실제 포트 이름을 확인하세요.")
        return

    print(f"✅ [USB] 인식 성공: {usb_port} 장치가 시스템에 존재합니다.")

    # 2. 권한 및 열기 테스트
    try:
        ser = serial.Serial(usb_port, baudrate, timeout=1)
        print(f"✅ [Serial] 포트 열기 성공: 권한이 정상이며 통신이 가능합니다.")
        ser.close()
    except serial.SerialException as e:
        if "Permission denied" in str(e):
            print("❌ [Serial] 권한 부족: 포트는 있지만 접근 권한이 없습니다.")
            print(f"   -> 터미널에 입력하여 해결: sudo chmod 666 {usb_port}")
        elif "Device or resource busy" in str(e):
            print("❌ [Serial] 사용 중: 다른 프로그램(또는 죽어있는 ROS 노드)이 포트를 점유하고 있습니다.")
        else:
            print(f"❌ [Serial] 기타 오류: {e}")

if __name__ == "__main__":
    print("=== 하드웨어 및 통신 상태 자가 진단 시작 ===")
    check_robot_lan(ROBOT_IP, ROBOT_PORT)
    check_gripper_usb(GRIPPER_USB_PORT, GRIPPER_BAUDRATE)
    print("\n=== 진단 종료 ===")