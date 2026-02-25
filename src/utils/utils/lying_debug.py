import PySpin
import sys

def test_pyspin_camera():
    # 1. Spinnaker 시스템 인스턴스 가져오기
    system = PySpin.System.GetInstance()

    # 2. 연결된 카메라 목록 가져오기
    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()

    print(f"🔍 발견된 카메라 수: {num_cameras}")

    if num_cameras == 0:
        print("❌ 카메라를 찾을 수 없습니다. 연결을 확인해 주세요.")
        cam_list.Clear()
        system.ReleaseInstance()
        return False

    # 3. 첫 번째 카메라 선택
    cam = cam_list.GetByIndex(0)

    try:
        # 4. 카메라 초기화
        cam.Init()
        print("✅ 카메라 초기화 완료")

        # (선택) 카메라 모델명 출력
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        node_device_information = PySpin.CCategoryPtr(nodemap_tldevice.GetNode('DeviceInformation'))
        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                if node_feature.GetName() == "DeviceModelName":
                    print(f"📷 모델명: {node_feature.ToString()}")

        # 5. 이미지 획득 시작
        cam.BeginAcquisition()
        print("▶️ 이미지 획득 시작...")

        # 6. 이미지 버퍼에서 가져오기 (타임아웃 1000ms 설정)
        image_result = cam.GetNextImage(1000)

        # 7. 이미지가 정상적으로 들어왔는지 확인
        if image_result.IsIncomplete():
            print(f"⚠️ 이미지 데이터가 불완전합니다. 에러 코드: {image_result.GetImageStatus()}")
        else:
            width = image_result.GetWidth()
            height = image_result.GetHeight()
            print(f"성공적으로 이미지를 받았습니다! (해상도: {width} x {height})")

            # 8. 이미지 저장 (JPEG 포맷)
            filename = "pyspin_test_image.jpg"
            image_result.Save(filename)
            print(f"💾 이미지가 저장되었습니다: {filename}")

        # 9. 메모리 릭(Leak) 방지를 위해 이미지 메모리 해제
        image_result.Release()

        # 10. 이미지 획득 종료
        cam.EndAcquisition()
        print("⏹️ 이미지 획득 종료")

    except PySpin.SpinnakerException as ex:
        print(f"❌ Spinnaker 에러 발생: {ex}")
    except Exception as e:
        print(f"❌ 알 수 없는 에러 발생: {e}")

    finally:
        # 11. 카메라 초기화 해제 및 리소스 정리 (매우 중요)
        try:
            cam.DeInit()
        except:
            pass
        del cam  # 참조 제거
        cam_list.Clear()
        system.ReleaseInstance()
        print("🔌 카메라 연결 해제 및 시스템 리소스 정리 완료")

if __name__ == "__main__":
    test_pyspin_camera()