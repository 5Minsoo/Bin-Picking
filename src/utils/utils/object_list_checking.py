#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState


class JointStateEmptyDetector(Node):
    def __init__(self):
        super().__init__("joint_state_empty_detector")

        self.topic = "/joint_states"
        self.sub = self.create_subscription(
            JointState,
            self.topic,
            self.cb,
            qos_profile_sensor_data
        )

        # 통계
        self.total = 0
        self.empty = 0
        self.mismatch = 0
        self.ok = 0

        self.last_empty_log_sec = 0.0
        self.last_ok_log_count = 0

        # 주기적으로 퍼블리셔/통계 출력
        self.timer = self.create_timer(2.0, self.print_status)

        self.get_logger().info(f"Listening: {self.topic}")

    def cb(self, msg: JointState):
        self.total += 1

        # 1) “완전 빈” 판단: name이 없거나, name은 있는데 관련 배열이 전부 비어있는 경우
        is_empty = (len(msg.name) == 0) or (
            len(msg.name) > 0
            and len(msg.position) == 0
            and len(msg.velocity) == 0
            and len(msg.effort) == 0
        )

        if is_empty:
            self.empty += 1

            # 로그 스팸 방지: 1초에 한 번만 경고
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self.last_empty_log_sec > 1.0:
                self.last_empty_log_sec = now
                self.get_logger().error(
                    f"[EMPTY JointState] name={len(msg.name)} pos={len(msg.position)} "
                    f"vel={len(msg.velocity)} eff={len(msg.effort)}"
                )
            return

        # 2) 길이 불일치(이것도 문제)
        # JointState 규약상 position/velocity/effort는 name과 같은 길이거나 0이어야 정상입니다.
        n = len(msg.name)
        bad = False

        def bad_len(arr_len: int) -> bool:
            return (arr_len != 0) and (arr_len != n)

        if bad_len(len(msg.position)) or bad_len(len(msg.velocity)) or bad_len(len(msg.effort)):
            bad = True

        if bad:
            self.mismatch += 1
            self.get_logger().warn(
                f"[MISMATCH JointState] name={n} pos={len(msg.position)} "
                f"vel={len(msg.velocity)} eff={len(msg.effort)} | first_name={msg.name[0] if n>0 else 'N/A'}"
            )
            return

        # 3) 정상 메시지
        self.ok += 1
        # 가끔만 찍기(너무 시끄러우면 주석 처리)
        if self.ok - self.last_ok_log_count >= 200:
            self.last_ok_log_count = self.ok
            self.get_logger().info(
                f"[OK] name={n} pos={len(msg.position)} | first={msg.name[0]}"
            )

    def print_status(self):
        # 퍼블리셔 목록(가능하면) 출력
        pubs = []
        try:
            infos = self.get_publishers_info_by_topic(self.topic)
            for i in infos:
                pubs.append(f"{i.node_namespace}{i.node_name} ({i.topic_type})")
        except Exception as e:
            pubs = [f"(publishers info unavailable: {e})"]

        self.get_logger().info(
            f"Stats: total={self.total} ok={self.ok} empty={self.empty} mismatch={self.mismatch} | pubs={pubs}"
        )


def main():
    rclpy.init()
    node = JointStateEmptyDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
