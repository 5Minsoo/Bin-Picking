#!/usr/bin/env python3
import rclpy
from rclpy.executors import MultiThreadedExecutor

from moveit_helper_functions import MoveItMoveHelper  # 파일/클래스명은 사용자 환경 그대로


class MoveitPractice(MoveItMoveHelper):
    def __init__(self):
        super().__init__()

        self._sent = False
        # 노드가 뜬 직후(약간 딜레이) 한 번만 실행
        self._timer = self.create_timer(0.2, self._run_once)

    def _run_once(self):
        if self._sent:
            return
        self._sent = True
        self._timer.cancel()

        # 사용자가 원한 딱 1줄 동작
        self.get_logger().info("Sending OMPL goal...")
        self.move_ompl_to_pose([0.8, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0])


def main():
    rclpy.init()

    node = MoveitPractice()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
