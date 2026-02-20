#!/usr/bin/env python3
import time
import math
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


def pos_dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def ang_dist(q1, q2):
    # q and -q are same => abs(dot)
    dot = abs(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3])
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)

def t_to_str(t):
    return f"{t.sec}.{t.nanosec:09d}"


class TFStaticDump(Node):
    def __init__(self):
        super().__init__("tf_static_dump")

        self.declare_parameter("max_wait_sec", 8.0)
        self.declare_parameter("settle_sec", 2.0)
        self.declare_parameter("focus_frame", "base_link")
        self.declare_parameter("pos_eps", 1e-6)
        self.declare_parameter("ang_eps", 1e-6)

        self.max_wait = float(self.get_parameter("max_wait_sec").value)
        self.settle = float(self.get_parameter("settle_sec").value)
        self.focus = self.get_parameter("focus_frame").value
        self.pos_eps = float(self.get_parameter("pos_eps").value)
        self.ang_eps = float(self.get_parameter("ang_eps").value)

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.db = {}  # child -> list[(parent,t,q,stamp)]
        self.received_any = False
        self.start_mono = time.monotonic()
        self.last_recv_mono = None

        self.create_subscription(TFMessage, "/tf_static", self.cb, qos)

        self.create_timer(1.0, self.status_tick)
        self.create_timer(0.2, self.finish_tick)

        self.get_logger().info(
            f"/tf_static 덤프 시작 (TRANSIENT_LOCAL). focus_frame='{self.focus}'\n"
            f" - max_wait_sec={self.max_wait}, settle_sec={self.settle}\n"
            f" - 받은 뒤 settle_sec 동안 추가 수신 기다렸다가 요약 출력 후 종료합니다."
        )

    def cb(self, msg: TFMessage):
        now = time.monotonic()
        self.received_any = True
        self.last_recv_mono = now

        for tf in msg.transforms:
            parent = tf.header.frame_id
            child = tf.child_frame_id
            stamp = tf.header.stamp

            tr = tf.transform.translation
            rq = tf.transform.rotation
            t = (float(tr.x), float(tr.y), float(tr.z))
            q = (float(rq.x), float(rq.y), float(rq.z), float(rq.w))

            self.db.setdefault(child, []).append((parent, t, q, stamp))

    def status_tick(self):
        pubs = self.count_publishers("/tf_static")
        elapsed = time.monotonic() - self.start_mono
        self.get_logger().info(
            f"[STATUS] elapsed={elapsed:.1f}s pubs(/tf_static)={pubs} received_any={self.received_any} "
            f"child_count={len(self.db)}"
        )

    def finish_tick(self):
        now = time.monotonic()
        elapsed = now - self.start_mono

        # 아예 못 받았으면: 환경/도메인 문제 가능성
        if (not self.received_any) and (elapsed >= self.max_wait):
            self.get_logger().error(
                f"{self.max_wait:.1f}s 동안 /tf_static를 못 받았습니다. "
                f"pubs(/tf_static)={self.count_publishers('/tf_static')} 인데도 못 받으면 "
                "ROS_DOMAIN_ID/환경(source)/root 실행(-E) 문제일 가능성이 큽니다."
            )
            rclpy.shutdown()
            return

        # 받았으면 마지막 수신 이후 settle 지나면 출력
        if self.received_any and self.last_recv_mono is not None:
            if (now - self.last_recv_mono) >= self.settle:
                self.print_report()
                rclpy.shutdown()

    def print_report(self):
        total = sum(len(v) for v in self.db.values())
        dup_children = [c for c, lst in self.db.items() if len(lst) > 1]

        self.get_logger().info(f"요약: child 개수={len(self.db)}, 총 항목 수={total}")
        if dup_children:
            self.get_logger().warn(f"중복 child_frame_id: {dup_children}")

            # 중복 child 중에서 값이 실제로 다른지도 검사(같으면 '중복이지만 동일값'일 수 있음)
            for child in dup_children:
                items = self.db[child]
                base = items[0]
                for (p,t,q,s) in items[1:]:
                    dp = pos_dist(base[1], t)
                    da = ang_dist(base[2], q)
                    if p != base[0] or dp > self.pos_eps or da > self.ang_eps:
                        self.get_logger().warn(
                            f"[CONFLICT] child='{child}'\n"
                            f"  A: {base[0]}->{child} t={base[1]} q={base[2]} stamp={t_to_str(base[3])}\n"
                            f"  B: {p}->{child} t={t} q={q} stamp={t_to_str(s)}\n"
                            f"  diff: dp={dp:.6e}m da={da:.6e}rad"
                        )

        # focus_frame 관련만 출력
        f = self.focus
        self.get_logger().info(f"--- focus_frame 관련 출력 ('{f}') ---")
        # 1) child가 focus인 항목
        if f in self.db:
            for (p,t,q,s) in self.db[f]:
                self.get_logger().warn(
                    f"[CHILD={f}] {p}->{f} t=({t[0]:+.4f},{t[1]:+.4f},{t[2]:+.4f}) "
                    f"q=({q[0]:+.4f},{q[1]:+.4f},{q[2]:+.4f},{q[3]:+.4f}) stamp={t_to_str(s)}"
                )
        else:
            self.get_logger().info(f"child_frame_id='{f}' 인 static TF는 없습니다.")

        # 2) parent가 focus인 항목들
        found_parent = False
        for child, items in self.db.items():
            for (p,t,q,s) in items:
                if p == f:
                    found_parent = True
                    self.get_logger().info(
                        f"[PARENT={f}] {f}->{child} t=({t[0]:+.4f},{t[1]:+.4f},{t[2]:+.4f}) "
                        f"q=({q[0]:+.4f},{q[1]:+.4f},{q[2]:+.4f},{q[3]:+.4f}) stamp={t_to_str(s)}"
                    )
        if not found_parent:
            self.get_logger().info(f"parent_frame_id='{f}' 인 static TF도 없습니다.")


def main():
    rclpy.init()
    node = TFStaticDump()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
