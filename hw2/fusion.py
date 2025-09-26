import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import math

class YawSub(Node):

    def __init__(self):
        super().__init__('yaw_subscriber')

        # create subscribers, publishers
        self.imu_subscription = self.create_subscription(Float64, '/yaw/imu', self.imu_listener_callback, 10)
        self.odom_subscription = self.create_subscription(Float64, '/yaw/odom', self.odom_listener_callback, 10)
        self.publisher = self.create_publisher(Float64, '/yaw/fused', 10)

        # create the alpha parameter
        self.declare_parameter('alpha', 0.98)
        self.alpha = self.get_parameter('alpha').value

        # set defaults for latest values
        self.imu_val = 0
        self.odom_val = 0

    def update_fused_yaw(self, msg):
        # update theta and wrap it
        fused_theta = self.alpha * self.imu_val + (1 - self.alpha) * self.odom_val
        fused_theta_wrapped = (fused_theta + math.pi) % (2 * math.pi) - math.pi

        # publish wrapped theta
        res = Float64()
        res.header = msg.header
        res.data = fused_theta_wrapped
        self.publisher.publish(res)

    def imu_listener_callback(self, msg):
        self.imu_val = msg.data
        self.update_fused_yaw(msg)

    def odom_listener_callback(self, msg):
        self.odom_val = msg.data
        self.update_fused_yaw(msg)

def main(args=None):
    rclpy.init(args=args)

    yaw_subscriber = YawSub()
    rclpy.spin(yaw_subscriber)

    # Destroy the node explicitly
    yaw_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()