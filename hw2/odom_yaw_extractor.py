import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from tf_transformations import euler_from_quaternion
import math

class OdomSub(Node):

    def __init__(self):
        super().__init__('odom_subscriber')
        self.subscription = self.create_subscription(Odometry, '/odom', self.listener_callback, 10)
        self.publisher = self.create_publisher(Float64, '/yaw/odom', 10)

    def listener_callback(self, msg):
        # get quaternion and convert to euler
        vals = msg.pose.pose.orientation
        quat = [vals.x, vals.y, vals.z, vals.w]
        _, _, yaw = euler_from_quaternion(quat)

        # wrap and publish
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        res = Float64()
        res.data = yaw
        self.pub.publish(res)

def main(args=None):
    rclpy.init(args=args)

    odom_subscriber = OdomSub()
    rclpy.spin(odom_subscriber)

    # Destroy the node explicitly
    odom_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()