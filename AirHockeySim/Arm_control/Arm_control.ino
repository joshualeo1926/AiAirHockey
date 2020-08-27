#include <Servo.h>

class Arm
{
  public:
    float link_1_len = 400;
    float link_2_len = 300;
    float joint_1_angle = 0;
    float joint_2_angle = 0;

    float joint_1_x = 0;
    float joint_1_y = 0;
    
    float joint_2_x = joint_1_x + link_1_len*sin(joint_1_angle + M_PI/2);
    float joint_2_y = joint_1_x + link_1_len*cos(joint_1_angle + M_PI/2);

    float end_effector_x = joint_2_x + link_2_len*sin(joint_2_angle + M_PI/2);
    float end_effector_y = joint_2_y + link_2_len*cos(joint_2_angle + M_PI/2);

    Servo joint_1;
    Servo joint_2;

    float joint_limit_max = ((float)140*M_PI)/((float)180);
    float joint_limit_min = ((float)-90*M_PI)/((float)180);

    int steps = 50;
    int delay_ = 2;
    
  public:
    void ikine(float x, float y)
    {
      if(x >= -260.0 && x <= 260.0 && y >= 240.0 && y <= 610.0)
      {
        float q1 = M_PI - acos(((float)sq(link_1_len) + sq(link_2_len) - sq(x) - sq(y))/((float)2*link_1_len*link_2_len));
        float q0 = atan2(x, y) - atan2((link_2_len*sin(q1)), (link_1_len + link_2_len*cos(q1)));
        
        joint_1_angle = q0;
        joint_2_angle = q1;
  
        joint_2_x = joint_1_x + link_1_len*sin(q0);
        joint_2_y = joint_1_x + link_1_len*cos(q0);
        end_effector_x = joint_2_x + link_2_len*sin(q0+q1);
        end_effector_y = joint_2_y + link_2_len*cos(q0+q1);
  
        float m = (y)/(x);
        float b = 0.0 - m * 0.0;
        float m_s = -1.0/m;
        float b_s = joint_2_y - m_s * joint_2_x;
        float xi = (b-b_s)/(m_s-m);
        
        if (m == INFINITY)
        {
          xi = joint_1_x;
        }
        float yi = m_s * xi + b_s;
        float dx = joint_2_x - xi;
        float dy = joint_2_y - yi;
        joint_2_x = xi - dx;
        joint_2_y = yi - dy;
  
        float theta = atan2(joint_2_y, joint_2_x);
        if(theta < 0.0)
        {
          theta += 2 * M_PI;
        }
        joint_1_angle = theta;

        if(joint_1_angle > joint_limit_max){joint_1_angle = -360 + joint_1_angle;}
        if(joint_2_angle > joint_limit_max){joint_2_angle = -360 + joint_2_angle;}
      }
    }

    float joint_1_rad_to_pwm(float rad)
    {
      return 63+int((((float)63)/(M_PI/2))*joint_1_angle);
    }

    float joint_2_rad_to_pwm(float rad)
    {
      return 98-int((((float)63)/(M_PI/2))*joint_2_angle);
    }

    void move_to(float x, float y)
    {
      if(end_effector_x != x && end_effector_y != y)
      {
        for(int i=0; i<steps; i++)
        {
          float current_x = (float)end_effector_x*((float)1-(float)i/(float)steps) + x*((float)i/(float)steps);
          float current_y = (float)end_effector_y*((float)1-(float)i/(float)steps) + y*((float)i/(float)steps);
          ikine(current_x, current_y);
          joint_1.write(joint_1_rad_to_pwm(joint_1_angle));
          joint_2.write(joint_2_rad_to_pwm(joint_2_angle));
          delay(delay_);
        }
      }
    }
};

Arm two_dof_arm;
unsigned long time_1 = 0;
unsigned long time_2 = 0; 
void setup() {
  two_dof_arm.joint_1.attach(9);
  two_dof_arm.joint_2.attach(10);
  Serial.begin(9600);
  two_dof_arm.ikine(200, 260);
  two_dof_arm.joint_1.write(two_dof_arm.joint_1_rad_to_pwm(two_dof_arm.joint_1_angle));
  two_dof_arm.joint_2.write(two_dof_arm.joint_2_rad_to_pwm(two_dof_arm.joint_2_angle));
  time_1 = millis();
  delay(5000);
  two_dof_arm.move_to(-10, 580);
  
  time_2 = millis();
  Serial.println(time_2 - time_1);
  //delay(10000);
}

void loop() {
  //two_dof_arm.move_to(100, 450);
  //delay(50);
  //two_dof_arm.move_to(250, 600);
  //delay(50);
  //two_dof_arm.move_to(100, 450);
  //delay(50);
  //two_dof_arm.move_to(0, 260);
  //delay(50);
  Serial.println('\n');
  Serial.println(two_dof_arm.joint_1_angle);
  Serial.println(two_dof_arm.joint_2_angle); 
  Serial.println(two_dof_arm.joint_1_angle*180/M_PI);
  Serial.println(two_dof_arm.joint_2_angle*180/M_PI);  
  Serial.println('\n');
  //-90min 140max
  //
}
