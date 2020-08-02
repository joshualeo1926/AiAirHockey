import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import random
import math

class field():
    #1200, 600, 180
    def __init__(self, width=1100, height=500, goal_width=170):
        self.width = width
        self.height = height
        self.goal_width = goal_width
        self.goal_y1 = int(self.height/2 - self.goal_width/2)
        self.goal_y2 = int(self.height/2 + self.goal_width/2)

    def get_field(self):
        background = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        background[:][:] = (175, 175, 175)
        for i in range(self.goal_width):
            background[self.goal_y1+i][0:4] = (0, 0, 0)
            background[self.goal_y1+i][self.width-5:self.width-1] = (0, 0, 0)
        for i in range(self.height):
            background[i][self.width//2] = (0, 0, 0)
        image = Image.fromarray(background, "RGB")
        
        return image

class paddle():
    #r=36
    def __init__(self, name, x=0, y=0, radius=30, mass=2000):
        self.name = name
        self.start_x = x
        self.start_y = y
        self.x = x
        self.y = y
        self.radius = radius
        self.mass = mass
        self.last_x = x
        self.last_y = y
        self.velocity = 0
        self.angle = 0
        self.x_vel = 0
        self.y_vel = 0

        self.max_vel = 125
        self.max_accel = 125#100
        self.accel_space = [-self.max_accel, -self.max_accel/2, 0, self.max_accel/2, self.max_accel]
    
    def check_vertical_bounds(self, height):
        if self.y - self.radius <= 0:
            self.y = self.radius
            return True
        elif self.y + self.radius > height:
            self.y = height - self.radius
            return True
        return False

    def check_left_boundary(self, width):
        if self.x - self.radius <= 0:
            self.x = self.radius
            return True
        elif self.x + self.radius > int(width / 2):
            self.x = int(width / 2) - self.radius
            return True
        return False

    def check_right_boundary(self, width):
        if self.x + self.radius > width:
            self.x = width - self.radius
            return True
        elif self.x - self.radius < int(width / 2):
            self.x = int(width / 2) + self.radius
            return True
        return False

    def vec_move(self, angle, speed, time_delta):
        angle = math.pi * angle
        speed = self.max_vel * (speed+1)/2
        self.x += math.sin(angle) * speed * time_delta
        self.y -= math.cos(angle) * speed * time_delta
        self.velocity = speed
        self.angle = angle

    def action(self, choice, time_delta):
        x_choice = choice//5
        y_choice = choice%5

        self.x_vel += self.accel_space[x_choice] * time_delta
        self.y_vel += self.accel_space[y_choice] * time_delta

        self.x += self.x_vel * time_delta
        self.y -= self.y_vel * time_delta

        self.x_vel = (self.x - self.last_x)/time_delta
        self.y_vel = (self.last_y - self.y)/time_delta
        self.velocity = math.sqrt(self.x_vel**2 + self.y_vel**2)
        #self.velocity = math.sqrt(self.x_vel**2 + self.y_vel**2)

        self.last_x = self.x
        self.last_y = self.y

    def acl_move(self, choice, time_delta):
        x_accel = self.max_accel * choice[0]
        y_accel = self.max_accel * choice[1]

        self.x_vel += x_accel * time_delta
        self.y_vel += y_accel * time_delta

        self.x += self.x_vel * time_delta
        self.y -= self.y_vel * time_delta

        self.x_vel = (self.x - self.last_x)/time_delta
        self.y_vel = (self.last_y - self.y)/time_delta
        self.velocity = math.sqrt(self.x_vel**2 + self.y_vel**2)
        #self.velocity = math.sqrt(self.x_vel**2 + self.y_vel**2)

        self.last_x = self.x
        self.last_y = self.y

    def move_tword(self, x, y, speed, time_delta):
        if int(self.x) == int(x) and int(self.y) == int(y):
            return True, 0
        dx = -(int(x) - int(self.x))  
        dy = -(int(y) - int(self.y))
        dx = int(dx)
        dy = int(dy)

        if dx == 0 and dy >= 0:
            angle = 0
        elif dx == 0 and dy <= 0:
            angle = math.pi
        else:
            angle = math.atan2(dy, dx) + 3*math.pi/2
        mag = math.sqrt(dx**2 + dy**2)
        speed = speed/125
        angle = angle/math.pi
        self.vec_move(angle, speed, time_delta)
        return False, angle

    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        self.last_x = self.start_x
        self.last_y = self.start_y
        self.x_vel = 0
        self.y_vel = 0

class puck():
    #r=31
    def __init__(self, x=0, y=0, radius=25, mass=500, friction=0.989):
        self.start_x = x
        self.start_y = y        
        self.x = x
        self.y = y
        self.radius = radius
        self.mass = mass
        self.max_speed = 1500
        self.speed = random.randint(self.max_speed//6, self.max_speed)
        self.angle = random.uniform(-math.pi, math.pi)
        self.friction = friction
        self.last_x = x
        self.last_y = y
        self.x_vel = 0
        self.y_vel = 0

    def move(self, time_delta):
        self.x += math.sin(self.angle) * self.speed * time_delta
        self.y -= math.cos(self.angle) * self.speed * time_delta
        self.speed *= self.friction

        self.x_vel = (self.x - self.last_x)/time_delta
        self.y_vel = (self.y - self.last_y)/time_delta
        #self.speed = math.sqrt(self.x_vel**2 + self.y_vel**2)

        if self.speed > self.max_speed:
            self.speed = self.max_speed

        self.last_x = self.x
        self.last_y = self.y  

    def check_boundary(self, width, height):
        # right side
        if self.x + self.radius > width:
            self.x = 2 * (width - self.radius) - self.x
            self.angle = -self.angle

        # left side
        elif self.x - self.radius < 0:
            self.x = 2 * self.radius - self.x
            self.angle = -self.angle

        # bottom
        if self.y + self.radius > height:
            self.y = 2 * (height - self.radius) - self.y
            self.angle = math.pi - self.angle

        # top
        elif self.y - self.radius < 0:
            self.y = 2 * self.radius - self.y
            self.angle = math.pi - self.angle
        
        if self.angle < 0:
            self.angle = 2*math.pi + self.angle

        if self.angle > 2*math.pi:
            self.angle = self.angle%(2*math.pi)

    def add_vectors(self, vec1, vec2):
        angle1 = vec1[0]
        length1 = vec1[1]
        angle2 = vec2[0]
        length2 = vec2[1]
        x = math.sin(angle1) * length1 + math.sin(angle2) * length2
        y = math.cos(angle1) * length1 + math.cos(angle2) * length2

        length = math.hypot(x, y)
        angle = math.pi / 2 - math.atan2(y, x)
        return angle, length
        
    def collision_paddle(self, paddle):
        dx = self.x - paddle.x
        dy = self.y - paddle.y

        # distance between the centers of the circle
        distance = math.hypot(dx, dy)

        # no collision takes place.
        if distance > self.radius + paddle.radius:
            return False

        # calculates angle of projection.
        tangent = math.atan2(dy, dx)
        temp_angle = math.pi / 2 + tangent
        total_mass = self.mass + paddle.mass       

        sav_speed = self.speed

        # The new vector for puck formed after collision.
        vec_a = (self.angle, self.speed * (self.mass - paddle.mass) / total_mass)
        vec_b = (temp_angle, paddle.velocity * paddle.mass / total_mass)

        (self.angle, self.speed) = self.add_vectors(vec_a, vec_b)

        self.speed = sav_speed*0.8 + paddle.velocity * paddle.mass/(self.mass+200)

        # speed should never exceed a certain limit.
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        #print(self.speed)
        # To prevent puck and paddle from sticking.
        offset = (self.radius + paddle.radius - distance + 1)#0.5 * (self.radius + paddle.radius - distance + 1)
        self.x += math.sin(temp_angle) * offset
        self.y -= math.cos(temp_angle) * offset
        #paddle.x -= math.sin(temp_angle) * offset
        #paddle.y += math.cos(temp_angle) * offset
        return True

    def reset(self, speed, player):
        if player == 1:
            self.angle = random.uniform(-math.pi, 0)
        elif player == 2:
            self.angle = random.uniform(0, math.pi)
        self.speed = speed

        self.x_vel = math.sin(self.angle) * self.speed
        self.y_vel = -math.cos(self.angle) * self.speed

        self.x = self.start_x
        self.y = random.randint(self.radius+5, 600-self.radius-5)#self.start_y

class AirHockeyEnvironment():
    def __init__(self, dt=1/30):
        self.env = field()
        self.paddle1 = paddle(name='1', x=56, y=self.env.height//2)
        self.paddle2 = paddle(name='2', x=self.env.width-56, y=self.env.height//2)
        self.puck = puck(x=self.env.width//2, y=self.env.height//2)

        self.field = self.env.get_field()

        self.n_time_step = 0

        self.puck_strike_reward = 5
        self.paddel_vel_reward = 50
        self.accuracy_reward = 100
        self.decay_rate = 0.01
        self.time_step_penalty = 1
        self.width_window = self.env.goal_width/2 #200

        self.dt = dt
        self.reward1 = 0
        self.np_field = None

        self.time_step_lim = 1000

        self.wanderd = 'center'
        self.wander = False
        self.attack = False

        self.paddel_1_stats = [0, 0, 0, 0, 0, 0]
        self.paddel_2_stats = [0, 0, 0, 0, 0, 0]

        self.closest_point = [0, 0]

    def get_env_state(self):
        env_state = []
        env_state.append(self.paddle1.x)
        env_state.append(self.paddle1.y)
        env_state.append(self.paddle1.x_accel)
        env_state.append(self.paddle1.y_accel)
        env_state.append(self.paddle1.velocity)
        env_state.append(self.paddle1.angle)
        env_state.append(self.paddle2.x)
        env_state.append(self.paddle2.y)
        env_state.append(self.paddle2.x_accel)
        env_state.append(self.paddle2.y_accel)
        env_state.append(self.paddle2.velocity)
        env_state.append(self.paddle2.angle)
        env_state.append(self.puck.x)
        env_state.append(self.puck.y)
        env_state.append(self.puck.angle)
        env_state.append(self.puck.speed)
        return env_state

    def set_state(self, env_state):
        self.paddle1.x = env_state[0]
        self.paddle1.y = env_state[1]
        self.paddle1.x_accel = env_state[2]
        self.paddle1.y_accel = env_state[3]
        self.paddle1.velocity = env_state[4]
        self.paddle1.angle = env_state[5]
        self.paddle2.x = env_state[6]
        self.paddle2.y = env_state[7]
        self.paddle2.x_accel = env_state[8]
        self.paddle2.y_accel = env_state[9]
        self.paddle2.velocity = env_state[10]
        self.paddle2.angle = env_state[11] 
        self.puck.x = env_state[12] 
        self.puck.y = env_state[13] 
        self.puck.angle = env_state[14] 
        self.puck.speed = env_state[15] 

    def inside_goal(self, side):
        """ Returns true if puck is within goal boundary"""
        if side == 0: 
            return (self.puck.x - self.puck.radius <= 3) and (self.puck.y >= self.env.goal_y1) and (self.puck.y <= self.env.goal_y2)

        if side == 1:
            return (self.puck.x + self.puck.radius >= self.env.width-3) and (self.puck.y >= self.env.goal_y1) and (self.puck.y <= self.env.goal_y2)

    def get_state(self, done):
        if done:
            self.cycles = 0
            self.paddle1.x_accel = 0
            self.paddle1.y_accel = 0
            self.paddle2.x_accel = 0
            self.paddle2.y_accel = 0
            self.paddle1.velocity = 0
            self.paddle2.velocity = 0
            self.paddle1.angle = 0
            self.paddle2.angle = 0
        return self.paddle1.x, self.paddle1.x_vel, self.paddle1.y, self.paddle1.y_vel, self.puck.x, self.puck.x_vel, self.puck.y, self.puck.y_vel, done

    def check_goals(self):
        if self.inside_goal(0):
            return True

        if self.inside_goal(1):
            return True

    def check_paddle_collisions(self):
        if self.puck.collision_paddle(self.paddle1):
            return True, False

        if self.puck.collision_paddle(self.paddle2):
            return False, True

        return False, False

    def puck_traj(self, time_delta):
        x_along_line = self.puck.x
        y_along_line = self.puck.y
        initial_x = self.puck.x
        initial_y = self.puck.y
        temp_speed = self.puck.speed
        line_angle = self.puck.angle
        hitting_bounds = False
        while not hitting_bounds:
            temp_speed *= self.puck.friction
            if temp_speed < 0.05:
                break   

            # check along line
            x_along_line = x_along_line + math.sin(line_angle) * temp_speed * time_delta
            y_along_line = y_along_line - math.cos(line_angle) * temp_speed * time_delta

            # right side
            if x_along_line + self.puck.radius > self.env.width or x_along_line - self.puck.radius < 0 or y_along_line + self.puck.radius > self.env.height or y_along_line - self.puck.radius < 0:
                hitting_bounds = True

        return initial_x, initial_y, x_along_line, y_along_line

    def paddle_control(self):
        puck_x = self.puck.x + 2 * math.sin(self.puck.angle) * self.puck.speed * self.dt
        puck_y = self.puck.y - 2 * math.cos(self.puck.angle) * self.puck.speed * self.dt
        puck_ang = self.puck.angle
        ang = 0
        vel = 0
        if puck_x + self.puck.radius > self.env.width or puck_x - self.puck.radius < 0 or puck_y + self.puck.radius > self.env.height or puck_y - self.puck.radius < 0:
                if puck_x + self.puck.radius > self.env.width or puck_x - self.puck.radius < 0:
                    puck_ang = -puck_ang
                else:
                    puck_ang = math.pi - puck_ang

        if puck_x > self.env.width/2 + 150 and self.puck.speed > 80:
            self.wander = False
            self.attack = True
        elif puck_x > self.env.width/2 and self.puck.speed <= 80:
            self.wander = False
            self.attack = True
        else:
            self.wander = True
            self.attack = False

        if self.wander:
            speed = 100
            dist = math.sqrt((self.env.width-100 - self.paddle2.x)**2 + (self.env.height/2 - self.paddle2.y)**2)
            if dist < 50:
                speed = 50 * dist/49
            if self.wanderd == 'center':
                if self.paddle2.move_tword(self.env.width-100, self.env.height/2, speed, self.dt):
                    self.wanderd = 'top'
            if self.wanderd == 'top':
                if self.paddle2.x != self.env.width-200 and self.paddle2.y != self.env.height/2:
                    self.wanderd = 'center'
            
        elif self.attack:

            angle = 0;
            
            if math.sqrt((self.puck.x - self.paddle2.x)**2 + (self.puck.y - self.paddle2.y)**2) >= 100:
                vel = 125 * (1 - (math.sqrt((self.puck.x + self.puck.radius + 20 - self.paddle2.x)**2 + (self.puck.y - self.paddle2.y)**2)/math.sqrt((self.env.width/2)**2 + (self.env.height)**2))**2)
                _, ang = self.paddle2.move_tword(puck_x + self.puck.radius + 20, puck_y, vel, self.dt)


            if math.sqrt((self.puck.x - self.paddle2.x)**2 + (self.puck.y - self.paddle2.y)**2) < 100:
                vel = random.randint(100, 125)
                _, ang = self.paddle2.move_tword(puck_x, puck_y + random.randint(-self.puck.radius, self.puck.radius), vel, self.dt)

    def terminate(self):
        self.n_time_step = 0
        rc = 10
        rd = self.get_dist_reward()
        rv = self.get_vel_reward()
    
    def reset(self):
        self.n_time_step = 0
        self.puck.reset(random.randint(self.puck.max_speed/6, self.puck.max_speed), 1)
        self.paddle1.reset()
        self.paddle2.reset()

        return [self.paddle1.x, self.paddle1.x_vel, self.paddle1.y, self.paddle1.y_vel, self.puck.x, self.puck.x_vel, self.puck.y, self.puck.y_vel], \
                [self.paddle2.x, self.paddle2.x_vel, self.paddle2.y, self.paddle2.y_vel, self.puck.x, self.puck.x_vel, self.puck.y, self.puck.y_vel]

    def add_vectors(self, vec1, vec2):
        angle1 = vec1[0]
        length1 = vec1[1]
        angle2 = vec2[0]
        length2 = vec2[1]
        x = math.sin(angle1) * length1 + math.sin(angle2) * length2
        y = math.cos(angle1) * length1 + math.cos(angle2) * length2

        length = math.hypot(x, y)
        angle = math.pi / 2 - math.atan2(y, x)
        return angle, length

    def get_dist_reward(self, paddle):
        x_along_line = self.puck.x
        y_along_line = self.puck.y
        temp_speed = self.puck.speed
        line_angle = self.puck.angle
        smallest_x = 9999.0
        smallest_y = 9999.0
        smallest_dist = 9999.0
        reward = 0

        while temp_speed >= 0.05:
            temp_speed *= self.puck.friction
            x_along_line = x_along_line + math.sin(line_angle) * temp_speed * self.dt
            y_along_line = y_along_line - math.cos(line_angle) * temp_speed * self.dt

            if x_along_line + self.puck.radius > self.env.width:
                x_along_line = 2 * (self.env.width - self.puck.radius) - x_along_line
                line_angle = -line_angle

            elif x_along_line - self.puck.radius < 0:
                x_along_line = 2 * self.puck.radius - x_along_line
                line_angle = -line_angle

            if y_along_line + self.puck.radius > self.env.height:
                y_along_line = 2 * (self.env.height - self.puck.radius) - y_along_line
                line_angle = math.pi - line_angle

            elif y_along_line - self.puck.radius < 0:
                y_along_line = 2 * self.puck.radius - y_along_line
                line_angle = math.pi - line_angle

            pdx = x_along_line - paddle.x
            pdy = y_along_line - paddle.y

            distance = math.hypot(pdx, pdy)

            if distance <= self.puck.radius + paddle.radius:
                tangent = math.atan2(pdy, pdx)
                temp_angle = math.pi / 2 + tangent
                total_mass = self.puck.mass + paddle.mass
                
                speed_multiplier = 2

                vec_a = (line_angle, temp_speed * (self.puck.mass - paddle.mass) / total_mass)
                vec_b = (temp_angle, speed_multiplier * paddle.velocity * paddle.mass / total_mass)

                (line_angle, _) = self.add_vectors(vec_a, vec_b)

                # To prevent puck and paddle from sticking.
                offset = 0.5 * (self.puck.radius + paddle.radius - distance + 1)
                x_along_line += math.sin(temp_angle) * offset
                y_along_line -= math.cos(temp_angle) * offset

            if paddle.name == '1':
                dx = self.env.width - x_along_line
                dy = self.env.height/2 - y_along_line
            elif paddle.name == '2':
                dx = 0 - x_along_line
                dy = self.env.height/2 - y_along_line

            if math.sqrt(dx**2 + dy**2) < smallest_dist:
                smallest_x = x_along_line
                smallest_y = y_along_line
                smallest_dist = math.sqrt(dx**2 + dy**2)
        
            if (x_along_line - self.puck.radius <= 3) and (y_along_line >= self.env.goal_y1) and (self.puck.y <= self.env.goal_y2):
                if paddle.name == '1':
                    reward -= self.accuracy_reward
                elif paddle.name == '2':
                    reward += 1.5*self.accuracy_reward
                return reward, [smallest_x, smallest_y]

            if (x_along_line + self.puck.radius >= self.env.width-3) and (y_along_line >= self.env.goal_y1) and (self.puck.y <= self.env.goal_y2):
                if paddle.name == '1':
                    reward += 1.5*self.accuracy_reward
                elif paddle.name == '2':
                    reward -= self.accuracy_reward
                return reward, [smallest_x, smallest_y]
            
        if smallest_dist <= self.width_window:
            reward += self.accuracy_reward
        elif smallest_dist > self.width_window:
            reward += self.accuracy_reward * math.exp(-self.decay_rate * (smallest_dist - self.width_window))

        return reward, [smallest_x, smallest_y]

    def get_vel_reward(self, paddle):
        if paddle.name == '1':
            dx = -(int(self.env.width) - int(self.puck.x))  
            dy = -(int(self.env.height/2) - int(self.puck.y))
        elif paddle.name == '2':
            dx = -(int(0) - int(self.puck.x))  
            dy = -(int(self.env.height/2) - int(self.puck.y))
        dx = int(dx)
        dy = int(dy)
        angle = 0
        if dx == 0 and dy >= 0:
            angle = 0
        elif dx == 0 and dy <= 0:
            angle = math.pi
        else:
            angle = math.atan2(dy, dx) + 3*math.pi/2
        
        max_vect = np.array([math.sin(angle)*self.puck.max_speed/100, -math.cos(angle)*self.puck.max_speed/100])
        current_vect = np.array([math.sin(self.puck.angle)*self.puck.speed/100, -math.cos(self.puck.angle)*self.puck.speed/100])
        projected = current_vect @ max_vect
        V = projected
        reward = 2 * (np.sign(V)*V**2/1000)
        return reward

    def get_state(self, reward1, reward2, done):
        return [self.paddle1.x, self.paddle1.x_vel, self.paddle1.y, self.paddle1.y_vel, self.puck.x, self.puck.x_vel, self.puck.y, self.puck.y_vel], \
                [self.env.width - self.paddle2.x, -self.paddle2.x_vel, self.paddle2.y, self.paddle2.y_vel, self.env.width - self.puck.x, -self.puck.x_vel, self.puck.y, self.puck.y_vel], \
                reward1, reward2, done
                
    def calulate_reward(self, paddle):
        vel_reward = self.get_vel_reward(paddle)
        dist_reward, self.closest_point = self.get_dist_reward(paddle)
        vel_2_reward = self.paddel_vel_reward * (paddle.velocity/(self.puck.max_speed/4))**2

        if dist_reward < 0:
            vel_reward = 0
            vel_2_reward = 0

        reward = (self.puck_strike_reward*np.sign(vel_reward)) + dist_reward + vel_reward + vel_2_reward

        if paddle.name == '1':
            self.paddel_1_stats = [self.puck.speed, paddle.velocity, reward, vel_reward, vel_2_reward, dist_reward]
        elif paddle.name == '2':
            self.paddel_2_stats = [self.puck.speed, paddle.velocity, reward, vel_reward, vel_2_reward, dist_reward]

        return reward

    def step(self, action1, action2, render=True):
        if self.puck.x > self.env.width//2 and self.puck.speed < 5 and False:
            return self.get_state(0, 0, True)

        if self.n_time_step >= self.time_step_lim:
            return self.get_state(-self.time_step_lim, -self.time_step_lim, True)

        self.n_time_step += 1
        np_field = np.array(self.field)

        self.puck.move(self.dt)
        
        self.paddle1.acl_move(action1, self.dt) 
        if self.paddle1.check_vertical_bounds(self.env.height):
            return self.get_state(-self.time_step_lim, 0, True)
        if self.paddle1.check_left_boundary(self.env.width):
            return self.get_state(-self.time_step_lim, 0, True)

        self.paddle_control()
        #self.paddle2.action(action2, self.dt)
        if self.paddle2.check_vertical_bounds(self.env.height):
            #return self.get_state(0, -self.time_step_lim, True)
            pass
        if self.paddle2.check_right_boundary(self.env.width):
            #return self.get_state(0, -self.time_step_lim, True)
            pass

        self.check_goals()

        self.puck.check_boundary(self.env.width, self.env.height)
        paddle1_bool, paddle2_bool = self.check_paddle_collisions()
        
        if paddle1_bool:
            reward = self.calulate_reward(self.paddle1)
            return self.get_state(reward, 0, True)

        #if paddle2_bool:
        #    reward = self.calulate_reward(self.paddle2)
        #    return self.get_state(0, reward, True)
        
        if render:        
            cv2.putText(np_field, 'cycle: %d /' % self.n_time_step + str(self.time_step_lim), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
            cv2.putText(np_field, 'P1_rew: %.2f' % self.paddel_1_stats[2] + ' Dst: %.2f' % self.paddel_1_stats[5] + ' Vel1: %.2f' % self.paddel_1_stats[3] + \
                        ' Vel2: %.2f' % self.paddel_1_stats[4], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
            cv2.putText(np_field, 'P1_vel: %.2f' % self.paddel_1_stats[1] + ' Pu_vel: %.2f' % self.paddel_1_stats[0], (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
            cv2.circle(np_field, (int(self.closest_point[0]), int(self.closest_point[1])), (5), (255, 0, 255), -1)
            
            # PADDLE 1
            cv2.circle(np_field, (int(self.paddle1.x), int(self.paddle1.y)), (self.paddle1.radius), (0, 0, 255), -1)
            cv2.circle(np_field, (int(self.paddle1.x), int(self.paddle1.y)), (self.paddle1.radius-12), (0, 0, 100), 14)

            # PADDLE 2
            cv2.circle(np_field, (int(self.paddle2.x), int(self.paddle2.y)), (self.paddle2.radius), (255, 0, 0), -1)
            cv2.circle(np_field, (int(self.paddle2.x), int(self.paddle2.y)), (self.paddle2.radius-12), (100, 0, 0), 14)

            # PUCK
            cv2.circle(np_field, (int(self.puck.x), int(self.puck.y)), (self.puck.radius), (220, 220, 220), -1)
            cv2.circle(np_field, (int(self.puck.x), int(self.puck.y)), (self.puck.radius-5), (200, 200, 200), -1)
            cv2.imshow("field", np_field)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.break_step = True
            else:
                self.break_step = False

        return self.get_state(-1, -1, False)