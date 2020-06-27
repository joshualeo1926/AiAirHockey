import pygame
import random as random
import math
import constants as const

class Puck:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.lx = x
        self.ly = y
        self.radius = const.PUCK_SIZE
        self.speed = random.randint(const.MAX_SPEED/6, const.MAX_SPEED)
        self.mass = const.PUCK_MASS
        self.angle = random.uniform(-math.pi, math.pi)

    def move(self, time_delta):
        self.lx = self.x
        self.ly = self.y  
        self.x += math.sin(self.angle) * self.speed * time_delta
        self.y -= math.cos(self.angle) * self.speed * time_delta
        self.speed *= const.FRICTION

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
        
        speed_multiplier = 2

        # The new vector for puck formed after collision.
        vec_a = (self.angle, self.speed * (self.mass - paddle.mass) / total_mass)
        vec_b = (temp_angle, speed_multiplier * paddle.velocity * paddle.mass / total_mass)

        (self.angle, self.speed) = self.add_vectors(vec_a, vec_b)

        # speed should never exceed a certain limit.
        if self.speed > const.MAX_SPEED:
            self.speed = const.MAX_SPEED

        # To prevent puck and paddle from sticking.
        offset = 0.5 * (self.radius + paddle.radius - distance + 1)
        self.x += math.sin(temp_angle) * offset
        self.y -= math.cos(temp_angle) * offset
        paddle.x -= math.sin(temp_angle) * offset
        paddle.y += math.cos(temp_angle) * offset
        return True
   
    def reset(self, speed, player):
        if player == 1:
            self.angle = random.uniform(-math.pi, 0)
        elif player == 2:
            self.angle = random.uniform(0, math.pi)
        self.speed = speed
        self.x = const.WIDTH / 2
        self.y = const.HEIGHT / 2

    def draw(self, screen):
        pygame.draw.circle(screen, const.WHITE, (int(self.x), int(self.y)), self.radius)