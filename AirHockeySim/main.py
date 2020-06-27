import sys
import os
from pygame.locals import *
import math
import random

from globals import *
import constants as const
from paddle import Paddle
from puck import Puck

paddle1 = Paddle(const.PADDLE1X, const.PADDLE1Y)
paddle2 = Paddle(const.PADDLE2X, const.PADDLE2Y)
puck = Puck(width / 2, height / 2)

def init():
	global clock, screen, smallfont, roundfont
	pygame.mixer.pre_init(44100, -16, 2, 2048)
	pygame.mixer.init()
	pygame.init()

	if len(sys.argv) > 1:
		pygame.display.set_caption('Air Hockey ' + str(sys.argv[1]))
	else:
		pygame.display.set_caption('Air Hockey')
	screen = pygame.display.set_mode((width, height))

	smallfont = pygame.font.SysFont("comicsans", 35)
	roundfont = pygame.font.SysFont("comicsans", 45)

	clock = pygame.time.Clock()

def render_field(background_color):
	screen.fill(background_color)
	# center circle
	pygame.draw.circle(screen, const.WHITE, (int(width / 2), int(height / 2)), 70, 5)
	# borders
	pygame.draw.rect(screen, const.WHITE, (0, 0, width, height), 5)
	# D-box
	pygame.draw.rect(screen, const.WHITE, (0, int(height / 2) - 150, 150, 300), 5)
	pygame.draw.rect(screen, const.WHITE, (int(width) - 150, int(height / 2) - 150, 150, 300), 5)
	# goals
	pygame.draw.rect(screen, const.BLACK, (0, const.GOAL_Y1, 5, const.GOAL_WIDTH))
	pygame.draw.rect(screen, const.BLACK, (width - 5, const.GOAL_Y1, 5, const.GOAL_WIDTH))
	# Divider
	pygame.draw.rect(screen, const.WHITE, (int(width / 2), 0, 3, height))

def reset_game():
    global shot_on_goal_1
    global shot_on_goal_2
    global puck_hit_by
    shot_on_goal_1 = False
    shot_on_goal_2 = False
    puck_hit_by = "none"
    puck.reset(random.randint(const.MAX_SPEED/6, const.MAX_SPEED), random.randint(1,2))
    paddle1.reset(20, height / 2)
    paddle2.reset(width - 20, height / 2)

def puck_traj(time_delta, shot_on_goal_1, shot_on_goal_2, puck):
	x_along_line = puck.x
	y_along_line = puck.y
	initial_x = puck.x
	initial_y = puck.y
	temp_speed = puck.speed
	line_angle = puck.angle
	hitting_bounds = False
	while not hitting_bounds:
		temp_speed *= const.FRICTION
		if temp_speed < 0.05:
			break   

		# check along line
		x_along_line = x_along_line + math.sin(line_angle) * temp_speed * time_delta
		y_along_line = y_along_line - math.cos(line_angle) * temp_speed * time_delta

		# goal attempt
		if (x_along_line - puck.radius <= 0) and (y_along_line >= const.GOAL_Y1) and (y_along_line <= const.GOAL_Y2) and not shot_on_goal_2:
			shot_on_goal_1 = True
		elif (x_along_line + puck.radius >= width) and (y_along_line >= const.GOAL_Y1) and (y_along_line <= const.GOAL_Y2) and not shot_on_goal_1:
			shot_on_goal_2 = True

		# right side
		if x_along_line + puck.radius > width or x_along_line - puck.radius < 0 or y_along_line + puck.radius > height or y_along_line - puck.radius < 0:
			hitting_bounds = True

	return initial_x, initial_y, x_along_line, y_along_line, shot_on_goal_1, shot_on_goal_2

def inside_goal(side):
    """ Returns true if puck is within goal boundary"""
    if side == 0:
        return (puck.x - puck.radius <= 0) and (puck.y >= const.GOAL_Y1) and (puck.y <= const.GOAL_Y2)

    if side == 1:
        return (puck.x + puck.radius >= width) and (puck.y >= const.GOAL_Y1) and (puck.y <= const.GOAL_Y2)

def sim_loop(player1_color, player2_color, background_color):
	# POINT VARS
	shot_on_goal_1 = False
	shot_on_goal_2 = False
	freez_points = False
	puck_hit_by = "none"
	paddle_1_saves = 0
	paddle_2_saves = 0
	paddle_1_shots = 0
	paddle_2_shots = 0
	paddle_1_goals = 0
	paddle_2_goals = 0
	nn_score1 = 0
	nn_score2 = 0

	last_score1 = 0
	last_score2 = 0

	round_time = clock.get_time()

	while True:
		# ROUND STAGNATION
		round_time += clock.get_time()
		if round_time >= const.ROUND_TIME_LIM * 1000 or puck.speed <= 0.02:
			round_time = 0
			reset_game()
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
				sys.exit()

		# DT
		time_delta = clock.get_time() / 1000.0

		if puck.angle < 0:
			puck.angle = 2*math.pi + puck.angle

		if puck.angle > 2*math.pi:
			puck.angle = puck.angle%(2*math.pi)

		os.system('cls' if os.name == 'nt' else 'clear')
		print('Time: ' + str(round_time/1000) +
			'\n\n==Left side==' + 
			'\nAngle: ' + str(round(380 - puck.angle*180/math.pi, 3)) + 
			'\nSpeed: ' + str(round(puck.speed, 3)) + 
			'\nX: ' + str(round(puck.x, 3)) + 
			'\nY: '+ str(round(puck.y, 3)) +
			'\nShots: ' + str(paddle_1_shots) +
			'\nSaves: ' + str(paddle_1_saves) +
			'\nGoals: ' + str(paddle_1_goals) +
			'\nScore: ' + str(nn_score1) +
			'\n\n==Right side==' + 
			'\nAngle: ' + str(round(puck.angle*180/math.pi, 3)) +
			'\nSpeed: ' + str(round(puck.speed, 3)) + 
			'\nX: ' + str(round(width - puck.x, 3)) + 
			'\nY: '+ str(round(puck.y, 3)) +
			'\nShots: ' + str(paddle_2_shots) +
			'\nSaves: ' + str(paddle_2_saves) +
			'\nGoals: ' + str(paddle_2_goals) +
			'\nScore: ' + str(nn_score2))


		# UPDATE OBJECTS
		paddle1.vec_move(random.uniform(-math.pi, math.pi), random.randint(0, 800), time_delta)
		paddle1.check_vertical_bounds(height)
		paddle1.check_left_boundary(width)

		paddle2.vec_move(random.uniform(-math.pi, math.pi), random.randint(0, 800), time_delta)
		paddle2.check_vertical_bounds(height)
		paddle2.check_right_boundary(width)

		puck.move(time_delta)

		# GOAL DETECTION
		if inside_goal(0):
			if puck_hit_by == "paddle1" or puck_hit_by == "none":
				nn_score1 -= 500
			else:
				nn_score2 += 500
				paddle_2_goals += 1
			reset_game()

		if inside_goal(1):
			if puck_hit_by == "paddle2" or puck_hit_by == "none":
				nn_score2 -= 500
			else:
				nn_score1 += 500
				paddle_1_goals += 1
			reset_game()

		# COLLISION
		puck.check_boundary(width, height)
		if puck.collision_paddle(paddle1):
			nn_score1 += 10
			if shot_on_goal_1 and puck_hit_by == "paddle2":
				paddle_1_saves += 1
				nn_score1 += 500
			freez_points = False
			puck_hit_by = "paddle1"
			shot_on_goal_1 = False
			shot_on_goal_2 = False

		if puck.collision_paddle(paddle2):
			nn_score2 += 10
			if shot_on_goal_2 and puck_hit_by == "paddle1":
				paddle_2_saves += 1
				nn_score2 += 500
			freez_points = False
			puck_hit_by = "paddle2"
			shot_on_goal_1 = False
			shot_on_goal_2 = False

		# RENDERING
		render_field(background_color)
		paddle1.draw(screen, player1_color)
		paddle2.draw(screen, player2_color)
		puck.draw(screen)

		colour_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
		initial_x, initial_y, x_along_line, y_along_line, shot_on_goal_1, shot_on_goal_2 = puck_traj(time_delta, shot_on_goal_1, shot_on_goal_2, puck)
		pygame.draw.line(screen, colour_list[1], (int(initial_x), int(initial_y)), (int(x_along_line), int(y_along_line)), 3)

		if shot_on_goal_2 and puck_hit_by == "paddle1" and not freez_points:
			paddle_1_shots += 1
			nn_score1 += 100
			freez_points = True
		elif shot_on_goal_1 and puck_hit_by == "paddle2" and not freez_points:
			paddle_2_shots += 1
			nn_score2 += 100
			freez_points = True

		pygame.display.flip()
		clock.tick(const.FPS)

		if last_score1 != nn_score1 or last_score2 != nn_score2 and 0:
			if len(sys.argv) > 1:
				print(str(sys.argv[1]),"nn_score1" , nn_score1, "nn_score2", nn_score2)
			else:
				print("nn_score1" , nn_score1, "nn_score2", nn_score2)
		last_score1 = nn_score1
		last_score2 = nn_score2

if __name__ == "__main__":
	init()
	while True:
		player1_color = (255, 0, 0)
		player2_color = (0, 0, 255)
		background_color = (127, 127, 127)
		init()
		sim_loop(player1_color, player2_color, background_color)