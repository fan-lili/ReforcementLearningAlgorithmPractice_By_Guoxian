# 搭建鸳鸯系统的马尔可夫决策过程，具体任务为一只鸳鸯通过障碍物到达指定地点
import pygame
import numpy as np
from load import *
import random

class YuanYangEnv:
    def __init__(self):
        # 定义马尔可夫过程的几个元素（S,A,P,R,gamma)
        self.states = []  # 包含100个状态
        for i in range(0, 100):
            self.states.append(i)
        self.actions = ['e', 's', 'w', 'n']  # 动作空间，上下左右
        self.gamma = 0.95  # 折扣因子
        self.value = np.zeros((10, 10))  # 每个状态的值函数，事先不知
        self.action_value = np.zeros((100, 4))  # 行为值函数，在无模型算法中使用
        self.path = []  # 经过的路径

        self.viewer = None  # 渲染窗口
        self.FPSCLOCK = pygame.time.Clock()

        self.screen_size = (1200, 800)  # 窗口大小
        self.bird_position = (0, 0)
        self.limit_distance_x = 120  # 每次走的x,y方向上的距离
        self.limit_distance_y = 90

        self.obstacle_size = [120, 90]  # 障碍物的大小，共设有2个障碍物
        self.obstacle1_x = []
        self.obstacle1_y = []
        self.obstacle2_x = []
        self.obstacle2_y = []

        for i in range(8):  # 障碍物由8个小障碍物构成
            # 第1个障碍物
            self.obstacle1_x.append(360)
            if i <= 3:
                self.obstacle1_y.append(90 * i)
            else:
                self.obstacle1_y.append(90 * (i + 2))
            # 第2个障碍物
            self.obstacle2_x.append(720)
            if i <= 4:
                self.obstacle2_y.append(90 * i)
            else:
                self.obstacle2_y.append(90 * (i + 2))

        self.bird_male_init_position = [0, 0]  # 初始位置
        self.bird_male_position = [0, 0]  # 当前位置
        self.bird_female_init_position = [1080, 0]  # 终点位置

    def collide(self, state_position):
        # 碰撞检测函数
        flag = 1  # 碰撞标志
        flag1 = 1
        flag2 = 1

        # 判断第1个障碍物
        dx = []
        dy = []
        for i in range(8):
            dx1 = abs(self.obstacle1_x[i] - state_position[0])
            dx.append(dx1)
            dy1 = abs(self.obstacle1_y[i] - state_position[1])
            dy.append(dy1)
        mindx = min(dx)
        mindy = min(dy)
        if mindx >= self.limit_distance_x or mindy >= self.limit_distance_y:
            flag1 = 0

        # 判断第2个障碍物
        second_dx = []
        second_dy = []
        for i in range(8):
            dx2 = abs(self.obstacle2_x[i] - state_position[0])
            second_dx.append(dx2)
            dy2 = abs(self.obstacle2_y[i] - state_position[1])
            second_dy.append(dy2)
        mindx = min(second_dx)
        mindy = min(second_dy)
        if mindx >= self.limit_distance_x or mindy >= self.limit_distance_y:
            flag2 = 0

        if flag1 == 0 and flag2 == 0:
            flag = 0
        # 判断是否与边界碰撞
        if state_position[0] > 1080 or state_position[0] < 0 or state_position[1] > 810 or state_position[1] < 0:
            flag = 1
        return flag

    def find(self, state_position):
        # 判断是否达到指定点
        flag = 0
        if abs(state_position[0] - self.bird_female_init_position[0]) < self.limit_distance_x \
                and abs(state_position[1] - self.bird_female_init_position[1]) < self.limit_distance_y:
            flag = 1
        return flag

    def state_to_position(self, state):
        # 从状态到像素坐标变换函数
        i = int(state / 10)
        j = state % 10
        position = [0, 0]
        position[0] = 120 * j
        position[1] = 90 * i
        return position

    def position_to_state(self, position):
        # 从像素坐标变换到状态
        i = position[0] / 120
        j = position[1] / 90
        return int(i + 10 * j)

    def reset(self):
        # 环境重置，随机产生初始状态
        flag1 = 1
        flag2 = 1
        while flag1 or flag2 == 1:
            state = self.states[int(random.random() * len(self.states))]
            state_position = self.state_to_position(state)
            flag1 = self.collide(state_position)  # 初始位置不能在障碍物处
            flag2 = self.find(state_position)  # 初始位置不能在终点处
        return state

    def transform(self, state, action):
        # 状态转移函数，回报函数设置为如达到终点，立即回报为1；若与障碍物碰撞，立即回报为-1；其余情况设为0
        # 对于蒙特卡洛算法，由于其对稀疏回报问题的估计方差无穷大，所以在每一步就要给出回报，变成稠密回报
        # 回报函数设置为如达到终点，立即回报为10;若与障碍物碰撞，立即回报为-10；其余情况设为-2
        current_position = self.state_to_position(state)
        next_position = [0, 0]
        flag_collide = 0
        flag_find = 0

        # 判断当前位置是否与障碍物碰撞
        flag_collide = self.collide(current_position)

        # 判断是否达到终点
        flag_find = self.find(current_position)

        # if flag_collide == 1 or flag_find == 1:
        #     return state,0,True
        if flag_collide == 1:
            return state, -10, True
        if flag_find == 1:
            return state, 10, True

        # 状态转移
        if action == 'e':
            next_position[0] = current_position[0] + 120
            next_position[1] = current_position[1]
        if action == 's':
            next_position[0] = current_position[0]
            next_position[1] = current_position[1] + 90
        if action == 'w':
            next_position[0] = current_position[0] - 120
            next_position[1] = current_position[1]
        if action == 'n':
            next_position[0] = current_position[0]
            next_position[1] = current_position[1] - 90

        # 判断next_state是否与障碍物碰撞
        flag_collide = self.collide(next_position)
        if flag_collide == 1:
            return self.position_to_state(current_position), -10, True

        # 判断next_state是否与达到终点
        flag_find = self.find(next_position)
        if flag_find == 1:
            return self.position_to_state(next_position), 10, True

        # 其余情况
        return self.position_to_state(next_position), -0.1, False

    def gameover(self):
        # 游戏结束
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

    def render(self):
        # 渲染游戏
        if self.viewer is None:
            pygame.init()  # 新建窗口

            self.viewer = pygame.display.set_mode(self.screen_size, 0, 32)
            pygame.display.set_caption('YuanyangEnv')

            self.bird_male = load_bird_male()
            self.bird_female = load_bird_female()
            self.background = load_background()
            self.obstacle = load_obstacle()

            self.viewer.blit(self.bird_female, self.bird_female_init_position)
            self.viewer.blit(self.background, (0, 0))
            self.font = pygame.font.SysFont('times', 15)

        # 画直线
        for i in range(11):
            pygame.draw.lines(self.viewer, (255, 255, 255), True, ((120 * i, 0), (120 * i, 900)), 1)
            pygame.draw.lines(self.viewer, (255, 255, 255), True, ((0, 90 * i), (1200, 90 * i)), 1)
        self.viewer.blit(self.bird_female, self.bird_female_init_position)

        # 画障碍物
        for i in range(8):
            self.viewer.blit(self.obstacle, (self.obstacle1_x[i], self.obstacle1_y[i]))
            self.viewer.blit(self.obstacle, (self.obstacle2_x[i], self.obstacle2_y[i]))

        # 画小鸟
        self.viewer.blit(self.bird_male, self.bird_male_position)

        # 画值函数
        for i in range(10):
            for j in range(10):
                surface = self.font.render(str(round(float(self.value[i, j]), 3)), True, (0, 0, 0))
                self.viewer.blit(surface, (120 * i + 5, 90 * j + 70))

        # 画行为值函数
        for i in range(100):
            y = int(i / 10)
            x = i % 10
            # 东方向的值函数
            surface = self.font.render(str(round(float(self.action_value[i, 0]), 2)), True, (0, 0, 0))
            self.viewer.blit(surface, (120 * x + 80, 90 * y + 45))
            # 南方向的值函数
            surface = self.font.render(str(round(float(self.action_value[i, 1]), 2)), True, (0, 0, 0))
            self.viewer.blit(surface, (120 * x + 50, 90 * y + 70))
            # 西方向的值函数
            surface = self.font.render(str(round(float(self.action_value[i, 2]), 2)), True, (0, 0, 0))
            self.viewer.blit(surface, (120 * x + 10, 90 * y + 45))
            # 北方向的值函数
            surface = self.font.render(str(round(float(self.action_value[i, 3]), 2)), True, (0, 0, 0))
            self.viewer.blit(surface, (120 * x + 50, 90 * y + 10))

        # 画路径点
        for i in range(len(self.path)):
            rec_position = self.state_to_position(self.path[i])
            pygame.draw.rect(self.viewer, [255, 0, 0], [rec_position[0], rec_position[1], 120, 90], 3)
            surface = self.font.render(str(i), True, (255, 0, 0))
            self.viewer.blit(surface, (rec_position[0] + 5, rec_position[1] + 5))

        pygame.display.update()
        self.gameover()
        self.FPSCLOCK.tick(30)

# yy = YuanYangEnv()
# yy.render()
# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             exit()