# -*- coding: utf-8 -*-
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import collections
import random
import os

# --- 超参数定义 ---
# 使得调整参数变得容易
# 训练相关
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择 GPU 或 CPU
REPLAY_BUFFER_SIZE = 20000  # 经验回放缓冲区的大小
LEARNING_RATE_ACTOR = 3e-4  # 演员网络的学习率
LEARNING_RATE_CRITIC = 3e-4  # 评论家网络的学习率
LEARNING_RATE_ALPHA = 3e-4  # Alpha (温度参数) 的学习率
BATCH_SIZE = 256  # 每次训练时从缓冲区采样的样本数量
GAMMA = 0.99  # 折扣因子
TAU = 0.005  # 目标网络软更新的系数
ALPHA = 0.2  # 固定的温度参数（如果TARGET_ENTROPY=None）
TARGET_ENTROPY = -3.0  # 目标熵，用于自动调整Alpha。设为-action_dim是常见做法。

# 网络结构
HIDDEN_DIM = 256  # 神经网络隐藏层的维度

class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """向缓冲区添加一条经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """从缓冲区中随机采样一个批次的经验"""
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """
    演员网络 (策略网络)
    输入: 状态
    输出: 动作的概率分布 (均值和标准差)
    """

    def __init__(self, state_dim, action_dim, action_range):
        super(Actor, self).__init__()
        self.action_range = torch.FloatTensor(action_range).to(DEVICE)

        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(HIDDEN_DIM, action_dim)
        self.log_std_layer = nn.Linear(HIDDEN_DIM, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)

        # 限制 log_std 的范围以保证训练稳定
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)

        return mean, std

    def sample(self, state):
        """根据策略采样动作，并计算其对数概率"""
        mean, std = self.forward(state)
        normal = Normal(mean, std)

        # 使用重参数化技巧 (rsample)，使得梯度可以回传
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)  # 将动作压缩到 [-1, 1] 范围

        # 将动作缩放到实际范围
        action = y_t * self.action_range

        # 计算对数概率，需要考虑 tanh 变换带来的影响
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_range * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    """
    评论家网络 (Q值网络)
    输入: 状态和动作
    输出: Q值
    SAC使用两个Critic网络 (Twin Critic) 来缓解Q值过高估计问题。
    """

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # 第一个Q网络
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )
        # 第二个Q网络
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)  # 将状态和动作拼接
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2


class RLAgent:
    def __init__(self, action_space_limits: dict, state_feature_names: list, model_path: str = './rl_model'):
        """
        初始化RL智能体
        Args:
            action_space_limits (dict): 动作空间的限制, e.g., {'demand_corr': (-0.2, 0.2), ...}
            state_feature_names (list): 用于构成状态向量的特征名称列表。
            model_path (str): 模型保存和加载的路径。
        """
        self.action_limits_dict = action_space_limits
        self.state_feature_names = state_feature_names
        self.model_path = model_path

        # 从字典中提取动作维度和范围
        self.action_dim = len(action_space_limits)
        self.action_range = np.array([limits[1] for limits in action_space_limits.values()])
        self.action_keys = list(action_space_limits.keys())

        # 状态维度
        self.state_dim = len(state_feature_names)

        print(f"RL Agent initialized on {DEVICE}")
        print(f"State dimension: {self.state_dim}, Action dimension: {self.action_dim}")

        # 初始化网络和优化器
        self.actor = Actor(self.state_dim, self.action_dim, self.action_range).to(DEVICE)
        self.critic = Critic(self.state_dim, self.action_dim).to(DEVICE)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(DEVICE)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE_CRITIC)

        # 复制初始权重到目标网络
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 自动调整温度参数 Alpha
        if TARGET_ENTROPY is not None:
            self.target_entropy = TARGET_ENTROPY
            self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LEARNING_RATE_ALPHA)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = ALPHA

        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # 确保模型保存目录存在
        os.makedirs(self.model_path, exist_ok=True)

    def _preprocess_state(self, state_dict: dict) -> np.ndarray:
        """将字典格式的状态转换为归一化的numpy数组"""
        state_vector = []
        for feature_name in self.state_feature_names:
            value = state_dict.get(feature_name)
            if value is None:
                # 如果数据缺失，使用0填充。在实际应用中，可能需要更复杂的插值或处理。
                value = 0.0
                print(f"警告: 状态特征 '{feature_name}' 在数据中缺失, 使用0填充。")
            state_vector.append(value)

        # 在这里可以添加状态归一化逻辑 (非常重要)
        # 示例: state_vector = (state_vector - mean) / std
        return np.array(state_vector, dtype=np.float32)

    def choose_action(self, state_dict: dict, deterministic: bool = False) -> dict:
        """
        根据当前状态选择一个动作
        Args:
            state_dict (dict): 从主程序接收的字典格式的状态
            deterministic (bool): 是否采用确定性策略 (评估时使用)
        Returns:
            dict: 包含具体动作值的字典
        """
        state = self._preprocess_state(state_dict)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        self.actor.eval()  # 设置为评估模式
        with torch.no_grad():
            if deterministic:
                # 评估时直接使用均值作为动作，更稳定
                mean, _ = self.actor(state_tensor)
                action_tensor = torch.tanh(mean) * self.actor.action_range
            else:
                # 训练时从分布中采样
                action_tensor, _ = self.actor.sample(state_tensor)
        self.actor.train()  # 恢复为训练模式

        action_np = action_tensor.cpu().numpy().flatten()

        # 将numpy数组转换回字典格式
        action_dict = {key: float(value) for key, value in zip(self.action_keys, action_np)}
        return action_dict

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        """将经验元组添加到回放缓冲区"""
        state_np = self._preprocess_state(state)
        next_state_np = self._preprocess_state(next_state)
        # 将动作字典转换为numpy数组
        action_np = np.array([action[key] for key in self.action_keys], dtype=np.float32)

        self.replay_buffer.add(state_np, action_np, reward, next_state_np, done)

    def train(self):
        """训练网络"""
        if len(self.replay_buffer) < BATCH_SIZE:
            return  # 缓冲区中的样本数不足，不进行训练

        # 从缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        # 转换为Tensor
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.FloatTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        # --- 更新 Critic 网络 ---
        with torch.no_grad():
            # 计算目标Q值
            next_actions, next_log_prob = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target_min = torch.min(q1_target, q2_target) - self.alpha * next_log_prob
            target_q = rewards + (1 - dones) * GAMMA * q_target_min

        # 计算当前Q值
        current_q1, current_q2 = self.critic(states, actions)

        # 计算Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # 优化Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- 更新 Actor 网络 ---
        # 演员网络的更新频率可以低于评论家网络，但这里我们设为相同
        new_actions, log_prob = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new_min = torch.min(q1_new, q2_new)

        # 计算Actor损失
        actor_loss = (self.alpha * log_prob - q_new_min).mean()

        # 优化Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- 更新 Alpha (温度) ---
        if TARGET_ENTROPY is not None:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # --- 软更新目标网络 ---
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

    def save_model(self, path_prefix):
        """
        根据给定的路径前缀保存模型。
        例如, path_prefix = './models/best'
        会保存为 './models/best_actor.pth' 和 './models/best_critic.pth'
        """
        try:
            actor_path = f"{path_prefix}_actor.pth"
            critic_path = f"{path_prefix}_critic.pth"
            torch.save(self.actor.state_dict(), actor_path)
            torch.save(self.critic.state_dict(), critic_path)
            # print(f"模型已保存到 {path_prefix}_*.pth") # 可以在主循环中打印信息
        except Exception as e:
            print(f"保存模型到 {path_prefix} 时发生错误: {e}")
            # --- 新增这部分来打印完整的Traceback ---
            print("--- 以下是完整的错误追溯信息 ---")
            traceback.print_exc()
            print("---------------------------------")
            # -----------------------------------------

    def load_model(self, path_prefix):
        """
        根据给定的路径前缀加载模型。
        返回一个布尔值，表示是否加载成功。
        """
        actor_path = f"{path_prefix}_actor.pth"
        critic_path = f"{path_prefix}_critic.pth"

        if not os.path.exists(actor_path) or not os.path.exists(critic_path):
            print(f"错误: 找不到模型文件 {path_prefix}_*.pth。将使用新网络。")
            return False

        try:
            self.actor.load_state_dict(torch.load(actor_path, map_location=DEVICE))
            self.critic.load_state_dict(torch.load(critic_path, map_location=DEVICE))
            self.critic_target.load_state_dict(self.critic.state_dict())
            print(f"成功从 {path_prefix}_*.pth 加载模型。")
            return True
        except Exception as e:
            print(f"加载模型 {path_prefix}_*.pth 时发生错误: {e}")
            # --- 新增这部分来打印完整的Traceback ---
            print("--- 以下是完整的错误追溯信息 ---")
            traceback.print_exc()
            print("---------------------------------")
            # -----------------------------------------
            return False


