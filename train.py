import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical    # 提供了一组概率分布的实现，这里用于表示离散类别型分布

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 环境参数
TARGET_RANGE = (4.0, 6.0)
INITIAL_HEIGHT = 10.0
X_BOUNDS = (0.0, 10.0)


# 修改后的环境类（支持固定初始位置）
class DropBlockEnv:
    def __init__(self):
        self.x = None
        self.y = None
        self.reset()

    def reset(self, fixed_x=None):
        self.y = INITIAL_HEIGHT
        if fixed_x is not None:  # 允许指定初始x坐标, 因为GRPO需要对同一个prompt采样多个response, 作为一组
            self.x = np.clip(fixed_x, X_BOUNDS[0], X_BOUNDS[1])
        else:
            self.x = np.random.uniform(X_BOUNDS[0], X_BOUNDS[1])
        return self._get_state()

    def _get_state(self):
        return np.array([self.x / X_BOUNDS[1], self.y / INITIAL_HEIGHT], dtype=np.float32)

    def step(self, action):
        new_x = self.x + (1.0 if action else -1.0)
        new_x = np.clip(new_x, X_BOUNDS[0], X_BOUNDS[1])
        self.y -= 1.0
        self.x = new_x

        done = self.y <= 0
        reward = self._calculate_reward(done)
        return self._get_state(), reward, done, {}

    def _calculate_reward(self, done):
        if done:
            return 10.0 if TARGET_RANGE[0] <= self.x <= TARGET_RANGE[1] else -10.0

        distance = abs(self.x - np.mean(TARGET_RANGE))
        in_target = TARGET_RANGE[0] <= self.x <= TARGET_RANGE[1]
        return (0.2 if in_target else -0.1) - 0.05 * distance - 0.1


# 策略网络（保持与PPO相同结构）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),    # (2, 128)
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),    # (128, 128)
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),    # (128, 2)
            nn.Softmax(dim=-1))

    def forward(self, x):
        return self.net(x)


# GRPO算法实现
class GRPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, beta=0.01, group_size=4,
                 epochs=4, batch_size=64):
        # print(state_dim): 2     # 状态空间的维度
        # print(action_dim): 2    # 动作空间的维度
        # print(lr): 0.0003    # Actor网络学习率
        # print(gamma): 0.99    # 累积折扣回报
        # print(epsilon): 0.2    # clip函数限制梯度更新幅度
        # print(beta): 0.01    # kl散度系数
        # print(group_size): 4    # 将初始状态相同的视为同一个组, 同一个组采样4条轨迹
        # print(epochs): 4     # 一批旧的轨迹样本更新4次参数
        # print(batch_size): 64    # 每次采样64个状态动作对

        # 策略模型
        self.actor = Actor(state_dim, action_dim).to(device)    # (2, 2)
        self.actor_old = Actor(state_dim, action_dim).to(device)    # (2, 2)
        self.actor_old.load_state_dict(self.actor.state_dict())    # 参考模型ref_model, 防止更新后的actor模型偏离ref_model过多

        # 优化器
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # 超参数
        self.gamma = gamma    # 0.99
        self.epsilon = epsilon    # 0.2
        self.beta = beta    # 0.01
        self.group_size = group_size  # 每组轨迹数    # 4
        self.epochs = epochs    # 4
        self.batch_size = batch_size    # 64

    def update(self, groups):
        # print(len(groups)): 10
        # print(len(groups[0])): 4
        # print(set(groups[0][0])): {'states', 'actions', 'old_probs', 'rewards'}

        """ 处理组数据进行策略更新
        groups: 包含多个组的列表，每个组包含group_size个轨迹
        """
        # 数据容器
        all_states, all_actions = [], []
        all_old_probs, all_advantages = [], []

        # 处理每个组
        for group in groups:
            # print(len(group)): 4
            # group里面包含4条初始状态相同的轨迹

            # 步骤1: 对齐组内轨迹的奖励, 对于长度较短的轨迹, 在rewards后面添零;
            max_length = max(len(ep["rewards"]) for ep in group)
            reward_matrix = []
            for ep in group:
                padded_rewards = ep["rewards"] + [0] * (max_length - len(ep["rewards"]))
                reward_matrix.append(padded_rewards)
            reward_matrix = np.array(reward_matrix)
            # print(reward_matrix.shape): (4, 10)
            # print(reward_matrix):
            # [[ -0.375  -0.325  -0.275   0.074  -0.275  0.074  -0.275   0.074  -0.275  10.        ]
            #  [ -0.375  -0.425  -0.45    -0.4    -0.35  -0.4   -0.35    -0.4   -0.45   -10.       ]
            #  [ -0.45   -0.45   -0.4     -0.45   -0.4   -0.35   -0.4    -0.35   -0.4   -10.       ]
            #  [ -0.375  -0.425  -0.375   -0.325  -0.275 -0.325  -0.375  -0.425  -0.375 -10.        ]]

            # 步骤2: 时间步级别的奖励归一化
            """ 关于GRPO算法，每个组是否要在每个时间步上做各自的归一化，还是整个组一起做归一化，不太确定. """
            normalized_rewards = np.zeros_like(reward_matrix)
            for t in range(max_length):
                step_rewards = reward_matrix[:, t]    # 取出每个时间步下, 4条轨迹对应位置的奖励
                # print(step_rewards): [-0.37535715 -0.37535715 -0.45       -0.37535715]
                if np.all(step_rewards == 0):  # 跳过填充的零
                    continue
                mu = np.mean(step_rewards)
                sigma = np.std(step_rewards) + 1e-8
                # print(mu): -0.3940178649037185
                # print(sigma): 0.032321310767648104
                normalized_rewards[:, t] = (step_rewards - mu) / sigma
                # print(normalized_rewards[:, t]): [ 0.57735009  0.57735009 -1.73205027  0.57735009]

            # 步骤3：计算每个轨迹的优势值
            for ep_idx, episode in enumerate(group):
                actual_length = len(episode["rewards"])
                returns = []
                discounted_return = 0

                # 反向计算折扣回报
                for t in reversed(range(actual_length)):
                    r = normalized_rewards[ep_idx, t]
                    discounted_return = r + self.gamma * discounted_return
                    returns.insert(0, discounted_return)

                # 处理填充部分
                returns += [0] * (max_length - actual_length)
                # print(returns):
                # [14.109296432906557, 13.668632669035645, 12.096134430425321, 10.63288888518053, 9.031466659131542,
                # 8.170741137621107, 6.521443921141949, 4.968112884503785, 3.2843047316097858, 1.732050805568877]

                # 收集有效数据
                valid_length = min(len(episode["states"]), len(returns))
                all_states.extend(episode["states"][:valid_length])
                all_actions.extend(episode["actions"][:valid_length])
                all_old_probs.extend(episode["old_probs"][:valid_length])
                all_advantages.extend(returns[:valid_length])

        # 转换为Tensor
        states_tensor = torch.FloatTensor(all_states).to(device)
        actions_tensor = torch.LongTensor(all_actions).to(device)
        old_probs_tensor = torch.FloatTensor(all_old_probs).to(device)
        advantages_tensor = torch.FloatTensor(all_advantages).to(device)
        # print(states_tensor.shape): torch.Size([400, 2])
        # print(actions_tensor.shape): torch.Size([400])
        # print(old_probs_tensor.shape): torch.Size([400])
        # print(advantages_tensor.shape): torch.Size([400])

        # 全局标准化优势值
        """  GRPO论文里没有提及到这一步，干脆将其注释掉  """
        # advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        # print(advantages_tensor.shape): torch.Size([400])

        # 策略优化
        for _ in range(self.epochs):    # 4    # 利用同一批轨迹数据, 更新参数4次
            indices = torch.randperm(len(states_tensor))
            for start in range(0, len(states_tensor), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]    # 从中随机挑选64个状态动作对
                # print(len(batch_idx)): torch.Size([64, 2])

                # 获取当前batch数据
                batch_states = states_tensor[batch_idx]
                batch_actions = actions_tensor[batch_idx]
                batch_old_probs = old_probs_tensor[batch_idx]
                batch_advantages = advantages_tensor[batch_idx]
                # print(batch_states.shape): torch.Size([64])
                # print(batch_actions.shape): torch.Size([64])
                # print(batch_old_probs.shape): torch.Size([64])
                # print(batch_advantages.shape): torch.Size([64])

                # 计算新策略概率
                new_probs = self.actor(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze()
                ratio = new_probs / (batch_old_probs + 1e-8)
                # print(new_probs.shape): torch.Size([64])
                # print(ratio.shape): torch.Size([64])

                # 策略损失（PPO裁剪）
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                # print(surr1.shape): torch.Size([64])
                # print(surr2.shape): torch.Size([64])
                # print(policy_loss): tensor(-0.1133, device='cuda:0', grad_fn=<NegBackward0>)

                # KL散度惩罚项
                with torch.no_grad():
                    old_p = self.actor_old(batch_states)
                    # print(old_p.shape): torch.Size([64, 2])
                new_p = self.actor(batch_states)
                # print(new_p.shape): torch.Size([64, 2])

                """ 计算了新概率分布 new_p 与旧概率分布 old_p 之间的 KL 散度，并对整个批次的结果取平均值。 """
                # 最开始的时候，actor_model和ref_model参数相同，KL散度为零
                kl_div = F.kl_div(torch.log(new_p + 1e-10), old_p, reduction='batchmean')
                # print(kl_div): tensor(0., device='cuda:0', grad_fn=<DivBackward0>)

                # 总损失
                total_loss = policy_loss + self.beta * kl_div
                # print(total_loss): tensor(-0.1133, device='cuda:0', grad_fn=<AddBackward0>)

                # 梯度更新
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        # 更新旧策略, 即每个update更新一次ref_model权重
        """ 在GRPO论文中，ref_model加载的是大语言模型SFT后权重，由于这里没有比较好的预训练权重，这里直接每隔一段时间
            更新ref_model一次，同时通过KL散度确保actor_model和ref_model不会偏离太多. """
        self.actor_old.load_state_dict(self.actor.state_dict())


# 数据收集函数（按组收集）
def collect_group_data(env, policy, group_size):
    """ 收集一个组的数据（相同初始状态）, 因为GRPO算法是对一个prompt, 采样生成多个response. """

    # 随机生成初始x坐标
    init_x = np.random.uniform(X_BOUNDS[0], X_BOUNDS[1])
    # print(init_x): 9.50714306409916

    group = []
    for _ in range(group_size):    # 同一个组内包含4条初始状态相同的轨迹
        # 重置到相同初始状态
        state = env.reset(fixed_x=init_x)
        # print(state): [0.9507143 1.       ]

        done = False
        episode = {
            "states": [],
            "actions": [],
            "old_probs": [],
            "rewards": []
        }

        while not done:
            state_tensor = torch.FloatTensor(state).to(device)
            # print(state_tensor): tensor([0.9507, 1.0000], device='cuda:0')
            with torch.no_grad():
                action_probs = policy(state_tensor)
                # print(action_probs): tensor([0.5161, 0.4839], device='cuda:0')
                dist = Categorical(action_probs)
                # print(dist): Categorical(probs: torch.Size([2]))
                action = dist.sample()
                # print(action): tensor(0, device='cuda:0')

            next_state, reward, done, _ = env.step(action.item())
            # print(next_state): [0.8507143 0.9      ]
            # print(reward): -0.375357153204958
            # print(done): False
            # print(_): {}

            # 记录数据
            episode["states"].append(state)
            episode["actions"].append(action.item())
            episode["old_probs"].append(action_probs[action].item())
            episode["rewards"].append(reward)

            state = next_state

        group.append(episode)

    # print(len(group)): 4    # group是一个列表，里面包含4条初始状态相同的轨迹值，存储着'old_probs', 'actions', 'states', 'rewards'等信息
    # print(set(group[0])): {'old_probs', 'actions', 'states', 'rewards'}

    return group


# 评估函数（保持与原始PPO完全相同）
def evaluate(policy, num_episodes=1000):
    env = DropBlockEnv()
    safe_landings = 0

    for _ in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                action_probs = policy.actor(state_tensor)
                action = torch.argmax(action_probs).item()

            next_state, _, done, _ = env.step(action)
            state = next_state

        final_x = state[0] * X_BOUNDS[1]
        if TARGET_RANGE[0] <= final_x <= TARGET_RANGE[1]:
            safe_landings += 1

    return safe_landings / num_episodes


# 训练流程（保持原始结构）
def train():

    env = DropBlockEnv()
    agent = GRPO(state_dim=2, action_dim=2, lr=3e-4, gamma=0.99, epsilon=0.2, beta=0.01,
                 group_size=4, epochs=4, batch_size=100)

    total_updates = 2501    # 总共更新2500次
    groups_per_update = 10  # 每次更新收集10个组, 每组包含4条轨迹
    eval_interval = 10

    # 初始评估
    success_rate = evaluate(agent, 10000)
    print(f"Initial Evaluation Safe Rate: {success_rate:.4f}")

    # 训练循环
    for update in range(total_updates):    # 2501
        # 收集数据
        groups = []
        for _ in range(groups_per_update):    # 10
            groups.append(collect_group_data(env, agent.actor, agent.group_size))
        # groups中含有10个组，每个组包含初始状态相同的4条轨迹
        # print(len(groups)): 10
        # print(len(groups[0])): 4
        # print(set(groups[0][0])): {'rewards', 'actions', 'states', 'old_probs'}

        # 策略更新
        agent.update(groups)

        # 定期评估
        if (update + 1) % eval_interval == 0:
            success_rate = evaluate(agent, 1000)
            print(f"Update {update + 1:4d} | Safe Rate: {success_rate:.4f}")

    # 最终评估
    success_rate = evaluate(agent, 10000)
    print(f"Final Evaluation Safe Rate: {success_rate:.4f}")
    torch.save(agent.actor.state_dict(), "grpo_policy.pth")


if __name__ == "__main__":
    train()

    """
    Initial Evaluation Safe Rate: 0.1484
    Update   10 | Safe Rate: 0.9070
    Update   20 | Safe Rate: 0.9930
    Update   30 | Safe Rate: 0.8910
    Update   40 | Safe Rate: 0.9510
    Update   50 | Safe Rate: 0.9600
    Update   60 | Safe Rate: 0.9900
    Update   70 | Safe Rate: 0.8850
    Update   80 | Safe Rate: 1.0000
    Update   90 | Safe Rate: 0.9970
    Update  100 | Safe Rate: 0.9990
    ......
    Update 2400 | Safe Rate: 0.9520
    Update 2410 | Safe Rate: 0.9810
    Update 2420 | Safe Rate: 0.9860
    Update 2430 | Safe Rate: 0.9950
    Update 2440 | Safe Rate: 0.9680
    Update 2450 | Safe Rate: 0.9790
    Update 2460 | Safe Rate: 0.9810
    Update 2470 | Safe Rate: 1.0000
    Update 2480 | Safe Rate: 0.9740
    Update 2490 | Safe Rate: 0.9830
    Update 2500 | Safe Rate: 0.9760
    Final Evaluation Safe Rate: 0.9777
    """