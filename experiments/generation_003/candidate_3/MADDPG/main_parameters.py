import argparse

def main_parameters():
    parser = argparse.ArgumentParser()
    ############################################ 选择环境 ############################################
    parser.add_argument("--env_name", type =str, default = "simple_tag_env", help = "name of the env",   
                        choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3', 'simple_tag_env']) 
    parser.add_argument("--render_mode", type=str, default = "None", help = "None | human | rgb_array")
    parser.add_argument("--episode_num", type = int, default = 5000) # 增加训练回合数
    parser.add_argument("--episode_length", type = int, default = 100) # ⚠️ 关键修改：从50增加到100，给智能体足够时间完成围捕
    parser.add_argument('--learn_interval', type=int, default=10,
                        help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=5e3, help='random steps before the agent start to learn')
    parser.add_argument('--tau', type=float, default=0.005, help='soft update parameter')  # 从0.01降到0.005，避免目标网络更新太快
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')  # 从0.95提高到0.98，更重视长期奖励（完成围捕）
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=256, help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.002, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.0002, help='learning rate of critic')
    # The parameters for the communication network
    # TODO
    parser.add_argument('--visdom', type=bool, default=False, help="Open the visdom")
    parser.add_argument('--size_win', type=int, default=200, help="Open the visdom") # 1000 原：200


    args = parser.parse_args()
    return args