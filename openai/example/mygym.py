# import gym
# env = gym.make('CartPole-v0')
# env.reset()
# for _ in range(1000): # run for 1000 steps
#     env.render()
#     action = env.action_space.sample() # pick a random action
#     env.step(action) # take action

import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
