#Code van Pong omgevormd om cartpole te kunnen spelen!
#helemaal zelf gedaan!
#nog eens laten lopen om te zien als het werkt??
import gym
import numpy as np

def update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
    """ See here: http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop"""
    epsilon = 1e-5
    for layer_name in weights.keys():
        g = g_dict[layer_name]
        expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g ** 2
        weights[layer_name] += (learning_rate * g) / (np.sqrt(expectation_g_squared[layer_name] + epsilon))
        g_dict[layer_name] = np.zeros_like(weights[layer_name])  # reset batch gradient buffer

def choose_action(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        # signifies left in openai gym
        return 0
    else:
        # signifies right in openai gym
        return 1

def preprocess_observations(inputobservation, prev_processed_obs):
    #geef enkel het verschil terug tussen vorige frame en dit frame
    tmp_obs = inputobservation
    if prev_processed_obs is not None:
        inputobservation = inputobservation - prev_processed_obs
    else:
        inputobservation = np.zeros(4)
        #print("in else: ", inputobservation)
    prev_processed_obs = tmp_obs
    return inputobservation, prev_processed_obs

def relu(vector):
    vector[vector < 0] = 0
    return vector

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def apply_neural_nets(observation_matrix, weights):
    #print(observation_matrix)
    #print("observatie:" , observation_matrix)
    """ Based on the observation_matrix and weights, compute the new hidden layer values and the new output layer values"""
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values) #om uiteindelijk links of rechts te hebben!
    return hidden_layer_values, output_layer_values

def discount_rewards(rewards, gamma):
    """ Actions you took 20 steps before the end result are less important to the overall result than an action you took a step ago.
    This implements that logic by discounting the reward on previous actions based on how long ago they were taken"""
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

def discount_with_rewards(gradient_log_p, episode_rewards, gamma):
    """ discount the gradient with the normalized rewards """
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return gradient_log_p * discounted_episode_rewards

def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
    """ See here: http://neuralnetworksanddeeplearning.com/chap2.html"""
    delta_L = gradient_log_p
    dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()
    delta_l2 = np.outer(delta_L, weights['2'])
    delta_l2 = relu(delta_l2)
    dC_dw1 = np.dot(delta_l2.T, observation_values)
    return {
        '1': dC_dw1,
        '2': dC_dw2
    }

def main():
    env = gym.make('CartPole-v0')
    observation = env.reset()  # This gets us the image

    #print(env.action_space)
    #0 = left
    # 1 = right
    # hyperparameters
    episode_number = 0
    batch_size = 10 #10 keer spelen tegen dat we het effectief updaten
    gamma = 0.95 # discount factor for reward
    decay_rate = 0.99
    num_hidden_layer_neurons = 20
    learning_rate = 1e-4

    episode_number = 0
    reward_sum = 0
    running_reward = None
    prev_processed_observations = None

    weights = {
        '1': np.random.randn(num_hidden_layer_neurons, 4) / np.sqrt(4),
        '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
    }

    #print(weights)

    # To be used with rmsprop algorithm (http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop)
    expectation_g_squared = {}
    g_dict = {}
    for layer_name in weights.keys():
        expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
        g_dict[layer_name] = np.zeros_like(weights[layer_name])

    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []

    while True:
        env.render()
        #processed is het verschil tussen de vorige en de huidige, de prev is de volledige observatie die nu gebeurd is eigenlijjk
        processed_observations, prev_processed_observations = preprocess_observations(observation, prev_processed_observations)
        hidden_layer_values, left_probability = apply_neural_nets(processed_observations, weights)

        episode_observations.append(processed_observations)
        episode_hidden_layer_values.append(hidden_layer_values)

        action = choose_action(left_probability)
        # carry out the chosen action
        observation, reward, done, info = env.step(action)
        #print(reward)
        reward_sum += reward
        episode_rewards.append(reward)

        fake_label = 1 if action == 0 else 0 #1 als links
        loss_function_gradient = fake_label - left_probability
        episode_gradient_log_ps.append(loss_function_gradient)


        if done:  # an episode finished
            print("last observation:" , observation, "in episode:" , episode_number)
            episode_number += 1

            # Combine the following values for the episode
            episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
            episode_observations = np.vstack(episode_observations)
            episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            episode_rewards = np.vstack(episode_rewards)

            # Tweak the gradient of the log_ps based on the discounted rewards
            episode_gradient_log_ps_discounted = discount_with_rewards(episode_gradient_log_ps, episode_rewards, gamma)

            gradient = compute_gradient(
                episode_gradient_log_ps_discounted,
                episode_hidden_layer_values,
                episode_observations,
                weights
            )

            # Sum the gradient for use when we hit the batch size
            for layer_name in gradient:
                g_dict[layer_name] += gradient[layer_name]

            if episode_number % batch_size == 0:
                update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)

            episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []  # reset values
            observation = env.reset()  # reset env
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            reward_sum = 0
            prev_processed_observations = None

main()

