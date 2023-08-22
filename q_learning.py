import numpy as np
import pandas as pd

class QLearningTable():
    def __init__(self, env, State, Location, actions=list(range(4)), learning_rate=0.01, reward_decay=0.9, e_greedy=0.95):
        self.env = env
        self.State = State
        self.Location = Location
        self.agent_dict = env.agent_dict
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q_table_final = pd.DataFrame(columns=self.actions, dtype=np.float64)
    
    def learn(self, state, action, reward, next_state, done):
        # Checking if the next step exists in the Q-table
        self.check_state_exist(next_state)

        # Current state in the current position
        q_predict = self.q_table.loc[state, action]

        # Checking if the next state is free or it is obstacle or goal
        if not done:
            q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()
        else:
            q_target = reward

        # Updating Q-table with new knowledge
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

        return self.q_table.loc[state, action]

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # Choosing random action - left 10 % for choosing randomly
            action = np.random.choice(self.actions)
        return action
    
    def check_state_exist(self, state):
        #if state not in self.q_table.index:
        #    row = pd.Series([0]*len(self.actions),index=self.q_table.columns)
        #    self.q_table = pd.concat(self.q_table, pd.DataFrame(data= row, index=str(state), dtype=np.float64))
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                ))

    def search(self, agent_name):

        initial_state = self.agent_dict[agent_name]["start"]
        
        steps = []
        all_costs = []
        shortest_route = [0]
        temp_route_1 = list(range(self.env.dimension[0]*self.env.dimension[1]))
        for episode in range(1000):
            observation = [initial_state.location.x, initial_state.location.y]
            cost = 0
            temp_route_2 = [observation]
            
            while True:
                
                action = self.choose_action(str(observation)) 
                observation_, reward, done, end_route = self.env.step(agent_name, observation, action)
                
                cost += self.learn(str(observation), action, reward, str(observation_), done)
                observation = observation_
                temp_route_2.append(observation)
                
                if end_route and len(temp_route_2) < len(temp_route_1):
                    temp_route_1 = temp_route_2
                    shortest_route = temp_route_2
                if done:
                    
                    all_costs += [cost]
                    break
                    
            # print("Episode: ", episode,"// Current_shortest_route: ", len(shortest_route), "// Is_reach_goal: ",bool(end_route) )
        #print("The q table is: ",self.q_table)
        # print("----------------------------SUMARY---------------------------------")
        if shortest_route !=  [0]:
            # print("The shortest route is : ", shortest_route)
            total_path = [self.State(0, self.Location(0,0))]
            
            for i in range(len(shortest_route)):
                total_path.append(self.State(i, self.Location(shortest_route[i][0], shortest_route[i][1])))
            return total_path
        else:
            print("There is no solution")
            return False

