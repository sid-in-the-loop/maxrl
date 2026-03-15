from argparse import Action
import numpy as np
import random
from collections import deque
import json
import os


class Maze_Env:
    def __init__(self, start):
        self.state = start
        self.action = 0
        self.reward = 0
        self.done = False
        self.info = {}

    def step(self, action):
        pass


class MazeGenerator:
    def __init__(self, size=7, seed=None, algorithm='prim'):
        self.algorithm = algorithm
        assert algorithm in ['prim', 'dfs'], "Invalid algorithm"
        self.size = size
        self.rng = random.Random(seed)
        self.grid = np.ones((size, size), dtype=int)
        self.start = (1, 1)
        self.goal = (size - 2, size - 2)
        self.actions = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }
        self.action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def generate(self, algorithm=None):
        self.grid.fill(1)
        if algorithm is None:
            algorithm = self.algorithm
        if self.algorithm == 'dfs':
            self.carve_passages_from(self.start[0], self.start[1])
        elif self.algorithm == 'prim':
            self.prim()
        self.grid[self.start] = 0
        self.grid[self.goal] = 0
        return self.grid

    def prim(self):
        start_x, start_y = self.start[0], self.start[1]
        self.grid[start_x, start_y] = 0
        frontier = []
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        for dx, dy in directions:
            nx, ny = start_x + dx, start_y + dy
            if 0 < nx < self.size - 1 and 0 < ny < self.size - 1:
                if self.grid[nx, ny] == 1:
                    frontier.append((nx, ny))
        while frontier:
            idx = self.rng.randint(0, len(frontier) - 1)
            fx, fy = frontier.pop(idx)
            neighbors = []
            for dx, dy in directions:
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[nx, ny] == 0:
                        neighbors.append((nx, ny))
            if neighbors:
                nx, ny = self.rng.choice(neighbors)
                mid_x, mid_y = (fx + nx) // 2, (fy + ny) // 2
                self.grid[mid_x, mid_y] = 0
                self.grid[fx, fy] = 0
                for dx, dy in directions:
                    new_x, new_y = fx + dx, fy + dy
                    if 0 < new_x < self.size - 1 and 0 < new_y < self.size - 1:
                        if self.grid[new_x, new_y] == 1 and (new_x, new_y) not in frontier:
                            frontier.append((new_x, new_y))

    def carve_passages_from(self, cx, cy):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.rng.shuffle(directions)
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            mx, my = cx + 2*dx, cy + 2*dy
            if 0 <= mx < self.size and 0 <= my < self.size:
                if self.grid[mx, my] == 1:
                    self.grid[nx, ny] = 0
                    self.grid[mx, my] = 0
                    self.carve_passages_from(mx, my)

    def get_state_action_pairs(self):
        optimal_actions = self.solve_bfs()
        state_action_pairs = {}
        cx, cy = self.start[0], self.start[1]
        for i in range(len(optimal_actions)):
            action = optimal_actions[i]
            state_action_pairs[(cx, cy)] = self.action_names[action]
            dx, dy = self.actions[action]
            cx, cy = cx + dx, cy + dy
        return state_action_pairs

    def format_data(self):
        walls = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == 1:
                    walls.append(f"({r},{c})")
        env = {
            "Walls": {', '.join(walls)},
            "Start": self.start,
            "Goal": self.goal,
        }
        policy = self.get_state_action_pairs()
        episode = {
            "env": env,
            "policy": policy,
            "optimal_len": len(policy.keys()),
        }
        return episode

    def solve_bfs(self):
        queue = deque([(self.start, [])])
        visited = set([self.start])
        while queue:
            (cx, cy), path = queue.popleft()
            if (cx, cy) == self.goal:
                return path
            for action_idx, (dx, dy) in self.actions.items():
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[nx, ny] == 0 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        new_path = path + [action_idx]
                        queue.append(((nx, ny), new_path))
        return None

    def to_prompt_string(self):
        walls = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == 1:
                    walls.append(f"({r},{c})")
        prompt = (
            f"Walls: {', '.join(walls)}\n"
            f"Start: {self.start}\n"
            f"Goal: {self.goal}\n"
            f"State: (0,0)\n"
            f"Action: "
        )
        return prompt

    def render_ascii(self):
        chars = {1: '#', 0: ' ', 2: 'S', 3: 'G'}
        temp_grid = self.grid.copy()
        temp_grid[self.start] = 2
        temp_grid[self.goal] = 3
        print("-" * (self.size + 2))
        for row in temp_grid:
            print("|" + "".join([chars[x] for x in row]) + "|")
        print("-" * (self.size + 2))

    def to_text_sequence(self):
        optimal_actions = self.solve_bfs()
        if optimal_actions is None:
            return None
        grid_tokens = []
        for r in range(self.size):
            for c in range(self.size):
                pos = (r, c)
                if pos == self.start:
                    grid_tokens.append("START")
                elif pos == self.goal:
                    grid_tokens.append("GOAL")
                elif self.grid[r, c] == 1:
                    grid_tokens.append("WALL")
                else:
                    grid_tokens.append("PATH")
            grid_tokens.append("NEWLINE")
        action_tokens = [self.action_names[a] for a in optimal_actions]
        sequence = ["<bos>", "GRID_START"] + grid_tokens + ["GRID_END", "PATH_START"] + action_tokens + ["DONE", "<eos>"]
        text_sequence = " ".join(sequence)
        return {
            "sequence": text_sequence,
            "optimal_path_length": len(optimal_actions)
        }


class DatasetGenerator:
    def __init__(self, size=7, seed=42, algorithm='prim', num_episodes=100, filename='dataset.json', save_dir='./data'):
        self.maze_generator = MazeGenerator(size=size, seed=seed, algorithm=algorithm)
        self.num_episodes = num_episodes
        self.episodes = []
        self.filename = filename
        self.save_dir = save_dir
        self.metadata = {
            "size": size,
            "seed": seed,
            "algorithm": algorithm,
            "num_episodes": num_episodes,
        }

    def generate(self):
        for i in range(self.num_episodes):
            self.episodes.append(self.maze_generator.format_data())
        return self.episodes, self.metadata

    def save(self, filename=None):
        if filename is None:
            filename = os.path.join(self.save_dir, self.filename)
        with open(filename, 'w') as f:
            json.dump(self.episodes, f, indent=4)
        with open(os.path.join(self.save_dir, self.filename.replace('.json', '_metadata.json')), 'w') as f:
            json.dump(self.metadata, f, indent=4)

    def generate_text_training_data(self, test_size=256):
        text_data = []
        for i in range(self.num_episodes):
            self.maze_generator.generate()
            text_item = self.maze_generator.to_text_sequence()
            if text_item is not None:
                text_data.append(text_item)
        split_rng = random.Random(self.metadata.get('seed', 42))
        split_rng.shuffle(text_data)
        test_data = text_data[:test_size]
        train_data = text_data[test_size:]
        os.makedirs(self.save_dir, exist_ok=True)
        train_filename = os.path.join(self.save_dir, 'train.json')
        with open(train_filename, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        test_filename = os.path.join(self.save_dir, 'test.json')
        with open(test_filename, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        metadata = {
            "size": self.maze_generator.size,
            "seed": self.metadata.get('seed'),
            "algorithm": self.maze_generator.algorithm,
            "num_train": len(train_data),
            "num_test": len(test_data),
            "test_size": test_size
        }
        metadata_filename = os.path.join(self.save_dir, 'training_metadata.json')
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Training data generated:")
        print(f"  Train: {len(train_data)}, saved to {train_filename}")
        print(f"  Test: {len(test_data)}, saved to {test_filename}")
        print(f"  Metadata saved to {metadata_filename}")
        return train_data, test_data, metadata


if __name__ == "__main__":
    dataset_gen = DatasetGenerator(
        size=17,
        seed=0,
        algorithm='prim',
        num_episodes=1000000,
        save_dir='./data/17x17_1M'
    )
    train_data, test_data, metadata = dataset_gen.generate_text_training_data(test_size=256)
