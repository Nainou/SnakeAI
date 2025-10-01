PvP Snake AI project in Python where 2–4 genetically trained snakes compete on a large grid.

Classic snake rules: eat food to grow, hitting walls/self/others = death, last snake alive wins.

Each snake uses a neural network trained with a genetic algorithm (PyTorch), evolving strategies like avoiding others or trapping opponents.

Training runs headless (no rendering) for speed, saving models as .pth.

Testing mode loads models, runs PvP matches with Pygame rendering and a visual neural network view for each snake.

Snakes should have distinct colors and can use different saved models.

File structure:


game/ (snake logic, env, rendering)
ai/ (model, genetic algo, utils)
train/train.py (headless training)
test/test.py (model testing & visualization)


Guidelines:

Use modular, object-oriented code.

Keep rendering, logic, and AI separate.

Comment well and follow clean architecture.

Ask clarifying questions if unsure.

Your role: act as an expert AI/game dev. Generate clean, extensible code and explain key design choices.