# ChessGAN project

## About this project
The creation of powerful chess programs in past years has made cheating in the game easy and prevalent. 
This fact has facilitated a need for high quality chess anti-cheats capable of detecting AI play. 
While such algorithms exist, they often focus on better-than-human play as opposed to non-human play. 
In an attempt to address this issue, this project implements ChessGAN.
ChessGAN is a generative adversarial network whose generator learns to play human-like chess,
while its discriminator is trained to distinguish human moves from AI moves.
We find that while such a discriminator was capable of distinguishing its generator, 
this training did not generalize to all human or AI play.
However, we did find that the use of this discriminator aided in the training of a generator which played 
high-quality chess with a move distribution closely resembling that of human players.

## Getting Started

### Prerequisites

This project's dependencies can be installed using conda by:
```sh
conda create --name <env> --file requirements.txt
```
It also uses data from the lichess open-database which can be found [here](https://database.lichess.org/), and the use of an external
engine such as stockfish, whose installation guide can be found [here](https://stockfishchess.org/download/)

### Installation & Use

Install simply by cloning the repo
```sh
git clone https://github.com/OsamaDabb/Chess_anti_cheat.git
```
From there, code involving the training of the baseline model can be found in "behavioural_cloning.ipynb", while 
"GAN architectures.ipynb" contains all work relating to GAN training and evaluation.
