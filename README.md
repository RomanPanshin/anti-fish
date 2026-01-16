# anti-fish — Antichess DDPG Bot (with Lichess integration)

A reinforcement-learning bot for **Antichess** (aka *Giveaway chess*), trained with a **DDPG-style actor–critic** pipeline and optionally runnable as a **Lichess bot** via the Lichess API.

> Antichess rules reminder: captures are mandatory; you win by getting rid of all your pieces (or being stalemated, depending on ruleset).

---

## Features

- **Antichess environment**: legal move generation + mandatory captures.
- **DDPG-style training loop** (actor–critic) for policy learning.
- **Self-play / evaluation** utilities (depending on your training scripts).
- **Lichess connector**: play live games on Lichess using an API token (bot account).

---

## Repository structure

- `chess/` — game / rules / move generation (Antichess-specific logic).
- `bspa/` — RL agent + training code (DDPG-style pipeline).
- `TODO:` add/update this section if you have a dedicated module for Lichess (e.g. `lichess/`, `bot/`, `client/`).

---

## Setup

### Requirements
- Python **3.10+** (recommended)
- `pip` / virtualenv

### Install
```bash
git clone https://github.com/RomanPanshin/anti-fish.git
cd anti-fish

python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

pip install -U pip
pip install -r requirements.txt   # TODO: create requirements.txt if missing
