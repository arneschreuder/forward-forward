#!/usr/bin/env bash

set -ex

python3 main.py \
	--seed=1 \
	--batch_size=1024 \
	--hidden_layers=4 \
	--hidden_units=2000 \
	--activation="relu" \
	--lr=0.003 \
	--threshold=2.0 \
	--num_epochs=30
