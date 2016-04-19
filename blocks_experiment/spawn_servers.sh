#!/bin/sh
THEANO_FLAGS='device=cpu' python spawn_dataserver.py --datasource train &
THEANO_FLAGS='device=cpu' python spawn_dataserver.py --datasource valid &

