#!/bin/bash

set -eu

isort --profile black -l 100 .
black -l 100 .
