#!/bin/bash

set -eu

poetry run flake8 --ignore=E501,W503 tests src --verbose