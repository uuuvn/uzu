#!/bin/sh
set -e
cd "$(dirname "$0")"
swiftc -O -whole-module-optimization -o tsbench main.swift
