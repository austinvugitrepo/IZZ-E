#!/bin/sh

websockify --daemon --web=/usr/share/novnc/ 8080 localhost:5901 
