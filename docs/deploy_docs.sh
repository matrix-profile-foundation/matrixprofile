#!/bin/bash

rsync -rav -e ssh --delete build/html/ matrixprofile.docs.matrixprofile.org:/home/tyler/www/matrixprofile.docs.matrixprofile.org/www