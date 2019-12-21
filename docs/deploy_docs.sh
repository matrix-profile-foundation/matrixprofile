#!/bin/bash

rsync -rav -e ssh --delete build/html/ matrixprofile-ts.docs.matrixprofile.org:/home/tyler/www/matrixprofile-ts.docs.matrixprofile.org/www