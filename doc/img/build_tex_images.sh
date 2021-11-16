#!/bin/bash
set -e

# This script isn't very safe or nice but I am lazy

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPT_DIR/tex

# Run pdflatex on all tex source files
for i in *tex; do pdflatex $i;done

# use imgmagick to convert pdfs to pngs
for i in cosphi.pdf sinphi.pdf thetaminus.pdf thetaplus.pdf; do convert -density 300 -quality 90 $i "${i%.*}".png;done
for i in mminus.pdf mplus.pdf; do convert -density 300 -quality 90 -resize 273x48 $i "${i%.*}".png;done

cd -
