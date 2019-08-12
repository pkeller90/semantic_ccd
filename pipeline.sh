#!/bin/bash

start_date=$(date '+%Y_%m_%d__%H_%M')
echo "====== Starting at:"
echo "====== $(date)"
echo "====================="
echo ""

echo "====================="
echo "== Starting extraction at:"
echo "== $(date)"
echo ""
cd datasets
echo "Writing to report_extraction_${start_date}.txt"
time python3 extract_all.py > "report_extraction_${start_date}.txt"
echo "== Done at $(date)" >> "report_extraction_${start_date}.txt"
echo ""
echo ""


echo "====================="
echo "== Starting visualization at:"
echo "== $(date)"
echo ""
cd ..
cd visualizations
echo "Writing to report_visualization_${start_date}.txt"
time python3 visualize_all.py > "report_visualization_${start_date}.txt"
echo "== Done at $(date)" >> "report_visualization_${start_date}.txt"
echo ""
echo ""

echo "====================="
echo "== Starting learning at:"
echo "== $(date)"
echo ""
cd ..
cd learning
echo "Writing to report_learning_${start_date}.txt"
time python3 learn_all.py > "report_learning_${start_date}.txt"
echo "== Done at $(date)" >> "report_learning_${start_date}.txt"
echo ""
echo ""


echo "====================="
echo "====== Done at:"
echo "====== $(date)"
