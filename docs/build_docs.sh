cd ..
pdoc --html adaptive_wavelets --output-dir docs
cp -r docs/adaptive_wavelets/* docs/
rm -rf docs/adaptive_wavelets
cd docs
python3 style_docs.py