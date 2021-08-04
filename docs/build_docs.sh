cd ..
pdoc --html awd --output-dir docs
cp -r docs/awd/* docs/
rm -rf docs/awd
cd docs
python3 style_docs.py