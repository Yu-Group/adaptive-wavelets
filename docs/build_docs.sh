cd ..
pdoc --html awd --output-dir docs
cp -r docs/awd/* docs/
cp -r docs/awd/*.html docs/
rm -rf docs/awd
cd docs
python3 style_docs.py