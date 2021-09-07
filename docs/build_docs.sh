cd ..
pdoc --html awave --output-dir docs
cp -r docs/awave/* docs/
cp -r docs/awave/*.html docs/
rm -rf docs/awave
rm -rf docs/tests
cd docs
python3 style_docs.py