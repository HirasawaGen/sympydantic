# TODO: modified to CICD pipeline
cd ..
echo $PWD
rm -rf test-sympydantic
mkdir test-sympydantic
cd test-sympydantic
uv init -p 3.12
ls -a
uv run main.py
uv add ../sympydantic/