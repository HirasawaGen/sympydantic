# TODO: modified to CICD pipeline
project_name=sympydantic
uv build
cd ..
echo $PWD
rm -rf demo-$project_name
mkdir demo-$project_name
cd demo-$project_name
uv init -p 3.12
ls -a
uv run main.py
uv cache clean
cp -r ../$project_name/test .
uv add ../$project_name/
uv run python -m sympydantic
uv add numpy
uv add pytest
uv run pytest