pip install -e requirements.txt

cd third_party/griddly
./build_release.sh && pip install -e python
cd ../..

pip install -e third_party/sample_factory
