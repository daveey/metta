# copy to s3
aws s3 ./train_dir sync s3://metta-ai/vast/workspace/metta/train_dir

# copy models from s3
aws s3 sync s3://metta-ai/vast/workspace/metta/train_dir ./train_dir

# train

