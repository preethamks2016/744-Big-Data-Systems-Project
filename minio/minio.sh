wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
sudo mv minio /usr/local/bin/
mkdir ~/minio
minio server ~/minio --console-address :9090
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/mc
mc alias set local http://127.0.0.1:9000 minioadmin minioadmin
mc admin info local
#creating buckets
mc mb local/resnet-images #local is an alias name we set using the above command.
mc cp --recursive ~/mydata/ local/resnet-images/ #source to destination bucket
#install minio python sdk
pip3 install minio
