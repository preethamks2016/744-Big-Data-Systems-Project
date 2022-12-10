python3 resnetModel.py  --batch_size=8  --num_workers=0 --output_file='hdf5-batch-8-work-0-thread-0.csv' --num_threads=0
python3 resnetModel.py  --batch_size=8  --num_workers=1 --output_file='hdf5-batch-8-work-1-thread-0.csv' --num_threads=0
python3 resnetModel.py  --batch_size=8  --num_workers=1 --output_file='hdf5-batch-8-work-1-thread-1.csv' --num_threads=1
python3 resnetModel.py  --batch_size=8  --num_workers=2 --output_file='hdf5-batch-8-work-2-thread-2.csv' --num_threads=2
python3 resnetModel.py  --batch_size=8  --num_workers=3 --output_file='hdf5-batch-8-work-3-thread-3.csv' --num_threads=3
python3 resnetModel.py  --batch_size=8  --num_workers=4 --output_file='hdf5-batch-8-work-4-thread-4.csv' --num_threads=4