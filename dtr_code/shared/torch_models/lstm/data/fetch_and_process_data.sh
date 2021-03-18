curl https://www.cs.umb.edu/~smimarog/textmining/datasets/r8-train-all-terms.txt > r8-train-all-terms.txt
curl https://www.cs.umb.edu/~smimarog/textmining/datasets/r8-test-all-terms.txt > r8-test-all-terms.txt
if [ -d "train_txt" ]; then rm -r ./train_txt; fi
if [ -d "test_txt" ]; then rm -r ./test_txt; fi
mkdir train_txt
mkdir test_txt

python3 split_data.py