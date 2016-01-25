echo "--------------------"
echo "Embed 10"
python ./code/rnnlm.py -train -embed 10 -hidden 200 -bs 2000 -ch 10000 -lr 0.001 -eps 1e-08 -epochs 2 -vocab ./data/vocab.txt < ./data/en.train
echo "--------------------"
echo "Embed 50"
python ./code/rnnlm.py -train -embed 50 -hidden 200 -bs 2000 -ch 10000 -lr 0.001 -eps 1e-08 -epochs 2 -vocab ./data/vocab.txt < ./data/en.train
echo "--------------------"
echo "Embed 100"
python ./code/rnnlm.py -train -embed 100 -hidden 200 -bs 2000 -ch 10000 -lr 0.001 -eps 1e-08 -epochs 2 -vocab ./data/vocab.txt < ./data/en.train
echo "--------------------"
echo "Embed 500"
python ./code/rnnlm.py -train -embed 500 -hidden 200 -bs 2000 -ch 10000 -lr 0.001 -eps 1e-08 -epochs 2 -vocab ./data/vocab.txt < ./data/en.train
echo "--------------------"
echo "Embed 1000"
python ./code/rnnlm.py -train -embed 1000 -hidden 200 -bs 2000 -ch 10000 -lr 0.001 -eps 1e-08 -epochs 2 -vocab ./data/vocab.txt < ./data/en.train
echo "--------------------"
echo "Embed 5000"
python ./code/rnnlm.py -train -embed 5000 -hidden 200 -bs 2000 -ch 10000 -lr 0.001 -eps 1e-08 -epochs 2 -vocab ./data/vocab.txt < ./data/en.train
echo "--------------------"
echo "Embed 10003"
python ./code/rnnlm.py -train -embed 10003 -hidden 200 -bs 2000 -ch 10000 -lr 0.001 -eps 1e-08 -epochs 2 -vocab ./data/vocab.txt < ./data/en.train
