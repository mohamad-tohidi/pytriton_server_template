set -e

python3 ./src/server.py &
sleep 10    
python3 ./src/app.py &   

wait
